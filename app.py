import spaces  # isort:skip
import contextlib
import gc
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from glob import glob

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import Transform3d

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model import PCAE
from util.dataset_mixamo import (
    BONES_IDX_DICT,
    JOINTS_NUM,
    KINEMATIC_TREE,
    MIXAMO_PREFIX,
    TEMPLATE_PATH,
    Joint,
    get_hips_transform,
)
from util.dataset_mixamo_additional import BONES_IDX_DICT as BONES_IDX_DICT_ADD
from util.dataset_mixamo_additional import JOINTS_NUM as JOINTS_NUM_ADD
from util.dataset_mixamo_additional import KINEMATIC_TREE as KINEMATIC_TREE_ADD
from util.dataset_mixamo_additional import TEMPLATE_PATH as TEMPLATE_PATH_ADD
from util.utils import (
    TimePrints,
    Timing,
    apply_transform,
    fix_random,
    get_normalize_transform,
    load_gs,
    make_archive,
    pose_local_to_global,
    pose_rot_to_global,
    sample_mesh,
    save_gs,
    str2bool,
    str2list,
    to_pose_local,
    to_pose_matrix,
    transform_gs,
)

# Monkey patching to correct the loaded example values from csv
Checkbox_postprocess = gr.Checkbox.postprocess
gr.Checkbox.postprocess = lambda self, value: (
    str2bool(value) if isinstance(value, str) else Checkbox_postprocess(self, value)
)
CheckboxGroup_postprocess = gr.CheckboxGroup.postprocess
gr.CheckboxGroup.postprocess = lambda self, value: (
    list(filter(None, str2list(lambda x: x.lstrip('"').rstrip('"'))(value)))
    if isinstance(value, str)
    else CheckboxGroup_postprocess(self, value)
)


def is_main_thread():
    import threading

    return threading.current_thread() is threading.main_thread()


# Monkey patching gradio to use let gr.Info & gr.Warning also print on console
def _log_message(
    message: str,
    level="info",
    duration: float | None = 10,
    visible: bool = True,
    *args,
    **xargs,
):
    from gradio.context import LocalContext

    if level in ("info", "success"):
        print(message)
    elif level == "warning":
        warnings.warn(message)

    blocks = LocalContext.blocks.get()
    event_id = LocalContext.event_id.get()
    if blocks is not None and event_id is not None:
        # Function called outside of Gradio if blocks is None
        # Or from /api/predict if event_id is None
        blocks._queue.log_message(
            event_id=event_id, log=message, level=level, duration=duration, visible=visible, *args, **xargs
        )


import gradio.helpers

gradio.helpers.log_message = _log_message


cmap = matplotlib.colormaps.get_cmap("plasma")


@dataclass()
class DB:
    mesh: trimesh.Trimesh = None
    gs: torch.Tensor = None
    gs_rest: torch.Tensor = None
    is_mesh: bool = None
    sample_mask: np.ndarray = None
    verts: torch.Tensor = None
    verts_normal: torch.Tensor = None
    faces: np.ndarray = None
    pts: torch.Tensor = None
    pts_normal: torch.Tensor = None
    global_transform: Transform3d = None

    output_dir: str = None
    joints_coarse_path: str = None
    normed_path: str = None
    sample_path: str = None
    bw_path: str = None
    joints_path: str = None
    rest_lbs_path: str = None
    rest_vis_path: str = None
    anim_path: str = None
    anim_vis_path: str = None

    bw: torch.Tensor = None
    joints: torch.Tensor = None
    joints_tail: torch.Tensor = None
    pose: torch.Tensor = None

    def clear(self):
        for k in self.__dict__:
            self.__dict__[k] = None
        return self


def clear(db: DB = None):
    if db is not None:
        db.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")
    return db


def get_conflict_mask(dominant_idx: torch.Tensor, fn1, fn2, bones_idx_dict: dict[str, int]):
    """
    Args:
        dominant_idx: (B, N) int
        fn1: fn(str) -> bool. Values of `bones_idx_dict` will be appended to `idx1` if `fn1(key) == True`
        fn2: fn(str) -> bool. Values of `bones_idx_dict` will be appended to `idx2` if `fn2(key) == True`
    Returns:
        mask: (B, N, `len(bones_idx_dict)`) Boolean mask.
        `mask[..., i] == True` if `dominant_idx[...] in idx1` and `i in idx2`.
    """
    idx1 = torch.tensor(tuple(v for k, v in bones_idx_dict.items() if fn1(k))).to(dominant_idx)
    idx2 = torch.tensor(tuple(v for k, v in bones_idx_dict.items() if fn2(k))).to(dominant_idx)
    mask2 = torch.isin(torch.arange(len(bones_idx_dict), device=dominant_idx.device), idx2)
    pts_mask1 = torch.isin(dominant_idx, idx1).unsqueeze(-1).expand([*dominant_idx.shape, len(bones_idx_dict)]).clone()
    pts_mask1[:, :, ~mask2] = False
    return pts_mask1


def bw_post_process(
    bw: torch.Tensor,
    bones_idx_dict: dict[str, int],
    above_head_mask: torch.Tensor = None,
    above_ear_mask_left: torch.Tensor = None,
    above_ear_mask_right: torch.Tensor = None,
    tail_mask: torch.Tensor = None,
    no_fingers=False,
):

    def _edit_bw(
        mask: torch.Tensor, bw: torch.Tensor, bones_idx_dict: dict[str, int], target_bone_name: str, value=1e5
    ):
        mask = mask.unsqueeze(-1).tile(bw.shape[-1])
        head_bone_mask = torch.zeros_like(mask)
        head_bone_mask[..., bones_idx_dict[f"{MIXAMO_PREFIX}{target_bone_name}"]] = True
        mask &= head_bone_mask
        bw[mask] = value
        return bw

    """bw: (B, N, `len(bones_idx_dict)`)"""
    assert bw.shape[-1] == len(bones_idx_dict)

    if above_head_mask is not None and all("Ear" not in b for b in bones_idx_dict):
        bw = _edit_bw(above_head_mask, bw, bones_idx_dict, "Head")
    if any("Ear" in b for b in bones_idx_dict):
        if above_ear_mask_left is not None:
            bw = _edit_bw(above_ear_mask_left, bw, bones_idx_dict, "LRabbitEar2", value=1.0)
        if above_ear_mask_right is not None:
            bw = _edit_bw(above_ear_mask_right, bw, bones_idx_dict, "RRabbitEar2", value=1.0)
    if tail_mask is not None and all("Tail" not in b for b in bones_idx_dict):
        bw = _edit_bw(tail_mask, bw, bones_idx_dict, "Spine")

    hands = {"Left", "Right"}
    fingers = {"Thumb", "Index", "Middle", "Ring", "Pinky"}
    if no_fingers:
        dominant_idx = torch.argmax(bw, dim=-1)
        for hand in hands:
            bw[
                get_conflict_mask(
                    dominant_idx,
                    lambda k: hand in k and any(x in k for x in fingers),
                    lambda k: k.endswith(f"{hand}Hand"),
                    bones_idx_dict,
                )
            ] = 1e5
        bw[get_conflict_mask(dominant_idx, lambda k: True, lambda k: any(x in k for x in fingers), bones_idx_dict)] = 0

    # Refine points dominated by conflict (mutually exclusive) joints (left & right limbs, different fingers)
    dominant_idx = torch.argmax(bw, dim=-1)
    if not no_fingers:
        for hand in hands:
            other_hand = next(iter((hands - {hand})))
            bw[get_conflict_mask(dominant_idx, lambda k: hand in k, lambda k: other_hand in k, bones_idx_dict)] = 0
            for finger in fingers:
                other_fingers = fingers - {finger}
                bw[
                    get_conflict_mask(
                        dominant_idx,
                        lambda k: hand in k and finger in k,
                        lambda k: hand in k and any(x in k for x in other_fingers),
                        bones_idx_dict,
                    )
                ] = 0

    conflict_sets = (
        {
            ("Neck", "Head", "RabbitEar"),
            ("LeftLeg", "LeftFoot", "LeftToeBase"),
            ("RightLeg", "RightFoot", "RightToeBase"),
            (
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
                "LeftHandThumb1",
                "LeftHandThumb2",
                "LeftHandThumb3",
                "LeftHandIndex1",
                "LeftHandIndex2",
                "LeftHandIndex3",
                "LeftHandMiddle1",
                "LeftHandMiddle2",
                "LeftHandMiddle3",
                "LeftHandRing1",
                "LeftHandRing2",
                "LeftHandRing3",
                "LeftHandPinky1",
                "LeftHandPinky2",
                "LeftHandPinky3",
            ),
            (
                "RightArm",
                "RightForeArm",
                "RightHand",
                "RightHandThumb1",
                "RightHandThumb2",
                "RightHandThumb3",
                "RightHandIndex1",
                "RightHandIndex2",
                "RightHandIndex3",
                "RightHandMiddle1",
                "RightHandMiddle2",
                "RightHandMiddle3",
                "RightHandRing1",
                "RightHandRing2",
                "RightHandRing3",
                "RightHandPinky1",
                "RightHandPinky2",
                "RightHandPinky3",
            ),
        },
        {
            (
                "Hips",
                "Spine",
                "Spine1",
                "Spine2",
                "Neck",
                "Head",
                "LeftShoulder",
                "RightShoulder",
                "RightUpLeg",
                "RightLeg",
                "RightFoot",
                "RightToeBase",
                "LeftUpLeg",
                "LeftLeg",
                "LeftFoot",
                "LeftToeBase",
                "RabbitEar",
                "FoxTail",
            ),
            (
                "RightForeArm",
                "RightHand",
                "RightHandThumb1",
                "RightHandThumb2",
                "RightHandThumb3",
                "RightHandIndex1",
                "RightHandIndex2",
                "RightHandIndex3",
                "RightHandMiddle1",
                "RightHandMiddle2",
                "RightHandMiddle3",
                "RightHandRing1",
                "RightHandRing2",
                "RightHandRing3",
                "RightHandPinky1",
                "RightHandPinky2",
                "RightHandPinky3",
                "LeftForeArm",
                "LeftHand",
                "LeftHandThumb1",
                "LeftHandThumb2",
                "LeftHandThumb3",
                "LeftHandIndex1",
                "LeftHandIndex2",
                "LeftHandIndex3",
                "LeftHandMiddle1",
                "LeftHandMiddle2",
                "LeftHandMiddle3",
                "LeftHandRing1",
                "LeftHandRing2",
                "LeftHandRing3",
                "LeftHandPinky1",
                "LeftHandPinky2",
                "LeftHandPinky3",
            ),
        },
        {("LRabbitEar",), ("RRabbitEar",)},
        {("FoxTail",), ("Neck", "Head", "RabbitEar"), ("RightFoot", "RightToeBase", "LeftFoot", "LeftToeBase")},
    )
    for conflict_parts in conflict_sets:
        for part in conflict_parts:
            other_parts = conflict_parts - {part}
            bw[
                get_conflict_mask(
                    dominant_idx,
                    lambda k: any(x in k for x in part),
                    lambda k: any(any(x in k for x in p) for p in other_parts),
                    bones_idx_dict,
                )
            ] = 0

    bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)
    # Only keep weights from the largest-weighted joints
    # bw[bw < 1e-4] = 0
    joints_per_point = 4
    thresholds = torch.topk(bw, k=joints_per_point, dim=-1, sorted=True).values[..., -1:]
    bw[bw < thresholds] = 0
    return bw


def reorganize_bone_data_(
    bone_data: torch.Tensor,
    bones_idx_dict: dict[str, int],
    template_dict: dict[str, int],
    template_data: torch.Tensor = None,
    is_pose_global=False,
    kinematic_tree: Joint = None,
):
    """
    Args:
        bone_data: [..., K, D]
        bones_idx_dict: {bone_name: bone_idx}, len == K
    Returns:
        [..., `len(template_dict)`, D]
    """
    module = torch if isinstance(bone_data, torch.Tensor) else np
    assert bone_data.shape[-2] == len(bones_idx_dict), "Unmatched bone data"
    if template_data is None:
        new_data = module.zeros(bone_data.shape[:-2] + (len(template_dict), bone_data.shape[-1]), dtype=bone_data.dtype)
        if module is torch:
            new_data = new_data.to(bone_data)
    else:
        new_data = template_data.clone() if module is torch else template_data.copy()
    for bone_name, i in template_dict.items():
        if bone_name in bones_idx_dict:
            new_data[..., i, :] = bone_data[..., bones_idx_dict[bone_name], :]
        elif is_pose_global:
            assert kinematic_tree is not None
            valid_parent = kinematic_tree[bone_name].get_first_valid_parent(bones_idx_dict)
            if valid_parent is not None:
                new_data[..., i, :] = bone_data[..., bones_idx_dict[valid_parent.name], :]
    return new_data


def vis_weights(verts: np.ndarray, weights: np.ndarray, faces: np.ndarray, vis_bone_index: int):
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()
    if len(verts.shape) == 3:
        verts = verts[0]
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    if len(weights.shape) == 3:
        weights = weights[0]
    assert all(x is None or isinstance(x, np.ndarray) for x in (verts, weights, faces))
    assert all(x is None or len(x.shape) == 2 for x in (verts, weights, faces))
    assert verts.shape[0] == weights.shape[0]
    assert faces is None or verts.shape[1] == faces.shape[1] == 3
    # assert weights.shape[1] == len(BONES_IDX_DICT)
    colors = cmap(weights[:, vis_bone_index])[:, :3]
    if faces is None:
        mesh = trimesh.PointCloud(verts, process=False, colors=colors)
    else:
        mesh = trimesh.Trimesh(verts, faces, process=False, maintain_order=True, vertex_colors=colors)
    extent = verts.max() - verts.min()
    axis = trimesh.creation.axis(origin_size=extent * 0.02)
    # return trimesh.util.concatenate([mesh, axis])
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="mesh")
    scene.add_geometry(axis, geom_name="axis")
    return scene


def vis_joints(
    verts: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    bones_idx_dict: dict[str, int],
    vis_bone_index: int = None,
    mark="l",
):
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()
    if len(verts.shape) == 3:
        verts = verts[0]
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    if len(joints.shape) == 3:
        joints = joints[0]
    assert all(x is None or isinstance(x, np.ndarray) for x in (verts, joints, faces))
    assert all(x is None or len(x.shape) == 2 for x in (verts, joints, faces))
    assert faces is None or verts.shape[1] == joints.shape[1] == faces.shape[1] == 3
    assert joints.shape[0] == len(bones_idx_dict)
    if faces is None:
        if vis_bone_index is None:
            vis_bone_index = 0
        colors = cmap(np.linalg.norm(joints[vis_bone_index] - verts, axis=-1))[:, :3]
    else:
        normals = trimesh.Trimesh(verts, faces, process=False, maintain_order=True).vertex_normals
        colors = (normals + 1) / 2
    if faces is None:
        mesh = trimesh.PointCloud(verts, process=False, colors=colors)
        mark = "o"
    else:
        mesh = trimesh.Trimesh(verts, faces, process=False, maintain_order=True, vertex_colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="mesh")
    # markers = []
    extent = verts.max() - verts.min()
    for joint, joint_name in zip(joints, bones_idx_dict):
        if "Hand" in joint_name:
            scaling = 0.01
        elif "Hips" in joint_name or "Spine" in joint_name:
            scaling = 0.06
        else:
            scaling = 0.04
        # marker = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        if mark == "l":
            marker = trimesh.creation.capsule(height=extent * scaling * 5, radius=extent * scaling * 0.2)
        elif mark == "o":
            marker = trimesh.creation.icosphere(radius=extent * scaling)
        marker.vertices = marker.vertices + joint
        # markers.append(marker)
        scene.add_geometry(marker, geom_name=joint_name)
    axis = trimesh.creation.axis(origin_size=extent * 0.02)
    # mesh = trimesh.util.concatenate([mesh, *markers, axis])
    scene.add_geometry(axis, geom_name="axis")
    return scene


def change_Model3D(value: str = None, display_mode="solid", is_pc=False):
    if is_pc:
        display_mode = "point_cloud"
    return gr.Model3D(value=value, display_mode=display_mode)


def pc2visible(pc: trimesh.PointCloud):
    """
    Gradio now only shows mesh vertices in `obj` when `display_mode="point_cloud"`.
    To display pure point clouds, we need to add "fake faces".
    """
    if not isinstance(pc, trimesh.PointCloud):
        return pc
    n = pc.shape[0]
    faces_ = np.split(np.arange(n), [n // 3, n // 3 * 2])
    faces = np.zeros([max(map(len, faces_)), 3])
    for i, x in enumerate(faces_):
        faces[: x.shape[0], i] = x
    mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=pc.vertices, faces=faces, vertex_colors=pc.colors, process=False)
    return mesh


def ply2visible(ply_path: str, is_gs=False):
    """
    Gradio now reads and renders `ply` as gsplats instead of mesh.
    To display `ply` as mesh, we need to convert it to other format.
    """
    if not isinstance(ply_path, str):
        return ply_path
    if not ply_path.endswith(".ply"):
        ply_path_ = f"{os.path.splitext(ply_path)[0]}.ply"
        if os.path.isfile(ply_path_):
            ply_path = ply_path_
    with TimePrints():
        print(f"Get input: {ply_path}")

    if not ply_path.endswith(".ply") or is_gs:
        return change_Model3D(ply_path, is_pc=False)
    with contextlib.suppress(Exception):
        load_gs(ply_path)
        gr.Warning("The input file seems to be Gaussian Splats, enable 'Input is GS' to display it")
    mesh = trimesh.load(ply_path, process=False, maintain_order=True)
    is_pc = isinstance(mesh, trimesh.PointCloud)
    new_path = f"{os.path.splitext(ply_path)[0]}.glb"
    mesh.export(new_path)
    return change_Model3D(new_path, is_pc=is_pc)


def get_masked_mesh(mesh: trimesh.Trimesh, mask: np.ndarray):
    if mask is None:
        return mesh
    mesh = mesh.copy()
    if isinstance(mesh, trimesh.PointCloud):
        mesh.vertices = mesh.vertices[mask]
        mesh.colors = mesh.colors[mask]
    else:
        mesh.update_vertices(mask)
    return mesh


def prepare_input(input_path: str, is_gs=False, opacity_threshold=0.0, db: DB = None, export_temp=False):
    if not (input_path and os.path.isfile(input_path)):
        raise gr.Error(f"Input file not found: '{input_path}', please re-upload the file")

    ply_path = f"{os.path.splitext(input_path)[0]}.ply"
    if os.path.isfile(ply_path):
        input_path = ply_path
    print(f"{input_path=}")

    if is_gs:
        if not input_path.endswith(".ply"):
            raise gr.Error("Input must be a `.ply` file for Gaussian Splats")
        try:
            gaussians = load_gs(input_path)
            db.gs = gaussians
        except:
            raise gr.Error("Fail to load the input file as Gaussian Splats")
        xyz, opacities, scales, rots, shs = gaussians.split((3, 1, 3, 4, 3), dim=-1)
        verts = xyz.numpy().astype(np.float32)
        sample_mask = (opacities >= opacity_threshold).squeeze(-1).numpy()
        assert sample_mask.any(), "No solid points"
        colors = shs.numpy().astype(np.float32)
        faces = None
        mesh = trimesh.PointCloud(verts, colors=colors, process=False)
        # mesh.export("input.ply")
    else:
        mesh: trimesh.Trimesh = trimesh.load(input_path, force="mesh")
        verts = np.array(mesh.vertices).astype(np.float32)
        sample_mask = None
        if isinstance(mesh, trimesh.PointCloud):
            faces = None
        else:
            verts_normal = np.array(mesh.vertex_normals).astype(np.float32)
            faces = np.array(mesh.faces)
    is_mesh = faces is not None
    pts = sample_mesh(get_masked_mesh(mesh, sample_mask), N, get_normals=is_mesh).astype(np.float32)
    pts = torch.from_numpy(pts).unsqueeze(0)
    verts = torch.from_numpy(verts).unsqueeze(0)
    if is_mesh:
        verts_normal = torch.from_numpy(verts_normal).unsqueeze(0)
        pts, pts_normal = torch.chunk(pts, 2, dim=-1)
    else:
        verts_normal = None
        pts_normal = None

    db.mesh = mesh
    db.is_mesh = is_mesh
    db.sample_mask = sample_mask
    db.verts = verts
    db.verts_normal = verts_normal
    db.faces = faces
    db.pts = pts
    db.pts_normal = pts_normal

    if export_temp:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(output_dir, exist_ok=True)
    db.output_dir = output_dir
    db.joints_coarse_path = os.path.join(output_dir, "joints_coarse.glb")
    db.normed_path = os.path.join(output_dir, f"normed{os.path.splitext(input_path)[-1]}")
    db.sample_path = os.path.join(output_dir, "sample.glb")
    db.bw_path = os.path.join(output_dir, "bw.glb")
    db.joints_path = os.path.join(output_dir, "joints.glb")
    db.rest_lbs_path = os.path.join(output_dir, f"rest_lbs.{'ply' if is_gs else 'glb'}")
    db.rest_vis_path = os.path.join(output_dir, "rest.glb")
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    db.anim_path = os.path.join(output_dir, f"{input_filename}.{'blend' if is_gs else 'fbx'}")
    db.anim_vis_path = os.path.join(output_dir, f"{input_filename}.glb")

    return {state: db}


@spaces.GPU
@torch.no_grad()
def model_forward_coarse(pts: torch.Tensor) -> torch.Tensor:
    pts = pts.to(device)
    joints = model_coarse(pts).joints
    return joints.cpu()


def preprocess(db: DB):
    mesh = db.mesh
    pts = db.pts
    pts_normal = db.pts_normal
    verts = db.verts
    verts_normal = db.verts_normal

    # Transform to Hips coordinates
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=True)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)
    # if is_mesh:
    #     pts_normal = F.normalize(norm.transform_normals(pts_normal), dim=-1)
    #     verts_normal = F.normalize(norm.transform_normals(verts_normal), dim=-1)
    with Timing(msg="Joints localization done in", print_fn=gr.Info):
        joints = model_forward_coarse(pts)
    joints, joints_tail = joints[..., :3], joints[..., 3:]
    hips = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Hips"]]
    rightupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightUpLeg"]]
    leftupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftUpLeg"]]
    # rotate = Transform3d()
    rotate = Transform3d(matrix=get_hips_transform(hips, rightupleg, leftupleg).transpose(-1, -2))
    global_transform = norm.compose(rotate)
    pts = rotate.transform_points(pts)
    verts = rotate.transform_points(verts)
    if db.is_mesh:
        pts_normal = F.normalize(rotate.transform_normals(pts_normal), dim=-1)
        verts_normal = F.normalize(rotate.transform_normals(verts_normal), dim=-1)
    vis_joints(
        mesh.vertices, norm.inverse().transform_points(joints), db.faces, bones_idx_dict=BONES_IDX_DICT, mark="o"
    ).export(db.joints_coarse_path)
    mesh.vertices = verts.squeeze(0).cpu().numpy()
    if db.gs is not None:
        db.gs = transform_gs(db.gs, global_transform)
        save_gs(db.gs, db.normed_path)
    else:
        mesh.export(db.normed_path)

    if hands_resample_ratio > 0:
        joints_tail_hips = rotate.transform_points(joints_tail).squeeze(0).cpu().numpy()
        hands_centers = [
            joints_tail_hips[BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftHand"]],
            joints_tail_hips[BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightHand"]],
        ]
        pts = sample_mesh(
            get_masked_mesh(mesh, db.sample_mask),
            N,
            get_normals=db.is_mesh,
            attn_ratio=hands_resample_ratio,
            attn_centers=hands_centers,
            attn_geo_ratio=geo_resample_ratio,
        ).astype(np.float32)
    else:
        pts = pts.squeeze(0).cpu().numpy()
        pts_normal = pts_normal.squeeze(0).cpu().numpy()
        pts = np.concatenate([pts, pts_normal], axis=-1)

    try:
        if db.is_mesh:
            if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                try:
                    visual = mesh.visual.to_color()
                except Exception:
                    visual = None
            elif isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                visual = mesh.visual
            else:
                visual = None
            pts_colors = (
                None
                if visual is None or visual.vertex_colors.shape[0] != mesh.vertices.shape[0]
                else visual.vertex_colors[mesh.kdtree.query(pts[..., :3])[1]]
            )
        elif mesh.colors.shape[0] > 0:
            pts_colors = mesh.colors[mesh.kdtree.query(pts)[1]]
        else:
            pts_colors = None
    except Exception:
        pts_colors = None
    pts_vis = trimesh.PointCloud(vertices=pts[..., :3], colors=pts_colors)
    pts_vis.export(db.sample_path)
    pts = torch.from_numpy(pts).unsqueeze(0)
    if db.is_mesh:
        pts, pts_normal = torch.chunk(pts, 2, dim=-1)

    db.verts = verts
    db.verts_normal = verts_normal
    db.pts = pts
    db.pts_normal = pts_normal
    db.global_transform = global_transform

    return {
        output_joints_coarse: change_Model3D(db.joints_coarse_path, display_mode="wireframe", is_pc=not db.is_mesh),
        output_normed_input: change_Model3D(db.normed_path, is_pc=not db.is_mesh),
        output_sample: change_Model3D(db.sample_path, is_pc=True),
        state: db,
    }


# @spaces.GPU  # always lead to "GPU task aborted"
@torch.no_grad()
def model_forward_bw(
    verts: torch.Tensor, verts_normal: torch.Tensor, pts: torch.Tensor, pts_normal: torch.Tensor, input_normal: bool
) -> torch.Tensor:
    device = next(model_bw.parameters()).device
    pts = pts.to(device)
    pts_normal = None if pts_normal is None else pts_normal.to(device)

    CHUNK = 100000  # present OOM for high-res models
    bw = []
    verts_chunks = torch.split(verts, CHUNK, dim=-2)
    verts_normal_chunks = (
        ([None] * len(verts_chunks)) if verts_normal is None else torch.split(verts_normal, CHUNK, dim=-2)
    )
    for verts_, verts_normal_ in zip(verts_chunks, verts_normal_chunks):
        verts_ = verts_.to(device)
        verts_normal_ = None if verts_normal_ is None else verts_normal_.to(device)
        bw_ = model_bw(pts, verts_).bw
        if input_normal:
            bw_normal = model_bw_normal(
                torch.cat([pts, pts_normal], dim=-1), torch.cat([verts_, verts_normal_], dim=-1)
            ).bw
            mask = get_conflict_mask(
                torch.argmax(bw_, dim=-1),
                lambda k: True,
                lambda k: any(x in k for x in ("Spine", "Shoulder", "Arm")),
                bones_idx_dict_bw,
            )
            bw_normal[mask] = bw_[mask]
            bw_ = bw_normal
        bw.append(bw_)
    bw = torch.cat(bw, dim=-2)

    return bw.cpu()


@spaces.GPU
@torch.no_grad()
def model_forward_bones(pts: torch.Tensor) -> tuple[torch.Tensor]:
    pts = pts.to(device)

    joints = model_joints.forward(pts).joints
    if joints_additional:
        joints_add = model_joints_add(pts).joints
        joints = reorganize_bone_data_(joints, BONES_IDX_DICT, bones_idx_dict_joints, template_data=joints_add)

    if model_pose.pose_input_joints:
        joints_ = joints.clone()
        if joints_additional:
            joints_ = reorganize_bone_data_(joints_, bones_idx_dict_joints, BONES_IDX_DICT)
    else:
        joints_ = None
    pose = model_pose(pts, joints=joints_).pose_trans

    return joints.cpu(), pose.cpu()


def infer(input_normal: bool, db: DB):
    pts = db.pts
    pts_normal = db.pts_normal
    verts = db.verts
    verts_normal = db.verts_normal

    if input_normal and not db.is_mesh:
        raise gr.Error("Normals are not available for point clouds or Gaussian Splats")

    # Norm data & infer the main model
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=False)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)
    # if is_mesh:
    #     pts_normal = F.normalize(norm.transform_normals(pts_normal), dim=-1)
    #     verts_normal = F.normalize(norm.transform_normals(verts_normal), dim=-1)
    if db.gs is not None:
        db.gs = transform_gs(db.gs, norm)

    with Timing(msg="Model inference done in", print_fn=gr.Info):
        bw = model_forward_bw(verts, verts_normal, pts, pts_normal, input_normal)
        joints, pose = model_forward_bones(pts)

    db.mesh.vertices = verts.squeeze(0).cpu().numpy()
    db.pts = pts
    db.verts = verts
    db.bw = bw
    db.joints = joints
    db.pose = pose
    db.global_transform = db.global_transform.compose(norm)
    return {state: db}


def vis(bw_fix: bool, bw_vis_bone: str, no_fingers: bool, db: DB):
    verts = db.verts
    bw = db.bw
    joints = db.joints
    pose = db.pose

    def _get_bone_loc(
        bone_xyz: torch.Tensor, bone_name: str, dim="y", type="tail", bone_idx_dict: dict[str, int] = bones_idx_dict_bw
    ):
        bone = bone_xyz[:, [bone_idx_dict[f"{MIXAMO_PREFIX}{bone_name}"]], :]
        bone_head, bone_tail = bone[..., :3], bone[..., 3:]

        if dim == "x":
            dim_idx = 0
        elif dim == "y":
            dim_idx = 1
        elif dim == "z":
            dim_idx = 2
        else:
            raise ValueError(f"Invalid dim: {dim}")

        if type == "head":
            bone = bone_head[..., dim_idx]
        elif type == "tail":
            bone = bone_tail[..., dim_idx]
        elif type == "center":
            bone = 0.5 * (bone_head[..., dim_idx] + bone_tail[..., dim_idx])
        else:
            raise ValueError(f"Invalid type: {type}")

        return bone

    if bw_fix:
        bw = bw_post_process(
            bw,
            bones_idx_dict=bones_idx_dict_bw,
            above_head_mask=verts[..., 1] >= _get_bone_loc(joints, "Head", "y", "tail"),
            above_ear_mask_left=(
                (verts[..., 1] >= _get_bone_loc(joints, "LRabbitEar2", "y", "tail"))
                & (verts[..., 0] > _get_bone_loc(joints, "Head", "x", "tail"))
                if bw_additional
                else None
            ),
            above_ear_mask_right=(
                (verts[..., 1] >= _get_bone_loc(joints, "RRabbitEar2", "y", "tail"))
                & (verts[..., 0] < _get_bone_loc(joints, "Head", "x", "tail"))
                if bw_additional
                else None
            ),
            # tail_mask=(verts[..., 2] <= (_get_bone_loc(joints, "Hips", "z", "head") - 0.2))
            # & ((verts[..., 0] - _get_bone_loc(joints, "Hips", "x", "head")).abs() <= 0.1)
            # & ((verts[..., 1] - _get_bone_loc(joints, "Hips", "y", "head")).abs() <= 0.15),
            no_fingers=no_fingers,
        )

        # Transform back to the input coordinates
        # transform_inv = db.global_transform.inverse()
        # data_ = {}
        # data_["joints_head"] = transform_inv.transform_points(joints[..., :3]).squeeze(0).cpu().numpy()
        # data_["joints_tail"] = transform_inv.transform_points(joints[..., 3:]).squeeze(0).cpu().numpy()
        # # data_["bw"] = bw.squeeze(0).cpu().numpy()
        # bones_idx_dict = dict(bones_idx_dict_joints)
        # from app_blender import remove_fingers_from_data
        # data_["joints_head"] = remove_fingers_from_data(data_["joints_head"], bones_idx_dict)
        # data_["joints_tail"] = remove_fingers_from_data(data_["joints_tail"], bones_idx_dict)
        # # data_["bw"] = remove_fingers_from_data(data_["bw"].T, bones_idx_dict, is_bw=True).T
        # np.savez(os.path.join("data/rignet/output", os.path.basename(db.anim_path).replace(".fbx", ".npz")), **data_)

    bw = bw.squeeze(0).cpu().numpy()
    verts = verts.squeeze(0).cpu().numpy()
    vis_weights(verts, bw, db.faces, vis_bone_index=bones_idx_dict_bw[f"{MIXAMO_PREFIX}{bw_vis_bone}"]).export(
        db.bw_path
    )
    joints, joints_tail = joints.squeeze(0)[..., :3].cpu().numpy(), joints.squeeze(0)[..., 3:].cpu().numpy()
    vis_joints(verts, joints, db.faces, bones_idx_dict=bones_idx_dict_joints).export(db.joints_path)

    if pose is not None:
        if joints_additional:
            pose = reorganize_bone_data_(
                pose,
                BONES_IDX_DICT,
                bones_idx_dict_joints,
                is_pose_global=True,
                kinematic_tree=KINEMATIC_TREE_ADD,
            )
            if not bw_additional:
                bw = reorganize_bone_data_(bw.T, BONES_IDX_DICT, bones_idx_dict_joints).T

        if "local" in model_pose.pose_mode:
            pose = to_pose_local(pose, input_mode=model_pose.pose_mode, return_quat=False)
            pose, _ = pose_local_to_global(
                pose,
                db.joints[..., :3],
                torch.tensor((KINEMATIC_TREE_ADD if joints_additional else KINEMATIC_TREE).parent_indices),
                relative_to_source=True,
            )
        elif model_pose.pose_mode in ("quat", "ortho6d"):
            pose, _ = pose_rot_to_global(
                pose,
                db.joints[..., :3],
                torch.tensor((KINEMATIC_TREE_ADD if joints_additional else KINEMATIC_TREE).parent_indices),
            )
        else:
            pose = to_pose_matrix(pose, input_mode=model_pose.pose_mode, source=db.joints[..., :3])

        pose[..., 0, :, :] = torch.eye(4)
        pose = pose.squeeze(0).cpu().numpy()
        lbs_transform = np.einsum("kij,nk->nij", pose, bw)
        if db.gs is None:
            rest_joints = apply_transform(joints, pose)
            vis_joints(
                apply_transform(verts, lbs_transform), rest_joints, db.faces, bones_idx_dict=bones_idx_dict_joints
            ).export(db.rest_lbs_path)
        else:
            db.gs_rest = transform_gs(db.gs, lbs_transform)
            save_gs(db.gs_rest, db.rest_lbs_path)

    db.verts = verts
    db.bw = bw
    db.joints = joints
    db.joints_tail = joints_tail
    db.pose = pose

    return {
        output_joints: change_Model3D(db.joints_path, display_mode="wireframe", is_pc=not db.is_mesh),
        output_bw: change_Model3D(db.bw_path, is_pc=not db.is_mesh),
        output_rest_lbs: change_Model3D(db.rest_lbs_path, is_pc=not db.is_mesh),
        state: db,
    }


def get_pose_ignore_list(pose: str = None, pose_parts: list[str] = None):
    kw_list: list[str] = ["Hips", "Ear", "Tail"]
    if pose:
        if pose == "T-pose":
            kw_list.extend(
                [
                    "Spine",
                    "Neck",
                    "Head",
                    "Shoulder",
                    "Arm",
                    "ForeArm",
                    "Hand",
                    "UpLeg",
                    "Leg",
                    "Foot",
                    "ToeBase",
                ]
            )  # all
        elif pose == "A-pose":
            kw_list.extend(
                [
                    "Spine",
                    "Neck",
                    "Head",
                    "ForeArm",
                    "Hand",
                    "UpLeg",
                    "Leg",
                    "Foot",
                    "ToeBase",
                ]
            )  # except for Shoulder & Arm
        elif pose == "大-pose":
            kw_list.extend(
                [
                    "Spine",
                    "Neck",
                    "Head",
                    "Shoulder",
                    "Arm",
                    "ForeArm",
                    "Hand",
                    "Leg",
                    "Foot",
                    "ToeBase",
                ]
            )  # except for UpLeg
    if pose_parts:
        if "Fingers" in pose_parts:
            kw_list.extend(["Thumb", "Index", "Middle", "Ring", "Pinky"])
        if "Arms" in pose_parts:
            kw_list.extend(["Arm", "ForeArm", "Hand"])
        if "Legs" in pose_parts:
            kw_list.extend(["UpLeg", "Leg", "Foot", "ToeBase"])
        if "Head" in pose_parts:
            kw_list.extend(["Head", "Neck"])
    return kw_list


def vis_blender(
    reset_to_rest: bool,
    remove_fingers: bool,
    rest_pose_type: str,
    ignore_pose_parts: list[str],
    animation_file: str,
    retarget: bool,
    inplace: bool,
    db: DB,
):
    if any(x is None for x in (db.mesh, db.joints, db.joints_tail, db.bw)):
        raise gr.Error("Run the inference first")

    if db.gs is not None:
        gr.Warning("It can take quite a long time to import and rig Gaussian Splats in Blender. Please wait patiently.")
        if isinstance(db.gs, torch.Tensor):
            db.gs = db.gs.numpy()
        if db.gs_rest is not None and isinstance(db.gs_rest, torch.Tensor):
            db.gs_rest = db.gs_rest.numpy()
    template_path = TEMPLATE_PATH_ADD if joints_additional else TEMPLATE_PATH

    data = dict(
        mesh=db.mesh,
        gs=db.gs_rest if reset_to_rest else db.gs,
        joints=db.joints,
        joints_tail=db.joints_tail,
        bw=db.bw,
        pose=db.pose,
        bones_idx_dict=dict(bones_idx_dict_joints),
        pose_ignore_list=get_pose_ignore_list(rest_pose_type, ignore_pose_parts),
    )
    if animation_file is not None:
        if not os.path.isfile(animation_file):
            raise gr.Error(f"Animation file {animation_file} does not exist")
        if not reset_to_rest:
            gr.Warning(
                "'Reset to Rest' is not enabled, so the animation may be incorrect if the input is not in T-pose"
            )

    if is_main_thread():
        from argparse import Namespace

        from app_blender import main

        main(
            Namespace(
                input_path=data,
                output_path=db.anim_path,
                template_path=template_path,
                keep_raw=False,
                rest_path=db.rest_vis_path if db.is_mesh else None,
                pose_local=False,
                reset_to_rest=reset_to_rest,
                remove_fingers=remove_fingers,
                animation_path=animation_file,
                retarget=retarget,
                inplace=inplace,
            )
        )
    else:
        # Directly call bpy here causes crash, because Blender does not support modifying data in child threads
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, **data)
            cmd = f"python app_blender.py --input_path '{f.name}' --output_path '{os.path.abspath(db.anim_path)}'"
            cmd += f" --template_path '{os.path.abspath(template_path)}'"
            if db.is_mesh:
                cmd += f" --rest_path '{os.path.abspath(db.rest_vis_path)}'"
            # if "local" in model_pose.pose_mode:
            #     cmd += " --pose_local"
            if reset_to_rest:
                cmd += " --reset_to_rest"
            if remove_fingers:
                cmd += " --remove_fingers"
            if animation_file is not None:
                cmd += f" --animation_path '{os.path.abspath(animation_file)}'"
                if retarget:
                    cmd += " --retarget"
                if inplace:
                    cmd += " --inplace"
            cmd += " > /dev/null 2>&1"
            # print(cmd)
            os.system(cmd)

    print(f"Output animatable model: '{db.anim_path}'")

    if db.is_mesh and db.anim_path.endswith(".fbx") and os.path.isfile(db.anim_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            # https://github.com/facebookincubator/FBX2glTF
            fbx2glb_path = "util/FBX2glTF"
            assert os.path.isfile(fbx2glb_path), f"'{fbx2glb_path}' not found"
            fbx2glb_cmd = f"{fbx2glb_path} --binary --keep-attribute auto --fbx-temp-dir '{tmpdir}' --input '{os.path.abspath(db.anim_path)}' --output '{os.path.abspath(db.anim_vis_path)}'"
            fbx2glb_cmd += " > /dev/null 2>&1"
            os.system(fbx2glb_cmd)
            print(f"Output visualization: '{db.anim_vis_path}'")
    else:
        db.rest_vis_path = None
        db.anim_vis_path = None

    anim_path = db.anim_path
    if os.path.isfile(db.anim_path):
        size_mb = os.path.getsize(db.anim_path) / (1024**2)
        if size_mb > 50:
            gr.Info(f"Animation file is too large ({size_mb:.2f}MB), compressing it")
            compressed_path = f"{os.path.splitext(db.anim_path)[0]}.zip"
            make_archive(db.anim_path, compressed_path)
            anim_path = compressed_path

    return {
        output_rest_vis: db.rest_vis_path,
        output_anim: anim_path,
        output_anim_vis: db.anim_vis_path,
        state: db,
    }


def finish(db: DB = None):
    if db is not None and db.output_dir and os.path.isdir(db.output_dir):
        print(f"Outputs stored in '{db.output_dir}'")
    clear(db)
    return {state: gr.skip() if db is None else db}


@Timing(msg="All done in", print_fn=gr.Success)
def _pipeline(
    input_path: str,
    is_gs=False,
    opacity_threshold=0.0,
    no_fingers=False,
    rest_pose_type: str = None,
    ignore_pose_parts: list[str] = None,
    input_normal=False,
    bw_fix=True,
    bw_vis_bone="LeftArm",
    reset_to_rest=False,
    animation_file: str = None,
    retarget=True,
    inplace=True,
    db: DB = None,
    export_temp=False,
):
    if db is None:
        db = DB()
    with TimePrints():
        print("*" * 50)
    clear(db)
    # Magic sleep to fix the random pydantic_core._pydantic_core.ValidationError in Gradio: https://github.com/gradio-app/gradio/issues/9366#issuecomment-2412903101
    time.sleep(0.1)
    yield prepare_input(input_path, is_gs, opacity_threshold, db, export_temp)
    time.sleep(0.1)
    yield preprocess(db)
    time.sleep(0.1)
    yield infer(input_normal, db)
    time.sleep(0.1)
    yield vis(bw_fix, bw_vis_bone, no_fingers, db)
    time.sleep(0.1)
    yield vis_blender(
        reset_to_rest, no_fingers, rest_pose_type, ignore_pose_parts, animation_file, retarget, inplace, db
    )
    time.sleep(0.1)
    yield finish(db=None)  # keep the outputs for possible re-animation later


def init_models():
    global device, N, hands_resample_ratio, geo_resample_ratio, bw_additional, joints_additional, bones_idx_dict_bw, bones_idx_dict_joints, model_bw, model_bw_normal, model_joints, model_joints_add, model_coarse, model_pose

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fix_random()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IS_HF_ZEROGPU = str2bool(os.getenv("SPACES_ZERO_GPU", False))

    N = 32768
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    ADDITIONAL_BONES = bw_additional = joints_additional = False

    model_bw = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM_ADD if ADDITIONAL_BONES else JOINTS_NUM,
    )
    if ADDITIONAL_BONES:
        model_bw.load("output/vroid/bw.pth")
    else:
        model_bw.load("output/best/new/bw.pth")
    model_bw.to("cpu" if IS_HF_ZEROGPU else device).eval()

    model_bw_normal = PCAE(
        N=N,
        input_normal=True,
        input_attention=True,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM_ADD if ADDITIONAL_BONES else JOINTS_NUM,
    )
    if ADDITIONAL_BONES:
        model_bw_normal.load("output/vroid/bw_normal.pth")
    else:
        model_bw_normal.load("output/best/new/bw_normal.pth")
    bones_idx_dict_bw = BONES_IDX_DICT_ADD if bw_additional else BONES_IDX_DICT
    model_bw_normal.to("cpu" if IS_HF_ZEROGPU else device).eval()

    model_joints = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
        joints_attn_causal=True,
    )
    model_joints.load("output/best/new/joints.pth")
    model_joints.to(device).eval()
    bones_idx_dict_joints = BONES_IDX_DICT_ADD if joints_additional else BONES_IDX_DICT
    assert not (bw_additional and not joints_additional)
    if ADDITIONAL_BONES:
        model_joints_add = PCAE(
            N=N,
            input_normal=False,
            deterministic=True,
            hierarchical_ratio=hierarchical_ratio,
            output_dim=JOINTS_NUM_ADD,
            kinematic_tree=KINEMATIC_TREE_ADD,
            predict_bw=False,
            predict_joints=True,
            predict_joints_tail=True,
            # joints_attn_causal=True,
        )
        model_joints_add.load("output/vroid/joints.pth")
        model_joints_add.to(device).eval()

    model_coarse = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        output_dim=JOINTS_NUM,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
    )
    model_coarse.load("output/best/new/joints_coarse.pth")
    model_coarse.to(device).eval()

    model_pose = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_pose_trans=True,
        pose_mode="ortho6d",
        pose_input_joints=True,
        pose_attn_causal=True,
    )
    model_pose.load("output/best/new/pose.pth")
    model_pose.to(device).eval()

    clear()


def init_blocks():
    global demo, state, output_joints_coarse, output_normed_input, output_sample, output_joints, output_bw, output_rest_vis, output_rest_lbs, output_anim_vis, output_anim

    title = "Make-It-Animatable"
    description = f"""
    <center>
    <h1> 💃 {title} </h1>
    <h2><b>An Efficient Framework for Authoring Animation-Ready 3D Characters</b></h2>
    <h3>
        📄 <a href='https://arxiv.org/abs/2411.18197' target='_blank'>Paper</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;🌟 <a href='https://jasongzy.github.io/Make-It-Animatable/' target='_blank'>Project Page</a>
    </h3>
    </center>
    """
    camera_position = [90, None, 2.5]
    animation_dir = "data/Mixamo/animation"

    with gr.Blocks(title=title, delete_cache=(3600, 3660)) as demo:
        gr.Markdown(description)
        gr.Markdown(
            """
            - Upload a 3D humanoid model or select an example.
                - If you have a character image and hope to make it animatable, these image-to-3D tools might be helpful: [InstantMesh](https://huggingface.co/spaces/TencentARC/InstantMesh), [CharacterGen](https://huggingface.co/spaces/VAST-AI/CharacterGen), [Tripo](https://www.tripo3d.ai/), and [Meshy](https://www.meshy.ai/).
            - Check if the **Input Settings** are suitable and click **Run**.
            - Your animatable model will soon be ready!
                - Optionally, choose another **Animation File** and click **Animate** to quickly apply new motions.
                - If something goes wrong, check the tips at page bottom.
            """
        )

        state = gr.State(DB())
        with gr.Row(variant="panel"):

            # Inputs
            with gr.Column():
                with gr.Row():
                    input_3d = gr.Model3D(label="Input 3D Model", display_mode="solid", camera_position=camera_position)

                with gr.Group():
                    with gr.Row():
                        with gr.Accordion("Input Settings", open=True):
                            input_no_fingers = gr.Checkbox(
                                label="No Fingers",
                                info="Whether the input model does not have ten separate fingers. Can also be used if the output has unsatisfactory finger results.",
                                value=True,
                                interactive=True,
                            )
                            input_rest_pose = gr.Dropdown(
                                ("T-pose", "A-pose", "大-pose", "No"),
                                label="Input Rest Pose",
                                info="If the input model is already in a rest pose, specify here for better performance.",
                                value="No",
                                interactive=True,
                            )
                            input_rest_parts = gr.CheckboxGroup(
                                ("Fingers", "Arms", "Legs", "Head"),
                                label="Input Rest Parts",
                                info="If certain parts of the input model are already in the T-pose, specify here for better performance.",
                                value=[],
                                interactive=True,
                            )
                            input_is_gs = gr.Checkbox(
                                label="Input is GS",
                                info="Whether the input model is Gaussians Splats (only support `.ply` format).",
                                value=False,
                                interactive=True,
                            )
                            input_opacity_threshold = gr.Slider(
                                0.0,
                                1.0,
                                value=0.01,
                                label="Opacity Threshold",
                                info="Only solid Gaussian Splats with opacities larger than this threshold are used in sampling.",
                                step=0.01,
                                interactive=True,
                                visible=bool(input_is_gs.value),
                            )

                    with gr.Row():
                        with gr.Accordion("Weight Settings", open=False):
                            input_normal = gr.Checkbox(
                                label="Use Normal",
                                info="Use normal information to improve performance when the input has limbs close to other ones. Only take effect when the input is a mesh.",
                                value=False,
                                interactive=True,
                            )
                            input_bw_fix = gr.Checkbox(
                                label="Weight Post-Processing",
                                info="Apply some empirical post-processes to the blend weights.",
                                value=True,
                                interactive=True,
                            )
                            input_bw_vis_bone = gr.Radio(
                                [n.lstrip(MIXAMO_PREFIX) for n in (bones_idx_dict_bw).keys()],
                                label="Bone Name of Weight Visualization",
                                value="LeftArm",
                                interactive=True,
                            )

                    with gr.Row():
                        with gr.Accordion("Animation Settings", open=True):
                            input_reset_to_rest = gr.Checkbox(
                                label="Reset to Rest",
                                info="Apply the predicted T-pose in the final animatable model. If no, its rest pose remains the input pose and the animation results may be incorrect.",
                                value=True,
                                interactive=True,
                            )
                            with gr.Row():
                                select_animation_file = gr.Dropdown(
                                    sorted(glob("*.fbx", root_dir=animation_dir)),
                                    label="Select Animation File",
                                    info="Select or upload the motion sequence (`.fbx`) to be applied to the animatable model. Examples can be downloaded from [Mixamo](https://www.mixamo.com) (select `X Bot` as the base character for best practice) . Please ensure the input 3D model is in T-pose or enable the above **Reset to Rest** first. If the animation file is not specified, the animation results will be in the predicted T-pose (static).",
                                    value=None,
                                    interactive=True,
                                )
                                input_animation_file = gr.File(
                                    label="Animation File",
                                    file_types=[".fbx"],
                                    value=lambda: "./data/Standard Run.fbx",
                                    interactive=True,
                                )
                            with gr.Row():
                                input_retarget = gr.Checkbox(
                                    label="Retarget Animation to Character",
                                    info="Produce better animation.",
                                    value=True,
                                    interactive=bool(input_animation_file.value),
                                )
                                input_inplace = gr.Checkbox(
                                    label="In Place",
                                    info="Keep a looping animation in place (e.g., walking, running...).",
                                    value=True,
                                    interactive=input_retarget.interactive,
                                )

                with gr.Row():
                    submit_btn = gr.Button("Run", variant="primary")
                    animate_btn = gr.Button("Animate", variant="secondary")
                with gr.Row():
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.ClearButton()

                with gr.Row(variant="panel"):
                    examples = gr.Examples(
                        examples="./data/examples",
                        inputs=[input_3d, input_is_gs, input_no_fingers, input_rest_pose, input_rest_parts],
                        label="Examples",
                        cache_examples=False,
                        examples_per_page=20,
                    )
                    examples.example_labels = examples.dataset.sample_labels = [
                        os.path.basename(x[0]) for x in examples.examples
                    ]

            inputs = (
                input_3d,
                input_no_fingers,
                input_rest_pose,
                input_rest_parts,
                input_is_gs,
                input_opacity_threshold,
                input_normal,
                input_bw_fix,
                input_bw_vis_bone,
                input_reset_to_rest,
                input_animation_file,
                input_retarget,
                input_inplace,
            )

            # Outputs
            with gr.Column():
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Coarse Localization"):
                            output_joints_coarse = gr.Model3D(
                                label="Joints (coarse)", display_mode="wireframe", camera_position=camera_position
                            )
                        with gr.Tab("Canonical Transformation"):
                            output_normed_input = gr.Model3D(
                                label="Canonicalized Input", display_mode="solid", camera_position=camera_position
                            )
                        with gr.Tab("Sampling"):
                            output_sample = gr.Model3D(
                                label="Sampled Point Clouds",
                                display_mode="point_cloud",
                                camera_position=camera_position,
                            )
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Joints"):
                            output_joints = gr.Model3D(
                                label="Joints", display_mode="wireframe", camera_position=camera_position
                            )
                        with gr.Tab("Blend Weights"):
                            output_bw = gr.Model3D(
                                label="Blend Weights", display_mode="solid", camera_position=camera_position
                            )
                with gr.Row():
                    with gr.Tabs(selected=1):
                        with gr.Tab("Rest Pose (joints)", id=0):
                            output_rest_lbs = gr.Model3D(
                                label="Rest Pose", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown(
                                "The transforming result here may be inaccurate. See **Rest Pose (texture preview)** for optimal visualization."
                            )
                        with gr.Tab("Rest Pose (texture preview)", id=1):
                            output_rest_vis = gr.Model3D(
                                label="Rest Pose", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown("**Point clouds** and **Gaussian Splats** are not supported for preview here.")
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Animatable Model (GLB preview)"):
                            output_anim_vis = gr.Model3D(
                                label="Animatable Model", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown(
                                """
                                - Gradio hasn't support the FBX format yet (see [this issue](https://github.com/gradio-app/gradio/issues/10007)), so we use [FBX2glTF](https://github.com/facebookincubator/FBX2glTF) internally to convert the exported FBX into GLB for quick preview here.
                                Due to this conversion process, some models may exhibit inconsistencies in material properties and texture rendering. Download the **FBX** file for higher fidelity.
                                - **Point clouds** and **Gaussian Splats** are not supported for preview here. Download the **FBX**/**BLEND** file to view their results.
                                """
                            )
                        with gr.Tab("Animatable Model (FBX/BLEND)"):
                            output_anim = gr.File(label="Animatable Model")
                            gr.Markdown(
                                """
                                - Recommend to view and edit in Blender.
                                - For **Gaussian Splats**, the **[3DGS Render Blender Addon by KIRI Engine](https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases/tag/v1.0.0)** is required to open the **BLEND** file here.
                                """
                            )
        with gr.Row():
            gr.Markdown(
                """
                Tips:
                - To Hugging Face demo users: 3D Gaussian Splats are not supported with the ZeroGPU environment (Python 3.10). Setup an environment with Python 3.11 and run this demo locally to enable GS support.
                - The output results may not be displayed properly if this browser tab is unfocused during inference.
                - If the results suffer from low blend weight quality (typically occurring when limbs are close together, e.g., inner thigh and armpit), try enabling the **Use Normal** option.
                - If the pose-to-rest transformation is unsatisfactory, try adding prior knowledge by specifying the **Input Rest Pose** and **Input Rest Parts**.
                    - Alternatively, you can uncheck **Reset to Rest** and clear the **Animation File**, so that the animation result becomes an invertible T-pose model that can be adjusted in Blender.
                - This demo is designed for standard human skeletons (compatible with the Mixamo definition). If the input 3D model includes significant accessories (e.g., hand-held objects, wings, long tails, long hair), the results may not be optimal.
                """
            )

            outputs = (
                output_joints_coarse,
                output_normed_input,
                output_sample,
                output_joints,
                output_bw,
                output_rest_vis,
                output_rest_lbs,
                output_anim_vis,
                output_anim,
            )
            for e in outputs:
                e.interactive = False

            # Events

            def clear_components(inputs: dict):
                return [None] * len(inputs)

            input_3d.upload(fn=ply2visible, inputs=[input_3d, input_is_gs], outputs=input_3d)
            input_is_gs.change(fn=ply2visible, inputs=[input_3d, input_is_gs], outputs=input_3d)
            input_is_gs.change(
                fn=lambda x: gr.Slider(visible=True) if x else gr.Slider(visible=False),
                inputs=input_is_gs,
                outputs=input_opacity_threshold,
                show_progress="hidden",
            )

            def pipeline(inputs: dict, progress=gr.Progress()):
                progress(0, "Starting...")
                if device.type == "cpu":
                    gr.Warning("Running on CPU will take a much longer time", duration=None)

                yield from progress.tqdm(
                    _pipeline(
                        input_path=inputs[input_3d],
                        is_gs=inputs[input_is_gs],
                        opacity_threshold=inputs[input_opacity_threshold],
                        no_fingers=inputs[input_no_fingers],
                        rest_pose_type=inputs[input_rest_pose],
                        ignore_pose_parts=inputs[input_rest_parts],
                        input_normal=inputs[input_normal],
                        bw_fix=inputs[input_bw_fix],
                        bw_vis_bone=inputs[input_bw_vis_bone],
                        reset_to_rest=inputs[input_reset_to_rest],
                        animation_file=inputs[input_animation_file],
                        retarget=inputs[input_retarget],
                        inplace=inputs[input_inplace],
                        db=inputs[state],
                        # export_temp=True,
                    )
                )
                # gr.Success("Finished successfully!")

            submit_event = submit_btn.click(
                fn=clear_components, inputs=set(outputs), outputs=outputs, show_progress="hidden"
            ).success(
                fn=pipeline, inputs=set(inputs + (state,)), outputs=set(outputs + (state,)), show_progress="minimal"
            )
            animate_event = animate_btn.click(
                fn=clear_components,
                inputs={output_rest_vis, output_anim, output_anim_vis},
                outputs=[output_rest_vis, output_anim, output_anim_vis],
            ).success(
                fn=vis_blender,
                inputs=[
                    input_reset_to_rest,
                    input_no_fingers,
                    input_rest_pose,
                    input_rest_parts,
                    input_animation_file,
                    input_retarget,
                    input_inplace,
                    state,
                ],
                outputs={output_rest_vis, output_anim, output_anim_vis, state},
            )
            animate_event.success(fn=finish, outputs={state})
            stop_btn.click(fn=lambda: [], cancels=[submit_event, animate_event]).success(
                fn=lambda: gr.Warning("Job cancelled") or []
            )
            clear_btn.click(fn=clear_components, inputs=set(outputs), outputs=outputs).success(
                fn=clear, inputs=state, outputs=state
            )

            def select2file(selected: str):
                return None if selected is None else os.path.join(animation_dir, selected)

            select_animation_file.input(
                lambda x: (select2file(x), gr.Checkbox(value=True)),
                inputs=select_animation_file,
                outputs=[input_animation_file, input_reset_to_rest],
                preprocess=False,
            )
            input_animation_file.upload(
                lambda: (gr.Dropdown(value=None), gr.Checkbox(value=True)),
                outputs=[select_animation_file, input_reset_to_rest],
            )
            input_animation_file.clear(
                lambda: gr.Dropdown(value=None), outputs=select_animation_file, show_progress="hidden"
            )
            input_animation_file.change(
                lambda x: (
                    (gr.Checkbox(interactive=True), gr.Checkbox(interactive=True))
                    if x
                    else (gr.Checkbox(interactive=False), gr.Checkbox(interactive=False))
                ),
                inputs=input_animation_file,
                outputs=[input_retarget, input_inplace],
                show_progress="hidden",
            )

            input_retarget.change(
                lambda x: gr.Checkbox(interactive=True) if x else gr.Checkbox(interactive=False),
                inputs=input_retarget,
                outputs=input_inplace,
                show_progress="hidden",
            )

            demo.unload(fn=lambda: not clear(state.value) or None)  # just to make sure fn returns None

    return demo


if __name__ == "__main__":
    init_models()
    demo = init_blocks()

    # for input_path in ["./data/examples/bunny.glb"]:
    #     for _ in _pipeline(
    #         input_path,
    #         is_gs=False,
    #         opacity_threshold=0.01,
    #         no_fingers=True,
    #         rest_pose_type="No",
    #         ignore_pose_parts=["Head"],
    #         input_normal=False,
    #         bw_fix=True,
    #         bw_vis_bone="Head",
    #         reset_to_rest=True,
    #         animation_file="./data/Standard Run.fbx",
    #         retarget=True,
    #         inplace=True,
    #     ):
    #         pass

    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[".", ".."], show_error=True, ssr_mode=False)
