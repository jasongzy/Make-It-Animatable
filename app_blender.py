import argparse
import os
import tempfile

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Scale

import util.blender_utils as blender_utils
from util.blender_utils import bpy as bpy
from util.utils import HiddenPrints, save_gs, transform_gs


def is_finger(bone_name: str):
    return any(f in bone_name for f in {"Thumb", "Index", "Middle", "Ring", "Pinky"})


def remove_fingers_from_data(data: np.ndarray, bones_idx_dict: dict[str, int], is_bw=False):
    assert data.shape[0] == len(bones_idx_dict)
    if is_bw:
        for k, v in bones_idx_dict.items():
            if is_finger(k):
                hand = "Left" if "Left" in k else "Right"
                data[bones_idx_dict[f"mixamorig:{hand}Hand"]] += data[v]
    data_new = [None] * len(bones_idx_dict)
    for k, v in bones_idx_dict.items():
        if is_finger(k):
            continue
        data_new[v] = data[v]
    data_new = np.stack([x for x in data_new if x is not None], axis=0)
    return data_new


def main(args: argparse.Namespace):
    if isinstance(args.input_path, str):
        data = np.load(args.input_path, allow_pickle=True)
    else:
        assert isinstance(args.input_path, dict)
        data = args.input_path
    mesh = data["mesh"]
    if isinstance(mesh, np.ndarray):
        mesh: trimesh.Trimesh = mesh.item()
    mesh = mesh.copy()
    gs = data["gs"]
    joints = data["joints"]
    joints_tail = data["joints_tail"]
    bw = data["bw"].copy()
    pose = data["pose"]
    bones_idx_dict = data["bones_idx_dict"]
    if isinstance(bones_idx_dict, np.ndarray):
        bones_idx_dict: dict[str, int] = bones_idx_dict.item()
    pose_ignore_list = list(data.get("pose_ignore_list", []))

    if args.remove_fingers:
        joints = remove_fingers_from_data(joints, bones_idx_dict)
        joints_tail = remove_fingers_from_data(joints_tail, bones_idx_dict)
        bw = remove_fingers_from_data(bw.T, bones_idx_dict, is_bw=True).T
        if pose is not None:
            pose = remove_fingers_from_data(pose, bones_idx_dict)
        joints_list = [None] * len(bones_idx_dict)
        for k, v in bones_idx_dict.items():
            joints_list[v] = k
        joints_list = [x for x in joints_list if not is_finger(x)]
        bones_idx_dict = {name: i for i, name in enumerate(joints_list)}
        assert len(bones_idx_dict) == joints.shape[0]

    with HiddenPrints(suppress_err=True):
        blender_utils.reset()

        template = blender_utils.load_file(args.template_path)
        for mesh_obj in blender_utils.get_all_mesh_obj(template):
            bpy.data.objects.remove(mesh_obj, do_unlink=True)
        armature_obj = blender_utils.get_armature_obj(template)
        armature_obj.animation_data_clear()
        with blender_utils.Mode("POSE", armature_obj):
            bpy.ops.pose.select_all(action="SELECT")
            bpy.ops.pose.transforms_clear()
        if args.keep_raw:
            scaling = 1.0
            bpy.context.view_layer.objects.active = armature_obj
            armature_obj.select_set(state=True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        else:
            matrix_world = armature_obj.matrix_world.copy()
            scaling = matrix_world.to_scale()[0]
            armature_obj.matrix_world.identity()
        blender_utils.update()

        with tempfile.NamedTemporaryFile(suffix=".glb") as f:
            if not args.keep_raw:
                verts = mesh.vertices
                verts[:, 1], verts[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
                mesh.vertices = verts / scaling
            mesh.export(f.name)
            mesh_obj = blender_utils.load_file(f.name)
            mesh_obj = blender_utils.get_all_mesh_obj(mesh_obj)[0]
            mesh_data: bpy.types.Mesh = mesh_obj.data
            mesh_obj.name = mesh_data.name = "mesh"
            for mat in mesh_data.materials:
                for link in mat.node_tree.links:
                    if link.from_node.bl_idname == "ShaderNodeNormalMap":
                        mat.node_tree.links.remove(link)

        blender_utils.set_rest_bones(armature_obj, joints / scaling, joints_tail / scaling, bones_idx_dict)
        if args.keep_raw:
            bpy.context.view_layer.objects.active = armature_obj
            armature_obj.select_set(state=True)
            armature_obj.rotation_mode = "XYZ"
            armature_obj.rotation_euler[0] = np.pi / 2
            bpy.ops.object.transform_apply(rotation=True)
            blender_utils.update()
        blender_utils.set_armature_parent([mesh_obj], armature_obj)
        blender_utils.set_weights([mesh_obj], bw, bones_idx_dict)
        if not args.keep_raw:
            armature_obj.matrix_world = matrix_world
        blender_utils.remove_empty()
        blender_utils.update()
        if pose is not None:
            pose_inv = pose.copy()
            if args.keep_raw:
                rot_mat = np.array([
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ], dtype=pose_inv.dtype)
                pose_inv = rot_mat @ pose_inv @ rot_mat.T
            if not args.pose_local:
                pose_inv[:, :3, 3] /= scaling
            blender_utils.set_bone_pose(armature_obj, pose_inv, bones_idx_dict, local=args.pose_local)
            for bone in armature_obj.pose.bones:
                bone.location = (0, 0, 0)
                if pose_ignore_list and any(x in bone.name for x in pose_ignore_list):
                    bone.matrix_basis = blender_utils.mathutils.Quaternion().to_matrix().to_4x4()
                blender_utils.update()
            if args.reset_to_rest:
                blender_utils.set_rest_bones(armature_obj, reset_as_rest=True)
            if args.rest_path:
                bpy.ops.export_scene.gltf(
                    filepath=args.rest_path,
                    check_existing=False,
                    use_selection=False,
                    export_animations=False,
                    export_rest_position_armature=False,
                    # export_yup=False,
                )
        if args.animation_path:
            blender_utils.load_mixamo_anim(
                [armature_obj, mesh_obj], args.animation_path, do_retarget=args.retarget, inplace=args.inplace
            )

        blender_utils.update()
        if args.output_path.endswith(".fbx"):
            bpy.ops.export_scene.fbx(
                filepath=args.output_path,
                check_existing=False,
                use_selection=False,
                add_leaf_bones=False,
                bake_anim=bool(args.animation_path),
                path_mode="COPY",
                embed_textures=True,
            )
        elif args.output_path.endswith(".blend"):
            scn = bpy.context.scene
            cam_obj = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
            cam_obj.location = (2, -3.5, 1)
            cam_obj.rotation_euler = (75 / 180 * np.pi, 0, 30 / 180 * np.pi)
            scn.collection.objects.link(cam_obj)
            scn.camera = cam_obj

            keyframes = blender_utils.get_keyframes([armature_obj])
            if keyframes:
                scn.frame_start, scn.frame_end = min(keyframes), max(keyframes)
                scn.frame_set(scn.frame_start)
                scn.render.image_settings.file_format = "FFMPEG"
                scn.render.ffmpeg.format = "MPEG4"
            scn.render.resolution_x = 1024
            scn.render.resolution_y = 1024
            scn.render.resolution_percentage = 50

            if gs is not None:
                mesh_obj.hide_set(True)
                with tempfile.NamedTemporaryFile(suffix=".ply") as f:
                    gs = torch.from_numpy(gs)
                    gs = transform_gs(gs, transform=(Scale(1 / scaling)))
                    save_gs(gs, f.name)
                    gs_obj = blender_utils.load_3dgs(f.name)
                    gs_obj = blender_utils.get_all_mesh_obj(gs_obj)[0]
                    gs_obj.name = gs_obj.data.name = "gs"
                blender_utils.set_armature_parent([gs_obj], armature_obj, type="ARMATURE_NAME", no_inv=True)
                blender_utils.set_weights([gs_obj], bw.repeat(4, axis=0), bones_idx_dict)
                bpy.ops.sna.dgs__set_render_engine_to_eevee_7516e()
                # bpy.ops.sna.dgs__start_camera_update_9eaff()

            if os.path.isfile(args.output_path):
                os.remove(args.output_path)
            bpy.ops.wm.save_as_mainfile(filepath=args.output_path)
        else:
            raise ValueError(f"Unsupported output format: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--template_path", type=str, default=None)
    parser.add_argument("--keep_raw", default=False, action="store_true")
    parser.add_argument("--rest_path", type=str, default=None)
    parser.add_argument("--pose_local", default=False, action="store_true")
    parser.add_argument("--reset_to_rest", default=False, action="store_true")
    parser.add_argument("--remove_fingers", default=False, action="store_true")
    parser.add_argument("--animation_path", type=str, default=None)
    parser.add_argument("--retarget", default=False, action="store_true")
    parser.add_argument("--inplace", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
