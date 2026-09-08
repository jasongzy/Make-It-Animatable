import os
import random
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from glob import glob
from types import MappingProxyType
from typing import Callable, Iterator

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util import blender_utils
from util.blender_utils import bpy as bpy
from util.utils import HiddenPrints, apply_transform, decompose_transform, quat_to_matrix, sample_mesh

MIXAMO_PREFIX = "mixamorig:"
MIXAMO_JOINTS = (
    "mixamorig:Hips",
    "mixamorig:Spine",
    "mixamorig:Spine1",
    "mixamorig:Spine2",
    "mixamorig:Neck",
    "mixamorig:Head",
    # "mixamorig:HeadTop_End",
    # Eyes
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    # Left fingers
    "mixamorig:LeftHandThumb1",
    "mixamorig:LeftHandThumb2",
    "mixamorig:LeftHandThumb3",
    # "mixamorig:LeftHandThumb4",
    "mixamorig:LeftHandIndex1",
    "mixamorig:LeftHandIndex2",
    "mixamorig:LeftHandIndex3",
    # "mixamorig:LeftHandIndex4",
    "mixamorig:LeftHandMiddle1",
    "mixamorig:LeftHandMiddle2",
    "mixamorig:LeftHandMiddle3",
    # "mixamorig:LeftHandMiddle4",
    "mixamorig:LeftHandRing1",
    "mixamorig:LeftHandRing2",
    "mixamorig:LeftHandRing3",
    # "mixamorig:LeftHandRing4",
    "mixamorig:LeftHandPinky1",
    "mixamorig:LeftHandPinky2",
    "mixamorig:LeftHandPinky3",
    # "mixamorig:LeftHandPinky4",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
    # Right fingers
    "mixamorig:RightHandThumb1",
    "mixamorig:RightHandThumb2",
    "mixamorig:RightHandThumb3",
    # "mixamorig:RightHandThumb4",
    "mixamorig:RightHandIndex1",
    "mixamorig:RightHandIndex2",
    "mixamorig:RightHandIndex3",
    # "mixamorig:RightHandIndex4",
    "mixamorig:RightHandMiddle1",
    "mixamorig:RightHandMiddle2",
    "mixamorig:RightHandMiddle3",
    # "mixamorig:RightHandMiddle4",
    "mixamorig:RightHandRing1",
    "mixamorig:RightHandRing2",
    "mixamorig:RightHandRing3",
    # "mixamorig:RightHandRing4",
    "mixamorig:RightHandPinky1",
    "mixamorig:RightHandPinky2",
    "mixamorig:RightHandPinky3",
    # "mixamorig:RightHandPinky4",
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase",
    # "mixamorig:LeftToe_End",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
    "mixamorig:RightToeBase",
    # "mixamorig:RightToe_End",
)
assert len(MIXAMO_JOINTS) == 52
JOINTS_NUM = len(MIXAMO_JOINTS)
MIXAMO_JOINTS_HANDS = (
    "mixamorig:LeftHand",
    "mixamorig:LeftHandThumb1",
    "mixamorig:LeftHandThumb2",
    "mixamorig:LeftHandThumb3",
    # "mixamorig:LeftHandThumb4",
    "mixamorig:LeftHandIndex1",
    "mixamorig:LeftHandIndex2",
    "mixamorig:LeftHandIndex3",
    # "mixamorig:LeftHandIndex4",
    "mixamorig:LeftHandMiddle1",
    "mixamorig:LeftHandMiddle2",
    "mixamorig:LeftHandMiddle3",
    # "mixamorig:LeftHandMiddle4",
    "mixamorig:LeftHandRing1",
    "mixamorig:LeftHandRing2",
    "mixamorig:LeftHandRing3",
    # "mixamorig:LeftHandRing4",
    "mixamorig:LeftHandPinky1",
    "mixamorig:LeftHandPinky2",
    "mixamorig:LeftHandPinky3",
    # "mixamorig:LeftHandPinky4",
    "mixamorig:RightHand",
    "mixamorig:RightHandThumb1",
    "mixamorig:RightHandThumb2",
    "mixamorig:RightHandThumb3",
    # "mixamorig:RightHandThumb4",
    "mixamorig:RightHandIndex1",
    "mixamorig:RightHandIndex2",
    "mixamorig:RightHandIndex3",
    # "mixamorig:RightHandIndex4",
    "mixamorig:RightHandMiddle1",
    "mixamorig:RightHandMiddle2",
    "mixamorig:RightHandMiddle3",
    # "mixamorig:RightHandMiddle4",
    "mixamorig:RightHandRing1",
    "mixamorig:RightHandRing2",
    "mixamorig:RightHandRing3",
    # "mixamorig:RightHandRing4",
    "mixamorig:RightHandPinky1",
    "mixamorig:RightHandPinky2",
    "mixamorig:RightHandPinky3",
    # "mixamorig:RightHandPinky4",
)
BONES_IDX_DICT = OrderedDict({name: i for i, name in enumerate(MIXAMO_JOINTS)})
BONES_IDX_DICT = MappingProxyType(BONES_IDX_DICT)
HANDS_IDX_DICT = OrderedDict({name: BONES_IDX_DICT[name] for name in MIXAMO_JOINTS_HANDS})
HANDS_IDX_DICT = MappingProxyType(HANDS_IDX_DICT)

# Exist in all characters
JOINTS_EXIST = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)
JOINTS_EXIST_MASK = tuple(name.lstrip(MIXAMO_PREFIX) in JOINTS_EXIST for name in MIXAMO_JOINTS)

# Priors for all poses
CONNECTED_BONES = (
    ("Hips", "Spine", "Spine1", "Spine2"),
    ("Shoulder", "Arm", "ForeArm", "Hand"),
    ("UpLeg", "Leg", "Foot", "ToeBase"),
    ("HandThumb1", "HandThumb2", "HandThumb3"),
    ("HandIndex1", "HandIndex2", "HandIndex3"),
    ("HandMiddle1", "HandMiddle2", "HandMiddle3"),
    ("HandRing1", "HandRing2", "HandRing3"),
    ("HandPinky1", "HandPinky2", "HandPinky3"),
)


def get_connected_idx_pairs(
    connected_bones: tuple[tuple[str]],
    bones_idx_dict: dict[str, int],
    left_right_keywords: tuple[str] = ("Arm", "Leg", "Hand"),
) -> tuple[tuple[int, int]]:
    assert all(len(bones) >= 2 for bones in connected_bones)
    connected_idx_pairs = []
    for bones in connected_bones:
        for i in range(len(bones) - 1):
            parent = bones[i]
            child = bones[i + 1]
            if any(any(x in b for b in bones) for x in left_right_keywords):
                pairs = [
                    (bones_idx_dict[f"{MIXAMO_PREFIX}Left{parent}"], bones_idx_dict[f"{MIXAMO_PREFIX}Left{child}"]),
                    (bones_idx_dict[f"{MIXAMO_PREFIX}Right{parent}"], bones_idx_dict[f"{MIXAMO_PREFIX}Right{child}"]),
                ]
            else:
                pairs = [(bones_idx_dict[f"{MIXAMO_PREFIX}{parent}"], bones_idx_dict[f"{MIXAMO_PREFIX}{child}"])]
            connected_idx_pairs.extend(pairs)
    return tuple(connected_idx_pairs)


CONNECTED_IDX_PAIRS = get_connected_idx_pairs(CONNECTED_BONES, BONES_IDX_DICT)

# Priors for rest pose (t-pose)
SYMMETRIC_BONES = tuple((x, x.replace("Left", "Right")) for x in MIXAMO_JOINTS if "Left" in x)
SYMMETRIC_IDX_PAIRS = tuple((BONES_IDX_DICT[x], BONES_IDX_DICT[y]) for x, y in SYMMETRIC_BONES)
# AXIS_BONES = tuple(x for x in MIXAMO_JOINTS if "Left" not in x and "Right" not in x)
AXIS_BONES = (
    f"{MIXAMO_PREFIX}Hips",
    f"{MIXAMO_PREFIX}Spine",
    f"{MIXAMO_PREFIX}Spine1",
    f"{MIXAMO_PREFIX}Spine2",
    f"{MIXAMO_PREFIX}Neck",
    f"{MIXAMO_PREFIX}Head",
)
AXIS_IDX = tuple(BONES_IDX_DICT[x] for x in AXIS_BONES)
REVERSE_DIR_BONES = (("LeftArm", "RightArm"), ("LeftForeArm", "RightForeArm"), ("LeftHand", "RightHand"))
REVERSE_DIR_IDX_PAIRS = tuple(
    (BONES_IDX_DICT[f"{MIXAMO_PREFIX}{x}"], BONES_IDX_DICT[f"{MIXAMO_PREFIX}{y}"]) for x, y in REVERSE_DIR_BONES
)
SAME_DIR_BONES = (
    (
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
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
    ("LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3"),
    (
        "RightArm",
        "RightForeArm",
        "RightHand",
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
    ("RightHandThumb1", "RightHandThumb2", "RightHandThumb3"),
    ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
    ("LeftToeBase", "RightToeBase"),
)
SAME_DIR_IDX_GROUPS = tuple(tuple(BONES_IDX_DICT[f"{MIXAMO_PREFIX}{x}"] for x in g) for g in SAME_DIR_BONES)


@dataclass(frozen=True)
class Joint:
    name: str
    index: int
    parent: Self | None
    children: list[Self]
    template_joints: tuple[str]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __iter__(self) -> Iterator[Self]:
        yield self
        for child in self.children:
            yield from child

    @cached_property
    def children_recursive(self) -> list[Self]:
        # return [child for child in self if child is not self]
        children_list = []
        if not self.children:
            return children_list
        for child in self.children:
            children_list.append(child)
            children_list.extend(child.children_recursive)
        return children_list

    def __len__(self):
        return len(self.children_recursive) + 1

    def __contains__(self, item: Self | str):
        if isinstance(item, str):
            return item == self.name or item in (child.name for child in self.children_recursive)
        elif isinstance(item, Joint):
            return item is Self or item in self.children_recursive
        else:
            raise TypeError(f"Item must be {self.__class__.__name__} or str, not {type(item)}")

    @cached_property
    def children_recursive_dict(self) -> dict[str, Self]:
        return {child.name: child for child in self.children_recursive}

    def __getitem__(self, index: int | str) -> Self:
        if index in (0, self.name):
            return self
        if isinstance(index, int):
            index -= 1
            return self.children_recursive[index]
        elif isinstance(index, str):
            return self.children_recursive_dict[index]
        else:
            raise TypeError(f"Index must be int or str, not {type(index)}")

    @cached_property
    def parent_recursive(self) -> list[Self]:
        parent_list = []
        if self.parent is None:
            return parent_list
        parent_list.append(self.parent)
        parent_list.extend(self.parent.parent_recursive)
        return parent_list

    @cached_property
    def joints_list(self) -> list[Self]:
        joints_list = [None] * len(self)
        for joint in self:
            joints_list[joint.index] = joint
        assert None not in joints_list
        return joints_list

    @cached_property
    def parent_indices(self) -> list[int]:
        return [-1 if joint.parent is None else joint.parent.index for joint in self.joints_list]

    def get_first_valid_parent(self, valid_names: list[str]) -> Self | None:
        return next((parent for parent in self.parent_recursive if parent.name in valid_names), None)

    @cached_property
    def tree_levels(self) -> dict[int, list[Self]]:
        levels = {0: [self]}
        if self.children:
            for child in self.children:
                for l, nodes in child.tree_levels.items():
                    levels.setdefault(l + 1, []).extend(nodes)
        return levels

    @cached_property
    def tree_levels_name(self) -> dict[int, list[int]]:
        return {l: [j.name for j in level] for l, level in self.tree_levels.items()}

    @cached_property
    def tree_levels_index(self) -> dict[int, list[int]]:
        return {l: [j.index for j in level] for l, level in self.tree_levels.items()}

    @cached_property
    def tree_levels_mask(self):
        return [
            [j in self.tree_levels_name[l] for j in self.template_joints] for l in range(len(self.tree_levels_name))
        ]


def build_skeleton(armature_obj: blender_utils.Object, bones_idx_dict: dict[str, int]):
    template_joints = tuple(bones_idx_dict.keys())

    def get_children(bone, parent=None):
        joint = Joint(
            bone.name, index=bones_idx_dict[bone.name], parent=parent, children=[], template_joints=template_joints
        )
        children = [b for b in bone.children if b.name in bones_idx_dict]
        if not children:
            return joint
        for child in bone.children:
            joint.children.append(get_children(child, parent=joint))
        return joint

    hips_bone = armature_obj.data.bones[f"{MIXAMO_PREFIX}Hips"]
    hips = get_children(hips_bone)
    return hips


def get_kinematic_tree(bone_path: str, bone_idx_dict: dict[str, int]):
    assert os.path.isfile(bone_path), f"File not found: {bone_path}"
    with HiddenPrints():
        blender_utils.reset()
        kinematic_tree = build_skeleton(
            blender_utils.get_armature_obj(blender_utils.load_file(bone_path)), bone_idx_dict
        )
        blender_utils.reset()
    return kinematic_tree


TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Mixamo/bones.fbx"))
KINEMATIC_TREE = get_kinematic_tree(TEMPLATE_PATH, BONES_IDX_DICT)
assert len(KINEMATIC_TREE) == len(MIXAMO_JOINTS)

# is_finger = lambda x: any(f in x for f in {"Thumb", "Index", "Middle", "Ring", "Pinky"})
# joints_list = [x for x in KINEMATIC_TREE.joints_list if not is_finger(x)]
# bones_idx_dict = {name: i for i, name in enumerate(joints_list)}
# tree = get_kinematic_tree(os.path.join(TEMPLATE_PATH, bones_idx_dict)
# np.save("tree.npy", np.array(tree, dtype=object))


@dataclass(frozen=True)
class MetaData:
    index: int | list[int]
    char_id: str | list[str]
    anim_id: str | list[str]
    char_path: str | list[str]
    anim_path: str | list[str]
    frames: int | list[int]
    keyframes: list[int] | list[list[int]]
    faces: np.ndarray | list[np.ndarray]
    rest: "PoseData | list[PoseData]"
    bones_idx_dict: dict[str, int]

    def __len__(self):
        assert not isinstance(
            self.index, int
        ), f"Non-batchified {self.__class__.__name__} can not be counted or subscripted"
        return len(self.index)

    def __getitem__(self, index: int):
        len(self)  # guard
        return self.__class__(**{k: v[index] for k, v in self.__dict__.items()})


def get_hips_normal(hips: torch.Tensor, rightupleg: torch.Tensor, leftupleg: torch.Tensor):
    """
    Args:
        hips: (..., 3)
        rightupleg: (..., 3)
        leftupleg: (..., 3)
    Returns:
        (..., 3) Hips plane normal: direction: face-forward, magnitude: indicating skeleton size.
    """
    module = torch if isinstance(hips, torch.Tensor) else np
    vector_right = rightupleg - hips
    vector_left = leftupleg - hips
    normal = module.cross(vector_right, vector_left, dim=-1)
    return normal


def get_plane_transform(origin: torch.Tensor, normal: torch.Tensor, tangent_ends: torch.Tensor):
    """
    Args:
        origin: (B, 3)
        normal: (B, 3)
        tangent_ends: (B, 2, 3) two ends: begin + end
    Returns:
        (B, 4, 4) transform matrix that moves to `origin`,
        and rotates to let the `normal` oriented in the positive direction of z-axis
        and the tangent vectors `tangent_ends[:, 1] - tangent_ends[:, 0]` oriented in the positive direction of x-axis.
    """
    assert origin.shape == normal.shape and len(origin.shape) == 2 and origin.shape[-1] == 3
    B = normal.shape[0]
    normal = F.normalize(normal, dim=-1)
    target_normal = F.normalize(torch.tensor([0, 0, 1]).to(normal), dim=-1).unsqueeze(0).repeat(B, 1)
    cross = torch.cross(normal, target_normal, dim=-1)
    dot = torch.sum(normal * target_normal, dim=-1, keepdim=True)
    # cross_norm = cross.norm(dim=-1, keepdim=True)
    skew_symmetric = torch.zeros((B, 3, 3)).to(normal)
    skew_symmetric[:, 0, 1] = -cross[:, 2]
    skew_symmetric[:, 0, 2] = cross[:, 1]
    skew_symmetric[:, 1, 0] = cross[:, 2]
    skew_symmetric[:, 1, 2] = -cross[:, 0]
    skew_symmetric[:, 2, 0] = -cross[:, 1]
    skew_symmetric[:, 2, 1] = cross[:, 0]
    rotation = (
        torch.eye(3).to(normal).unsqueeze(0).repeat(B, 1, 1)
        + skew_symmetric
        # + skew_symmetric @ skew_symmetric * ((1 - dot) / (cross_norm**2)).unsqueeze(-1)
        + skew_symmetric @ skew_symmetric * (1 / (1 + dot + 1e-8)).unsqueeze(-1)
    )

    # Handle singularity where normal is anti-parallel to target_normal (dot ~ -1)
    # In this case, we rotate 180 degrees around the X-axis.
    # This maps (0, 0, -1) to (0, 0, 1).
    rotation_singular = torch.tensor(
        [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], device=normal.device, dtype=normal.dtype
    )
    rotation_singular = rotation_singular.unsqueeze(0).expand(B, 3, 3)
    mask = (dot < -1.0 + 1e-6).view(B, 1, 1)
    rotation = torch.where(mask, rotation_singular, rotation)

    # assert torch.allclose(torch.einsum("bij,bj->bi", rotation, normal), target_normal, atol=1e-5)
    rotation_matrix = torch.eye(4).to(normal).unsqueeze(0).repeat(B, 1, 1)
    rotation_matrix[:, :3, :3] = rotation
    transl = -origin
    transl_matrix = torch.eye(4).to(normal).unsqueeze(0).repeat(B, 1, 1)
    transl_matrix[:, :3, 3] = transl
    matrix = rotation_matrix @ transl_matrix

    # Rotate around z-axis
    tangent_ends = apply_transform(tangent_ends, matrix.unsqueeze(1).expand(-1, 2, -1, -1))
    tangent = tangent_ends[:, 1] - tangent_ends[:, 0]
    # assert torch.allclose(tangent[:, 2], torch.zeros_like(tangent[:, 2]), atol=1e-5)
    angles = torch.atan2(tangent[:, 1], tangent[:, 0])
    cos_angles = torch.cos(-angles)
    sin_angles = torch.sin(-angles)
    rotation_matrix2 = torch.zeros((B, 4, 4)).to(tangent)
    rotation_matrix2[:, 0, 0] = cos_angles
    rotation_matrix2[:, 0, 1] = -sin_angles
    rotation_matrix2[:, 1, 0] = sin_angles
    rotation_matrix2[:, 1, 1] = cos_angles
    rotation_matrix2[:, 2, 2] = 1
    rotation_matrix2[:, 3, 3] = 1
    matrix = rotation_matrix2 @ matrix

    return matrix


def get_hips_transform(hips: torch.Tensor, rightupleg: torch.Tensor, leftupleg: torch.Tensor):
    """
    Args:
        hips: (B, 3)
        rightupleg: (B, 3)
        leftupleg: (B, 3)
    Returns:
        (B, 4, 4) transform matrix that moves Hips to the origin,
        and rotates body to face forward in the positive direction of z-axis
        and right-to-left oriented in the positive direction of x-axis.
    """
    # assert hips.shape == rightupleg.shape == leftupleg.shape and len(hips.shape) == 2 and hips.shape[-1] == 3
    normal = get_hips_normal(hips, rightupleg, leftupleg)
    tangent_ends = torch.stack((rightupleg, leftupleg), dim=-2)
    return get_plane_transform(hips, normal, tangent_ends)


@dataclass(frozen=True)
class PoseData:
    verts: torch.Tensor = None
    """(B, N, 3)"""
    verts_normal: torch.Tensor = None
    """(B, N, 3)"""
    pts: torch.Tensor = None
    """(B, N_sample, 3)"""
    pts_normal: torch.Tensor = None
    """(B, N_sample, 3)"""
    weights: torch.Tensor = None
    """(B, N, K)"""
    joints: torch.Tensor = None
    """(B, K, 3)"""
    joints_tail: torch.Tensor = None
    """(B, K, 3)"""
    joints_pose: torch.Tensor = None
    """(B, K, 4) rest-to-pose local rotations (quaternions relative to posed parent bone)"""
    joints_pose_rel2rest: torch.Tensor = None
    """(B, K, 4) rest-to-pose local rotations (quaternions relative to rest parent bone)"""
    joints_transform: torch.Tensor = None
    """(B, K, 4, 4) rest-to-pose transformation matrix"""
    meta: MetaData = None

    def __len__(self):
        assert isinstance(
            self.verts, torch.Tensor
        ), f"Non-batchified {self.__class__.__name__} can not be counted or subscripted"
        return self.verts.shape[0]

    def __getitem__(self, index: int):
        len(self)  # guard
        return self.__class__(**{k: v[index] for k, v in self.__dict__.items()})

    @cached_property
    def weights_mask(self):
        """(B, N, K) boolean mask"""
        return ~torch.isnan(self.weights)

    @cached_property
    def joints_mask(self):
        """(B, K, 3) boolean mask"""
        return ~torch.isnan(self.joints)

    @cached_property
    def joints_mask_(self):
        """(B, K) boolean mask"""
        return self.joints_mask.all(-1, keepdim=False)

    @cached_property
    def joints_pose_matrix(self):
        """(B, K, 3, 3) rest-to-pose local rotation matrix"""
        return quat_to_matrix(self.joints_pose)

    @cached_property
    def joints_pose_inv(self):
        """(B, K, 4) pose-to-rest local rotations (quaternions)"""
        pose_inv = self.joints_pose.clone()
        pose_inv[..., 1:] *= -1
        return pose_inv

    @cached_property
    def joints_pose_inv_matrix(self):
        """(B, K, 3, 3) pose-to-rest local rotation matrix"""
        return quat_to_matrix(self.joints_pose_inv)

    @cached_property
    def joints_pose_rel2rest_matrix(self):
        """(B, K, 3, 3) rest-to-pose local rotation matrix (relative to rest parent)"""
        return quat_to_matrix(self.joints_pose_rel2rest)

    @cached_property
    def joints_pose_rel2rest_inv(self):
        """(B, K, 4) pose-to-rest local rotations (quaternions) (relative to rest parent)"""
        pose_inv = self.joints_pose_rel2rest.clone()
        pose_inv[..., 1:] *= -1
        return pose_inv

    @cached_property
    def joints_pose_rel2rest_inv_matrix(self):
        """(B, K, 3, 3) pose-to-rest local rotation matrix (relative to rest parent)"""
        return quat_to_matrix(self.joints_pose_rel2rest_inv)

    @cached_property
    def joints_transform_decomposed(self):
        """(B, K, 3+4+3=10) rest-to-pose transformation: translation, rotation (quaternion), scaling"""
        return decompose_transform(self.joints_transform, return_quat=True, return_concat=True)

    @cached_property
    def joints_transform_inv(self):
        """(B, K, 4, 4) pose-to-rest transformation matrix"""
        return self.joints_transform.inverse()

    @cached_property
    def joints_transform_inv_decomposed(self):
        """(B, K, 3+4+3=10) pose-to-rest transformation: translation, rotation (quaternion), scaling"""
        return decompose_transform(self.joints_transform_inv, return_quat=True, return_concat=True)

    @cached_property
    def verts_transform(self):
        """(B, N, 4, 4) rest-to-pose transformation"""
        return torch.einsum("bkij,bnk->bnij", self.joints_transform, self.weights)

    @cached_property
    def verts_transform_inv(self):
        """(B, N, 4, 4) pose-to-rest transformation"""
        return self.verts_transform.inverse()

    @cached_property
    def rest_joints(self) -> torch.Tensor:
        """(B, K, 3)"""
        # rest_joints = apply_transform(self.joints, self.joints_transform_inv)
        # # assert torch.allclose(rest_joints, default_collate([x.joints for x in self.meta.rest]), atol=1e-4)
        # warnings.warn(
        #     "Unreliable because `joints_transform` only contains standard bones. Please use `meta.rest.joints` instead.",
        # )
        return default_collate([x.joints for x in self.meta.rest])

    @cached_property
    def rest_joints_tail(self) -> torch.Tensor:
        """(B, K, 3)"""
        return default_collate([x.joints_tail for x in self.meta.rest])

    @cached_property
    def rest_verts(self) -> torch.Tensor:
        """(B, N, 3)"""
        # rest_verts = apply_transform(self.verts, self.verts_transform_inv)
        # # assert torch.allclose(rest_verts, default_collate([x.verts for x in self.meta.rest]), atol=1e-4)
        # warnings.warn(
        #     "Unreliable because `joints_transform` only contains standard bones. Please use `meta.rest.verts` instead.",
        # )
        rest_verts = default_collate([x.verts for x in self.meta.rest])
        return rest_verts

    @cached_property
    def non_rest_mask(self):
        """(B,)"""
        frames = default_collate(self.meta.frames)
        # pose_scalar = self.joints_pose[..., 0].nan_to_num(nan=1.0)
        pose_vector = self.joints_pose[..., 1:].nan_to_num(nan=0.0)
        mask = (pose_vector.abs() >= 1e-5).any(-1).any(-1)
        mask |= frames != -1
        return mask

    @cached_property
    def hips_plane(self):
        """[
        (B, 3) Hips position.
        (B, 3) Hips plane normal: direction: face-forward, magnitude: indicating skeleton size.
        (B, 2, 3) Hips plane tangent ends: vector direction: right to left.
        ]
        """
        bones_idx_dict = self.meta.bones_idx_dict[0]
        assert len(bones_idx_dict) == self.joints.shape[1]
        hips = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}Hips"]]
        rightupleg = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}RightUpLeg"]]
        leftupleg = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}LeftUpLeg"]]
        normal = get_hips_normal(hips, rightupleg, leftupleg)
        if torch.isnan(normal).any():
            raise RuntimeError("Cannot find hips plane")
        # normal = F.normalize(normal, dim=-1)
        tangent_ends = torch.stack((rightupleg, leftupleg), dim=-2)
        return hips, normal, tangent_ends

    @cached_property
    def hips_transform(self):
        """(B, 4, 4)
        transformation that moves Hips to the origin,
        and rotates body to face forward in the positive direction of z-axis
        and right-to-left oriented in the positive direction of x-axis.
        """
        bones_idx_dict = self.meta.bones_idx_dict[0]
        assert len(bones_idx_dict) == self.joints.shape[1]
        hips = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}Hips"]]
        rightupleg = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}RightUpLeg"]]
        leftupleg = self.joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}LeftUpLeg"]]
        if torch.isnan(torch.stack([hips, rightupleg, leftupleg])).any():
            raise RuntimeError("Cannot find hips plane")
        return get_hips_transform(hips, rightupleg, leftupleg)

    @cached_property
    def hips_transform_rest(self):
        """(B, 4, 4)
        transformation that moves Hips to the origin,O
        and rotates body to face forward in the positive direction of z-axis
        and right-to-left oriented in the positive direction of x-axis.
        """
        bones_idx_dict = self.meta.bones_idx_dict[0]
        assert len(bones_idx_dict) == self.rest_joints.shape[1]
        hips = self.rest_joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}Hips"]]
        rightupleg = self.rest_joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}RightUpLeg"]]
        leftupleg = self.rest_joints[:, bones_idx_dict[f"{MIXAMO_PREFIX}LeftUpLeg"]]
        if torch.isnan(torch.stack([hips, rightupleg, leftupleg])).any():
            raise RuntimeError("Cannot find hips plane")
        return get_hips_transform(hips, rightupleg, leftupleg)


@lru_cache(maxsize=99)
def _get_char_rest_data(
    char_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], Joint]:
    """char_id: required for cache"""
    armature_list = blender_utils.get_all_armature_obj()
    assert len(armature_list) == 1, f"Armature number is not 1: {armature_list}"
    armature_obj = armature_list[0]
    rest_bones, rest_bones_tail, bones_idx_dict = blender_utils.get_rest_bones(armature_obj)
    # rest_bones, bones_idx_dict = None, blender_utils.get_bones_idx_dict(armature_obj)
    rest_verts, faces, weights = blender_utils.get_rest_vertices(blender_utils.get_all_mesh_obj(), bones_idx_dict)
    kinematic_tree = build_skeleton(armature_obj, bones_idx_dict)
    return rest_verts, faces, rest_bones, rest_bones_tail, weights, bones_idx_dict, kinematic_tree


def reorganize_bone_data(
    bone_data: np.ndarray,
    bones_idx_dict: dict[str, int],
    template_dict: dict[str, int],
    kinematic_tree: Joint = None,
    is_bw=False,
) -> np.ndarray:
    """
    Args:
        bone_data: [..., k]
        bones_idx_dict: {bone_name: bone_idx}, len == k
    Returns:
        [..., `len(template_dict)`]
    """
    assert bone_data.shape[-1] == len(bones_idx_dict), "Unmatched bone data"
    new_data = np.full(bone_data.shape[:-1] + (len(template_dict),), fill_value=np.nan, dtype=bone_data.dtype)
    for bone_name, i in template_dict.items():
        if bone_name in bones_idx_dict:
            new_data[..., i] = bone_data[..., bones_idx_dict[bone_name]]
    # assert np.allclose(new_data, bone_data)
    if is_bw:
        # Some vertices are controlled by extra bones (sum(weights) << 1)
        if kinematic_tree is not None:
            # Tranfer weights of extra bones to their nearest valid parent
            for bone_name, i in bones_idx_dict.items():
                if bone_name not in template_dict:
                    valid_parent = kinematic_tree[bone_name].get_first_valid_parent(template_dict)
                    if valid_parent is not None:
                        new_data[..., template_dict[valid_parent.name]] += bone_data[..., i]
        # If an extra bone is seperated from character kinematic tree, mask them with NaN
        invalid = ~np.isclose(np.nan_to_num(new_data, copy=True, nan=0.0).sum(-1), 1, atol=1e-2)
        new_data[invalid] = np.nan
    return new_data


def animate_char(
    char_path: str,
    anim_path: str,
    template_dict: dict[str, int],
    load_fn: Callable[..., list[blender_utils.Object]],
    frame_list: int | list[int | None] = None,
    retarget=False,
    inplace=False,
    verbose=False,
):
    """
    frame_list: List of frame indices (relative to keyframes). \
        If `int`, use that frame only; \
        If `None`, use all valid frames; \
        If `None` in list, replace them with random valid frames; \
        If `-1` in list, use the canonical frame.
    """
    char_id = os.path.splitext(os.path.basename(char_path))[0]
    with HiddenPrints(enable=not verbose):
        # blender_utils.reset()
        blender_utils.remove_all()
        obj_list = load_fn(char_file=char_path, anim_file=anim_path, do_retarget=retarget, inplace=inplace)
        blender_utils.update()
        # print(list(bpy.context.scene.objects))

        keyframes = blender_utils.get_keyframes(obj_list, mute_global_anim=(not retarget) and inplace)
        assert len(keyframes) > 0, "No keyframe found"
        frame_start, frame_end = min(keyframes), max(keyframes)
        keyframes = list(range(frame_start, frame_end + 1))
        if keyframes == [1, 2]:  # for a-pose, etc.
            keyframes = [2]
        if frame_list is None:
            frame_list = keyframes
        else:
            if isinstance(frame_list, int):
                frame_list = [frame_list]
            try:
                frame_list = [idx if idx is None or idx == -1 else keyframes[idx] for idx in frame_list]
            except IndexError:
                raise IndexError(f"Frame index out of range: {frame_list} with keyframes length {len(keyframes)}")
        if None in frame_list:
            rand_num = frame_list.count(None)
            frame_index_randoms = list(np.random.choice(keyframes, size=rand_num, replace=rand_num > len(keyframes)))
            frame_list = [frame_index_randoms.pop() if idx is None else idx for idx in frame_list]
        frame_list = list(map(int, frame_list))
        assert frame_list and all(
            (frame_start <= i <= frame_end or i == -1) for i in frame_list
        ), f"Invalid frame index: {frame_list}"

        # blender_utils.select_mesh(obj_list, deselect_first=True)
        (
            rest_verts,
            faces,
            rest_bones,
            rest_bones_tail,
            weights,
            bones_idx_dict,
            kinematic_tree,
        ) = _get_char_rest_data(char_id)
        rest_verts = rest_verts.astype(np.float32)
        rest_bones = reorganize_bone_data(rest_bones.T, bones_idx_dict, template_dict=template_dict).T
        rest_bones = rest_bones.astype(np.float32)
        rest_bones_tail = reorganize_bone_data(rest_bones_tail.T, bones_idx_dict, template_dict=template_dict).T
        rest_bones_tail = rest_bones_tail.astype(np.float32)
        rest_bones_quat = np.zeros((len(template_dict), 4)).astype(np.float32)
        rest_bones_quat[:, 0] = 1.0
        rest_bones_quat_rel2rest = rest_bones_quat.copy()
        rest_bones_transform = np.tile(np.eye(4), (len(template_dict), 1, 1)).astype(np.float32)
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-10)
        # assert np.allclose(weights.sum(-1), 1)
        weights = reorganize_bone_data(
            weights, bones_idx_dict, template_dict=template_dict, kinematic_tree=kinematic_tree, is_bw=True
        ).astype(np.float32)
        assert not np.isnan(weights).all(), "All weights are NaN after processing"
        verts_list: list[np.ndarray] = []
        joints_list: list[np.ndarray] = []
        joints_tail_list: list[np.ndarray] = []
        joints_quat_list: list[np.ndarray] = []
        joints_quat_rel2rest_list: list[np.ndarray] = []
        joints_transform_list: list[np.ndarray] = []
        for frame in frame_list:
            if frame == -1:
                verts = rest_verts
                bones = rest_bones
                bones_tail = rest_bones_tail
                bones_quat = rest_bones_quat
                bones_quat_rel2rest = rest_bones_quat_rel2rest
                bones_transform = rest_bones_transform
            else:
                bpy.context.scene.frame_set(frame)
                blender_utils.update()
                verts = blender_utils.get_pose_vertices(blender_utils.get_all_mesh_obj(obj_list))
                bones, bones_tail, bones_quat, bones_quat_rel2rest, bones_transform = blender_utils.get_pose_bones(
                    blender_utils.get_armature_obj(obj_list)
                )
                bones = reorganize_bone_data(bones.T, bones_idx_dict, template_dict=template_dict).T
                bones_tail = reorganize_bone_data(bones_tail.T, bones_idx_dict, template_dict=template_dict).T
                bones_quat = reorganize_bone_data(bones_quat.T, bones_idx_dict, template_dict=template_dict).T
                bones_quat_rel2rest = reorganize_bone_data(
                    bones_quat_rel2rest.T, bones_idx_dict, template_dict=template_dict
                ).T
                bones_transform = reorganize_bone_data(bones_transform.T, bones_idx_dict, template_dict=template_dict).T
            verts_list.append(verts.astype(np.float32))
            joints_list.append(bones.astype(np.float32))
            joints_tail_list.append(bones_tail.astype(np.float32))
            joints_quat_list.append(bones_quat.astype(np.float32))
            joints_quat_rel2rest_list.append(bones_quat_rel2rest.astype(np.float32))
            joints_transform_list.append(bones_transform.astype(np.float32))

    # trimesh.Scene([trimesh.PointCloud(verts_list[-1]),trimesh.PointCloud(np.nan_to_num(joints_list[-1]))]).export("test.glb")
    return (
        rest_verts,
        rest_bones,
        rest_bones_tail,
        faces,
        weights,
        verts_list,
        joints_list,
        joints_tail_list,
        joints_quat_list,
        joints_quat_rel2rest_list,
        joints_transform_list,
        frame_list,
        keyframes,
    )


class MixamoDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        character_subdir="character_refined",
        extra_character_dir: list[str] = None,
        animation_subdir="animation",
        extra_animation_dir: list[str] = None,
        same_animation_first=True,
        retarget=False,
        inplace=False,
        sample_points=2048,
        hands_resample_ratio=0.0,
        geo_resample_ratio=0.0,
        sample_vertices=-1,
        sample_frames=1,
        include_rest=True,
        get_normals=False,
        bones_idx_dict: dict[str, int] = BONES_IDX_DICT,
        split="train",
        train_with_val=False,
    ):
        super().__init__()
        assert os.path.isdir(data_dir), f"Invalid {data_dir=}"
        self.data_dir = data_dir
        assert sample_points > 0, f"Invalid {sample_points=}"
        self.sample_points = int(sample_points)
        assert 0.0 <= hands_resample_ratio < 1.0, f"Invalid {hands_resample_ratio=}"
        self.hands_resample_ratio = float(hands_resample_ratio)
        assert 0.0 <= geo_resample_ratio < 1.0, f"Invalid {geo_resample_ratio=}"
        assert (
            self.hands_resample_ratio + geo_resample_ratio < 1.0
        ), f"{self.hands_resample_ratio+geo_resample_ratio=} >= 1.0"
        self.geo_resample_ratio = float(geo_resample_ratio)
        assert sample_vertices != 0, f"Invalid {sample_vertices=}"
        if sample_vertices < 0:
            print("Using all vertices of each character (typically used in eval mode), only support batch size 1!")
        self.sample_vertices = int(sample_vertices)
        self.get_normals = get_normals
        self.same_animation_first = same_animation_first
        self.retarget = retarget
        self.inplace = inplace
        if isinstance(sample_frames, (int, float)):
            sample_frames = [None] * int(sample_frames)
        else:
            sample_frames = list(sample_frames)
        self.include_rest = include_rest
        if self.include_rest and -1 not in sample_frames:
            sample_frames = [-1] + list(sample_frames)
        self.sample_frames = tuple(sample_frames)

        assert split in ("train", "val", "test"), f"Invalid {split=}"
        self.split = split
        self.character_list = sorted(glob(os.path.join(self.data_dir, character_subdir, "*.fbx")))
        assert len(self.character_list) > 0, f"No character found in {self.data_dir}"
        if self.split == "train":
            if not train_with_val:
                self.character_list = self.character_list[:-5]
        else:
            self.character_list = self.character_list[-5:]

        if self.split == "train" and extra_character_dir:
            if isinstance(extra_character_dir, str):
                extra_character_dir = [extra_character_dir]
            for extra_dir in extra_character_dir:
                extra_char_list = sorted(glob(os.path.join(extra_dir, "*.fbx")))
                assert len(extra_char_list) > 0, f"No extra character found in {extra_dir}"
                self.character_list.extend(extra_char_list)

        self.animation_list = sorted(glob(os.path.join(self.data_dir, animation_subdir, "*.fbx")))
        assert len(self.animation_list) > 0, f"No animation found in {self.data_dir}"
        self.animation_list_full = self.animation_list.copy()
        if self.split != "train":
            self.animation_list = self.animation_list[:10]
        elif extra_character_dir and len(extra_char_list) > 1000:
            self.animation_list = self.animation_list[: len(self.animation_list) // 100]

        self.extra_anim_list = None
        if self.split == "train" and extra_animation_dir:
            if isinstance(extra_animation_dir, str):
                extra_animation_dir = [extra_animation_dir]
            for extra_dir in extra_animation_dir:
                self.extra_anim_list = sorted(glob(os.path.join(extra_dir, "*.fbx")))
                assert len(self.extra_anim_list) > 0, f"No extra animation found in {extra_dir}"
                self.animation_list.extend(self.extra_anim_list)

            # self.animation_list = list(filter(lambda x: "i-pose" in x, self.animation_list))
            # self.animation_list = self.animation_list[:1]

        self.load_fn = blender_utils.load_mixamo_anim
        self.bones_idx_dict = OrderedDict(bones_idx_dict)

    def __len__(self):
        return len(self.character_list) * len(self.animation_list)

    def get_index(self, index: int):
        if self.same_animation_first:
            anim_index, char_index = divmod(index, len(self.character_list))
        else:
            char_index, anim_index = divmod(index, len(self.animation_list))
        return char_index, anim_index

    def shuffle_train_anim(self):
        if self.split != "train":
            return
        if len(self.animation_list_full) > len(self.animation_list):
            k = len(self.animation_list)
            if self.extra_anim_list:
                k -= len(self.extra_anim_list)
                if k <= 0:
                    print("Warning: `animation_list` may be modified after initialization, cannot shuffle.")
                    return
            self.animation_list = random.sample(self.animation_list_full, k=k)
            if self.extra_anim_list:
                self.animation_list.extend(self.extra_anim_list)

    def __getitem__(self, index: int):
        char_index, anim_index = self.get_index(index)
        (
            rest_verts,
            rest_joints,
            rest_joints_tail,
            faces,
            weights,
            verts_list,
            joints_list,
            joints_tail_list,
            joints_quat_list,
            joints_quat_rel2rest_list,
            joints_transform_list,
            frame_list,
            keyframes,
        ) = animate_char(
            self.character_list[char_index],
            self.animation_list[anim_index],
            template_dict=self.bones_idx_dict,
            frame_list=self.sample_frames,
            load_fn=self.load_fn,
            retarget=self.retarget,
            inplace=self.inplace,
            verbose=False,
        )

        if self.sample_vertices > 0:
            verts_num = verts_list[0].shape[0]
            verts_i = np.random.choice(verts_num, self.sample_vertices, replace=self.sample_vertices > verts_num)
            weights = weights[verts_i, :]
        else:  # to use all vertices
            verts_i = None
            # weights = weights[np.newaxis]  # broadcasting is ok for single batch

        pts_list = []
        verts_normal_list = [] if self.get_normals else None
        pts_normal_list = [] if self.get_normals else None
        for i, verts in enumerate(verts_list):
            mesh = trimesh.Trimesh(verts, faces, process=False, maintain_order=True)
            # mesh.export(file_obj=f"{os.path.splitext(os.path.basename(self.character_list[char_index]))[0]}-{anim_index:04d}-{frame_list[i]:03d}.ply")  # fmt: skip

            if self.get_normals:
                verts_normal = np.array(mesh.vertex_normals)
                if verts_i is not None:
                    verts_normal = verts_normal[verts_i]
                verts_normal_list.append(verts_normal.astype(np.float32))

            if verts_i is not None:
                verts_list[i] = verts[verts_i]

            if self.hands_resample_ratio > 0:
                joints_tail = joints_tail_list[i]
                attn_centers = np.array(
                    [
                        joints_tail[self.bones_idx_dict[f"{MIXAMO_PREFIX}LeftHand"]],
                        joints_tail[self.bones_idx_dict[f"{MIXAMO_PREFIX}RightHand"]],
                    ]
                )
                if np.isnan(attn_centers).any():
                    attn_ratio = 0.0
                    attn_centers = None
                else:
                    attn_ratio = self.hands_resample_ratio
            else:
                attn_ratio = 0.0
                attn_centers = None
            pts = sample_mesh(
                mesh,
                self.sample_points,
                get_normals=self.get_normals,
                attn_ratio=attn_ratio,
                attn_centers=attn_centers,
                attn_geo_ratio=self.geo_resample_ratio,
            )
            if self.get_normals:
                pts, normals = np.split(pts, 2, axis=-1)
                pts_normal_list.append(normals.astype(np.float32))
            pts_list.append(pts.astype(np.float32))

        weights = [weights] * len(frame_list)  # enable stacking across batches

        meta = MetaData(
            index=index,
            char_id=os.path.splitext(os.path.basename(self.character_list[char_index]))[0],
            anim_id=os.path.splitext(os.path.basename(self.animation_list[anim_index]))[0],
            char_path=os.path.abspath(self.character_list[char_index]),
            anim_path=os.path.abspath(self.animation_list[anim_index]),
            frames=frame_list,
            keyframes=keyframes,
            faces=faces,
            rest=PoseData(verts=rest_verts, joints=rest_joints, joints_tail=rest_joints_tail),
            bones_idx_dict=self.bones_idx_dict,
        )
        # After batchified by `collate`, values in data will follow their type hinting (torch.Tensor), but not yet now (lists of np.ndarray)
        data = PoseData(
            pts=pts_list,
            pts_normal=pts_normal_list,
            verts=verts_list,
            verts_normal=verts_normal_list,
            weights=weights,
            joints=joints_list,
            joints_tail=joints_tail_list,
            joints_pose=joints_quat_list,
            joints_pose_rel2rest=joints_quat_rel2rest_list,
            joints_transform=joints_transform_list,
            meta=meta,
        )
        return data


def collate(batch: list[PoseData]):
    if batch[0].meta is not None:
        meta_all = []
        for meta in [data.meta for data in batch]:
            frame_num = len(meta.frames)
            meta_all.append({k: ([v] * frame_num if k != "frames" else v) for k, v in meta.__dict__.items()})
        meta_all = {k: [x for meta in meta_all for x in meta[k]] for k in meta_all[0]}  # stack & flatten
        assert (
            len(set(map(len, meta_all.values()))) == 1
        )  # All lists in the dict are now aligned and have the same length
        meta_all = MetaData(**meta_all)
    else:
        meta_all = None

    data_all = [
        {
            k: np.nan if v is None else default_collate(v)
            for k, v in data.__dict__.items()
            if not isinstance(v, MetaData)
        }
        for data in batch
    ]
    data_all: dict[str, torch.Tensor] = default_collate(data_all)
    b, t = next(iter(data_all.values())).shape[:2]
    # assert b == len(batch) and t == len(batch[0].meta.frames)
    for k, v in data_all.items():
        if len(v.shape) == 1 and torch.isnan(v).all():
            data_all[k] = None
        else:
            data_all[k] = v.reshape(b * t, *v.shape[2:])  # new_batch_size = dataloader.batch_size * frames
    posedata = PoseData(**data_all, meta=meta_all)
    # Prefetch cached_property
    posedata.hips_transform
    posedata.weights_mask
    posedata.joints_mask
    return posedata


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # print(torch.utils.data.get_worker_info())
    blender_utils.reset()


def connect_loss_fn(joints: torch.Tensor, joints_tail: torch.Tensor = None, connected_idx_pairs=CONNECTED_IDX_PAIRS):
    if joints_tail is None:
        joints, joints_tail = joints[..., :3], joints[..., 3:]
    idx_pairs = torch.tensor(connected_idx_pairs, device=joints.device)
    idx_tail, idx_head = idx_pairs[..., 0], idx_pairs[..., 1]
    return F.smooth_l1_loss(joints[..., idx_head, :], joints_tail[..., idx_tail, :])


def rest_prior_loss_fn(
    rest_joints: torch.Tensor,
    rest_joints_tail: torch.Tensor = None,
    symmetric_idx_paris=SYMMETRIC_IDX_PAIRS,
    axis_idx=AXIS_IDX,
    # reverse_dir_idx_pairs=REVERSE_DIR_IDX_PAIRS,
    # same_dir_idx_groups=SAME_DIR_IDX_GROUPS,
):
    if rest_joints_tail is not None:
        rest_joints = torch.cat((rest_joints, rest_joints_tail), dim=-1)
    device = rest_joints.device
    # symmetric about yz plane
    symmetric_idx_pairs = torch.tensor(symmetric_idx_paris, device=device)
    symmetric_loss = F.smooth_l1_loss(
        rest_joints[..., symmetric_idx_pairs[:, 0], 0], -rest_joints[..., symmetric_idx_pairs[:, 1], 0]
    )
    # on the x axis
    axis_idx = torch.tensor(axis_idx, device=device)
    axis_loss = F.smooth_l1_loss(rest_joints[..., axis_idx, 0], -rest_joints[..., axis_idx, 0])
    # # reverse direction
    # reverse_idx = torch.tensor(reverse_dir_idx_pairs, device=device)
    # reverse_vectors1 = rest_joints[..., reverse_idx[:, 0], 3:] - rest_joints[..., reverse_idx[:, 0], :3]
    # reverse_vectors2 = rest_joints[..., reverse_idx[:, 1], 3:] - rest_joints[..., reverse_idx[:, 1], :3]
    # reverse_loss = (1 - F.cosine_similarity(reverse_vectors1, -reverse_vectors2, dim=-1)).mean()
    # # same direction
    # same_loss = 0.0
    # for same_idx_group in same_dir_idx_groups:
    #     same_idx = torch.tensor(same_idx_group, device=device)
    #     same_vectors = F.normalize(rest_joints[..., same_idx, 3:] - rest_joints[..., same_idx, :3], dim=-1)
    #     same_loss += F.smooth_l1_loss(same_vectors, same_vectors.mean(-2, keepdim=True).expand_as(same_vectors))
    # # same_loss /= len(same_dir_idx_groups)
    return symmetric_loss + axis_loss  # + reverse_loss + same_loss


def keep_exists(data: torch.Tensor, joints_exist_mask=JOINTS_EXIST_MASK):
    """
    Args:
        data: (B, `len(joints_exist_mask)`, ...)
    Returns:
        data: (B, `len(list(filter(None, joints_exist_mask)))`, ...)
    """
    mask = torch.tensor(joints_exist_mask, device=data.device)
    assert data.shape[1] == len(joints_exist_mask)
    data = data.transpose(0, 1)
    data = data[mask]
    data = data.transpose(0, 1)
    return data.contiguous()


if __name__ == "__main__":
    import json

    from pytorch3d.transforms import Rotate, Transform3d, Translate, quaternion_to_matrix, random_rotations
    from torch.utils.data import DataLoader

    from util.utils import compose_transform, fix_random, get_normalize_transform, pose_local_to_global

    fix_random()
    dataset = MixamoDataset(
        "../data/Mixamo",
        sample_frames=1,
        sample_points=32768,
        hands_resample_ratio=0.5,
        sample_vertices=-1,
        include_rest=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    print(len(dataset))

    # Test transformation matrix
    assert dataset.sample_vertices == -1 and dataloader.batch_size == 1
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        data: PoseData
        meta = data.meta

        rest_joints, posed_joints = data.rest_joints[-1], data.joints[-1]
        rest_verts, posed_verts = data.rest_verts[-1], data.verts[-1]
        faces = meta.faces[-1]
        weights = data.weights[-1]
        pose = data.joints_transform[-1]
        if not data.weights_mask.all():
            continue

        # pose1 = compose_transform(decompose_transform(pose, return_quat=True, return_concat=True))
        # assert torch.allclose(pose, pose1, atol=1e-5)
        # pose2 = compose_transform(decompose_transform(pose.numpy(), return_quat=True, return_concat=True))
        # assert np.allclose(pose.numpy(), pose2, atol=1e-5)
        # pose_decompose1 = decompose_transform(pose, return_quat=True, return_concat=True)
        # pose_decompose2 = torch.from_numpy(decompose_transform(pose.numpy(), return_quat=True, return_concat=True))
        # # import bpy, mathutils
        # # pose_decompose3 = torch.from_numpy(np.concatenate(mathutils.Matrix(pose[0].numpy()).decompose(), -1))

        pose_inv_decomposed = data.joints_transform_inv_decomposed[-1]
        transl = pose_inv_decomposed[..., :3]
        rotation = pose_inv_decomposed[..., 3:-3]
        scaling = pose_inv_decomposed[..., -3:]
        assert torch.allclose(scaling, torch.ones_like(scaling), atol=1e-5)
        pose_transform_inv = Rotate(quaternion_to_matrix(rotation).transpose(-1, -2)).compose(Translate(transl))
        assert torch.allclose(pose_transform_inv.get_matrix().transpose(-1, -2), pose.inverse(), atol=1e-5)

        pose_matrix = data.joints_transform[-1]
        pose_inv_matrix = data.joints_transform_inv[-1]
        parents = torch.tensor(KINEMATIC_TREE.parent_indices)
        pose_matrix_, posed_joints_ = pose_local_to_global(
            data.joints_pose_matrix[-1], rest_joints, parents, global_transl=posed_joints[0] - rest_joints[0]
        )
        assert torch.allclose(posed_joints, posed_joints_, atol=1e-3)
        pose_inv_matrix_, rest_joints_ = pose_local_to_global(
            data.joints_pose_rel2rest_inv_matrix[-1],
            posed_joints,
            parents,
            global_transl=rest_joints[0] - posed_joints[0],
            relative_to_source=False,
        )
        assert torch.allclose(rest_joints, rest_joints_, atol=1e-3)
        pose_inv_matrix__, rest_joints__ = pose_local_to_global(
            data.joints_pose_inv_matrix[-1],
            posed_joints,
            parents,
            global_transl=rest_joints[0] - posed_joints[0],
            relative_to_source=True,
        )
        assert torch.allclose(rest_joints, rest_joints__, atol=1e-3)

        # rotate = Rotate(R=random_rotations(len(data)))
        # norm = get_normalize_transform(rotate.transform_points(data.pts), keep_ratio=True)
        # transform = rotate.compose(norm)
        # transform_decomposed = decompose_transform(
        #     transform.get_matrix().transpose(-1, -2), return_quat=True, return_concat=True
        # )
        # assert torch.allclose(transform_decomposed[:, -3], transform_decomposed[:, -2]) and torch.allclose(
        #     transform_decomposed[:, -2], transform_decomposed[:, -1]
        # )  # keep_ratio=True

        # posed_joints = transform[-1].transform_points(posed_joints)
        # posed_verts = transform[-1].transform_points(posed_verts)
        # pose = transform[-1].get_matrix().transpose(-1, -2) @ pose

        # LBS manually
        # ! PyTorch3D wants a right-multiplied transformation matrix (applied on row vectors) <https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.Transform3d>
        pose_transform = Transform3d(matrix=pose.transpose(-1, -2))
        posed_joints_ = pose_transform.transform_points(rest_joints.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(posed_joints - posed_joints_).abs().mean()}")
        # trimesh.Trimesh(posed_joints, process=False, maintain_order=True).export("posed_joints_gt.ply")
        # trimesh.Trimesh(posed_joints_, process=False, maintain_order=True).export("posed_joints.ply")
        rest_joints_ = pose_transform.inverse().transform_points(posed_joints.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(rest_joints - rest_joints_).abs().mean()}")
        # trimesh.Trimesh(rest_joints, process=False, maintain_order=True).export("rest_joints_gt.ply")
        # trimesh.Trimesh(rest_joints_, process=False, maintain_order=True).export("rest_joints.ply")

        verts_transform = Transform3d(matrix=torch.einsum("kij,nk->nij", pose, weights).transpose(-1, -2))
        posed_verts_ = verts_transform.transform_points(rest_verts.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(posed_verts - posed_verts_).abs().mean()}")
        # trimesh.Trimesh(posed_verts, faces, process=False, maintain_order=True).export("posed_verts_gt.ply")
        # trimesh.Trimesh(posed_verts_, faces, process=False, maintain_order=True).export("posed_verts.ply")
        rest_verts_ = verts_transform.inverse().transform_points(posed_verts.unsqueeze(1)).squeeze(1)
        tqdm.write(f"{(rest_verts - rest_verts_).abs().mean()}")
        # trimesh.Trimesh(rest_verts, faces, process=False, maintain_order=True).export("rest_verts_gt.ply")
        # trimesh.Trimesh(rest_verts_, faces, process=False, maintain_order=True).export("rest_verts.ply")

    # Count vertices of all characters
    verts_num = {}
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        meta, verts = data.meta, data.verts
        verts_num[meta.char_id[0]] = verts.shape[1]
        if i == len(dataset.character_list) - 1:
            break
    print(np.min(list(verts_num.values())))
    print(np.max(list(verts_num.values())))
    print(np.mean(list(verts_num.values())))
    with open("./character_verts.json", "w") as f:
        json.dump(verts_num, f, indent=4, ensure_ascii=False)

    # Count frames of all animations
    dataset.same_animation_first = False
    frame_num = {}
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        meta = data.meta
        frame_num[meta.anim_id[0]] = len(meta.keyframes[0])
        if i == len(dataset.animation_list) - 1:
            break
    print(np.min(list(frame_num.values())))
    print(np.max(list(frame_num.values())))
    print(np.mean(list(frame_num.values())))
    with open("./animation_frames.json", "w") as f:
        json.dump(frame_num, f, indent=4, ensure_ascii=False)
