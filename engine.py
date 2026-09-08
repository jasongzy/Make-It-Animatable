import json
import math
import os
from dataclasses import dataclass
from functools import cached_property

import torch
import torch.amp
import torch.nn.functional as F
from pytorch3d.transforms import Rotate, Transform3d, random_rotations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util.lr_sched as lr_sched
import util.misc as misc
from model import PCAE, Output
from util.dataset_mixamo import PoseData
from util.utils import (
    apply_transform,
    decompose_transform,
    get_normalize_transform,
    matrix_to_ortho6d,
    matrix_to_quat,
    ortho6d_to_matrix,
    pose_local_to_global,
    pose_rot_to_global,
    quat_to_matrix,
    quat_transl_to_dualquat,
    robust_dataloader,
    to_pose_local,
    to_pose_matrix,
)


@dataclass(frozen=False)
class GT:
    data: PoseData
    global_transform: Transform3d
    global_transform_rest: Transform3d
    device: torch.device

    d_real: torch.Tensor = None
    d_fake: torch.Tensor = None

    @cached_property
    def global_transform_matrix(self):
        """(B, 4, 4)"""
        return self.global_transform.get_matrix().transpose(-1, -2).to(self.device, non_blocking=True)

    @cached_property
    def global_transform_inv_matrix(self):
        """(B, 4, 4)"""
        return self.global_transform.inverse().get_matrix().transpose(-1, -2).to(self.device, non_blocking=True)

    @cached_property
    def global_transform_rest_matrix(self):
        """(B, 4, 4)"""
        return self.global_transform_rest.get_matrix().transpose(-1, -2).to(self.device, non_blocking=True)

    @cached_property
    def global_transform_rest_inv_matrix(self):
        """(B, 4, 4)"""
        return self.global_transform_rest.inverse().get_matrix().transpose(-1, -2).to(self.device, non_blocking=True)

    @cached_property
    def bw(self):
        """(B, N, K)"""
        return self.data.weights.to(self.device, non_blocking=True)

    @cached_property
    def bw_mask(self):
        """(B, N, K)"""
        return self.data.weights_mask.to(self.device, non_blocking=True)

    @cached_property
    def joints_raw(self):
        """(B, K, 3)"""
        return self.data.joints.to(self.device, non_blocking=True)

    @cached_property
    def joints(self):
        """(B, K, 3)"""
        return self.global_transform.transform_points(self.data.joints).to(self.device, non_blocking=True)

    @cached_property
    def joints_tail(self):
        """(B, K, 3)"""
        return self.global_transform.transform_points(self.data.joints_tail).to(self.device, non_blocking=True)

    @cached_property
    def joints_dual(self):
        """(B, K, 6)"""
        return torch.cat([self.joints, self.joints_tail], dim=-1)

    @cached_property
    def joints_mask(self):
        """(B, K)"""
        return self.data.joints_mask_.to(self.device, non_blocking=True)

    @cached_property
    def rest_joints_raw(self):
        """(B, K, 3)"""
        return self.data.rest_joints.to(self.device, non_blocking=True)

    @cached_property
    def rest_joints(self):
        """(B, K, 3)"""
        rest_joints = self.data.rest_joints
        rest_joints = self.global_transform_rest.transform_points(rest_joints)
        # assert torch.allclose(rest_joints[:, 0], torch.zeros_like(rest_joints[:, 0]), atol=1e-5)
        return rest_joints.to(self.device, non_blocking=True)

    @cached_property
    def rest_joints_tail(self):
        """(B, K, 3)"""
        rest_joints_tail = self.data.rest_joints_tail
        rest_joints_tail = self.global_transform_rest.transform_points(rest_joints_tail)
        return rest_joints_tail.to(self.device, non_blocking=True)

    @cached_property
    def rest_joints_(self):
        """(B, K, 3) transformed by global_transform of posed joints"""
        rest_joints = self.data.rest_joints
        rest_joints = self.global_transform.transform_points(rest_joints)
        return rest_joints.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_local_matrix(self):
        """(B, K, 3, 3)"""
        pose = self.data.joints_pose_inv_matrix
        pose = torch.einsum("bij,bnjk->bnik", self.global_transform.get_matrix().transpose(-1, -2)[..., :3, :3], pose)
        pose = torch.einsum(
            "bnij,bjk->bnik", pose, self.global_transform.inverse().get_matrix().transpose(-1, -2)[..., :3, :3]
        )
        root_trans = torch.einsum(
            "bij,bjk->bik",
            self.global_transform_rest.get_matrix().transpose(-1, -2)[..., :3, :3],
            self.global_transform.inverse().get_matrix().transpose(-1, -2)[..., :3, :3],
        )
        pose = torch.cat([torch.einsum("bij,bnjk->bnik", root_trans, pose[:, :1]), pose[:, 1:]], dim=1)
        return pose.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_local_quat(self):
        """(B, K, 4)"""
        # pose = self.data.joints_pose_inv
        pose = matrix_to_quat(self.pose_p2r_local_matrix)
        return pose.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_local_ortho6d(self):
        """(B, K, 6)"""
        return matrix_to_ortho6d(self.pose_p2r_local_matrix).to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r(self):
        """(B, K, 4, 4)"""
        pose = self.data.joints_transform_inv
        pose = torch.einsum("bij,bnjk->bnik", self.global_transform_rest.get_matrix().transpose(-1, -2), pose)
        pose = torch.einsum("bnij,bjk->bnik", pose, self.global_transform.inverse().get_matrix().transpose(-1, -2))
        return pose.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_rot(self):
        """(B, K, 3, 3)"""
        return self.pose_p2r[..., :3, :3].to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_quat(self):
        """(B, K, 4)"""
        pose = matrix_to_quat(self.pose_p2r_rot)
        return pose.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_ortho6d(self):
        """(B, K, 6)"""
        pose = matrix_to_ortho6d(self.pose_p2r_rot)
        return pose.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_transl_quat(self):
        """(B, K, 3+4=7)"""
        # pose_decomposed = self.data.joints_transform_inv_decomposed
        pose_decomposed = decompose_transform(self.pose_p2r)
        pose_decomposed = pose_decomposed[..., :-3]  # remove scalings, as they should all be 1.0
        # pose_decomposed_gt[..., :3] /= 100.0
        return pose_decomposed.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_dualquat(self):
        """(B, K, 4+4=8)"""
        transl, rotation = self.pose_p2r_transl_quat.split([3, 4], dim=-1)
        dualquat = quat_transl_to_dualquat(quat=rotation, transl=transl, transl_first=True)
        # transl_, rotation_ = dualquat_to_quat_trans(dualquat, transl_first=True)
        # assert torch.allclose(rotation, rotation_)
        # assert torch.allclose(transl, transl_, atol=1e-5)
        return dualquat.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_transl_matrix(self):
        """(B, K, 3+3*3=12)"""
        transl_matrix = decompose_transform(self.pose_p2r, return_quat=False, return_concat=True)[..., :-3]
        return transl_matrix.to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_transl_ortho6d(self):
        """(B, K, 3+6=9)"""
        transl, rotation = decompose_transform(self.pose_p2r, return_quat=False, return_concat=False)[:2]
        rotation = matrix_to_ortho6d(rotation)
        return torch.cat([transl, rotation], dim=-1).to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_target_quat(self):
        """(B, K, 3+4=7)"""
        target = self.rest_joints
        quat = decompose_transform(self.pose_p2r, return_quat=True, return_concat=False)[1]
        # from util.utils import compose_transform_trt
        # pose_p2r_ = compose_transform_trt([self.joints, quat, target])
        # assert torch.allclose(pose_p2r_, self.pose_p2r, atol=1e-5)
        return torch.cat([target, quat], dim=-1).to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_target_matrix(self):
        """(B, K, 3+3*3=12)"""
        target = self.rest_joints
        matrix = decompose_transform(self.pose_p2r, return_quat=False, return_concat=False)[1]
        return torch.cat([target, matrix.reshape(*matrix.shape[:-2], 9)], dim=-1).to(self.device, non_blocking=True)

    @cached_property
    def pose_p2r_target_ortho6d(self):
        """(B, K, 3+6=9)"""
        target = self.rest_joints
        rotation = decompose_transform(self.pose_p2r, return_quat=False, return_concat=False)[1]
        rotation = matrix_to_ortho6d(rotation)
        return torch.cat([target, rotation], dim=-1).to(self.device, non_blocking=True)

    @cached_property
    def global_inv(self):
        """(B, K, 3+4+1=8)"""
        global_inv = decompose_transform(self.global_transform.inverse().get_matrix().transpose(-1, -2))
        # remove scalings of yz, as xyz share the same values (keep_ratio=True)
        global_inv = global_inv.unsqueeze(-2)[..., :-2]
        return global_inv.to(self.device, non_blocking=True)

    @cached_property
    def non_rest_mask(self):
        """(B,)"""
        return self.data.non_rest_mask.to(self.device, non_blocking=True)


discriminator = {}


def get_discriminator(device: torch.device = None, distributed=False, gpus: tuple[int] = None):
    global discriminator
    if "rest_joints_model" in discriminator:
        model_D = discriminator["rest_joints_model"]
        optimizer_D = discriminator["rest_joints_optimizer"]
        # adversarial_loss = discriminator["loss_fn"]
    else:
        from model import JointsDiscriminatorAttn

        assert device is not None and gpus is not None
        model_D = JointsDiscriminatorAttn()
        # model_D.load_state_dict(torch.load("output/model_d.pth", map_location="cpu"))
        model_D.to(device, non_blocking=True)
        model_without_ddp = model_D
        if distributed:
            model_D = torch.nn.parallel.DistributedDataParallel(model_D, device_ids=gpus)
            model_without_ddp = model_D.module
        # optimizer_D = torch.optim.Adam(model_without_ddp.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.AdamW(model_without_ddp.parameters(), lr=1e-4)
        # adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
        discriminator.update({"rest_joints_model": model_D, "rest_joints_optimizer": optimizer_D})
    return model_D, optimizer_D


def get_loss_discriminator(model_D: torch.nn.Module, gt: GT):
    from model import adv_loss_d
    from util.dataset_mixamo import keep_exists

    assert gt.d_real is not None and gt.d_fake is not None
    input_real, input_fake = gt.d_real.detach(), gt.d_fake.detach()
    mask_real = mask_fake = None
    model_D.train()

    additional_fake = torch.cat((gt.joints, gt.joints_tail), dim=-1).detach()
    additional_fake = additional_fake[gt.non_rest_mask]
    # additional_fake = keep_exists(additional_fake)
    additional_fake = additional_fake.nan_to_num(nan=0.0)
    input_fake = torch.cat((gt.d_fake, additional_fake), dim=0)

    mask_real = gt.joints_mask.clone()
    mask_fake = mask_real.clone()
    additional_fake_mask = mask_real[gt.non_rest_mask]
    mask_fake = torch.cat((mask_real, additional_fake_mask), dim=0)

    loss_D = adv_loss_d(model_D(input_real, mask=mask_real), model_D(input_fake, mask=mask_fake))
    return loss_D


def get_loss(output: Output, gt: GT, criterion: torch.nn.Module, args):
    from util.dataset_mixamo import rest_prior_loss_fn

    if args.use_additional_bones:
        from util.dataset_mixamo_additional import KINEMATIC_TREE, connect_loss_fn
    else:
        from util.dataset_mixamo import KINEMATIC_TREE, connect_loss_fn

    bw, joints, global_trans, pose_trans = output
    loss = 0.0
    loss_value_dict: dict[str, float] = {}
    vis_data = {}

    if args.predict_bw:
        loss_bw = criterion(bw[gt.bw_mask], gt.bw[gt.bw_mask])
        loss += loss_bw
        loss_value_dict["loss/bw"] = loss_bw.item()

    if args.predict_joints:
        joints_gt = gt.joints_dual if args.predict_joints_tail else gt.joints
        if joints.ndim == 3:
            loss_joints = criterion(joints[gt.joints_mask], joints_gt[gt.joints_mask])
            loss += 1e-1 * loss_joints
            loss_value_dict["loss/joints"] = loss_joints.item()
            if args.predict_joints_tail and args.use_joints_connect_loss:
                loss_joints_connect = connect_loss_fn(joints)
                loss += 1e-1 * loss_joints_connect
                loss_value_dict["loss/joints_connect"] = loss_joints_connect.item()
            if args.use_joints_rest_loss or args.use_joints_rest_prior_loss:
                rest_joints = apply_transform(joints[..., :3], gt.pose_p2r.nan_to_num(nan=0.0))
                if args.use_joints_rest_loss:
                    # https://github.com/pytorch/pytorch/issues/15506
                    # Torch will produce NaN gradients if any element of the involved tensor is NaN,
                    # even if those NaNs are not accessed (e.g. being masked) in loss computation.
                    # So we have to replace NaNs with zeros.
                    loss_joints_rest = criterion(rest_joints[gt.joints_mask], gt.rest_joints[gt.joints_mask])
                    loss += 1e-1 * loss_joints_rest
                    loss_value_dict["loss/joints_rest"] = loss_joints_rest.item()
                if args.use_joints_rest_prior_loss:
                    loss_rest_prior = rest_prior_loss_fn(rest_joints)
                    loss += 1e-2 * loss_rest_prior
                    loss_value_dict["loss/joints_rest_prior"] = loss_rest_prior.item()
        else:
            assert args.joints_attn_causal_discrete
            from model import tokenize

            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = joints.shape[-1]
            joints_logits = joints[gt.joints_mask]
            labels = tokenize(joints_gt[gt.joints_mask], num_discrete=vocab_size)
            loss_joints = loss_fct(joints_logits.view(-1, vocab_size), labels.view(-1))
            loss += 1e-1 * loss_joints
            loss_value_dict["loss/joints"] = loss_joints.item()

    if args.predict_global_trans:
        loss_global = criterion(global_trans, gt.global_inv)
        loss += 1e-2 * loss_global
        loss_value_dict["loss/global"] = loss_global.item()

    if args.predict_pose_trans:
        if pose_trans.ndim == 3:
            if "local" in args.pose_mode:
                if args.pose_mode == "local_quat":
                    pose_gt = gt.pose_p2r_local_quat
                elif args.pose_mode == "local_ortho6d":
                    pose_gt = gt.pose_p2r_local_matrix
                    pose_trans = to_pose_local(pose_trans, input_mode=args.pose_mode, return_quat=False)
            elif args.pose_mode == "quat":
                pose_gt = gt.pose_p2r_quat
            elif args.pose_mode == "ortho6d":
                pose_gt = gt.pose_p2r_rot
                pose_trans = ortho6d_to_matrix(pose_trans)
            elif args.pose_mode == "transl_quat":
                pose_gt = gt.pose_p2r_transl_quat
            elif args.pose_mode == "dual_quat":
                pose_gt = gt.pose_p2r_dualquat
            elif args.pose_mode == "transl_ortho6d":
                pose_gt = gt.pose_p2r_transl_matrix
                transl, rotation = torch.split(pose_trans, [3, 6], dim=-1)
                rotation = ortho6d_to_matrix(rotation)
                rotation = rotation.reshape(*rotation.shape[:-2], 3 * 3)
                pose_trans = torch.cat([transl, rotation], dim=-1)
            elif args.pose_mode == "target_quat":
                pose_gt = gt.pose_p2r_target_quat
            elif args.pose_mode == "target_ortho6d":
                pose_gt = gt.pose_p2r_target_matrix
                target, rotation = torch.split(pose_trans, [3, 6], dim=-1)
                rotation = ortho6d_to_matrix(rotation)
                rotation = rotation.reshape(*rotation.shape[:-2], 3 * 3)
                pose_trans = torch.cat([target, rotation], dim=-1)
            loss_pose = criterion(pose_trans[gt.joints_mask], pose_gt[gt.joints_mask])
            loss += (1.0 if "local" in args.pose_mode or args.pose_mode in ("quat", "ortho6d") else 1e-1) * loss_pose
            loss_value_dict["loss/pose"] = loss_pose.item()
            if any(
                (
                    args.use_pose_rest_loss,
                    args.use_pose_rest_prior_loss,
                    args.use_pose_connect_loss,
                    args.use_pose_adv_loss,
                )
            ):
                if "local" in args.pose_mode:
                    if args.pose_mode == "local_quat":
                        pose_trans_ = quat_to_matrix(pose_trans)
                    else:
                        pose_trans_ = pose_trans
                    root_trans_inv = torch.einsum(
                        "bij,bjk->bik", gt.global_transform_matrix, gt.global_transform_rest_inv_matrix
                    )
                    root_pose = torch.einsum("bij,bnjk->bnik", root_trans_inv[..., :3, :3], pose_trans_[:, :1])
                    pose_trans_matrix, _ = pose_local_to_global(
                        torch.cat([root_pose, pose_trans_[:, 1:]], dim=1),
                        gt.joints.nan_to_num(nan=0.0),
                        torch.tensor(KINEMATIC_TREE.parent_indices),
                        gt.rest_joints_.nan_to_num(nan=0.0)[:, 0] - gt.joints.nan_to_num(nan=0.0)[:, 0],
                        relative_to_source=True,
                    )
                    root_trans = torch.einsum(
                        "bij,bjk->bik", gt.global_transform_rest_matrix, gt.global_transform_inv_matrix
                    )
                    pose_trans_matrix = torch.einsum("bij,bnjk->bnik", root_trans, pose_trans_matrix)
                    rest_joints = apply_transform(gt.joints.nan_to_num(nan=0.0), pose_trans_matrix)
                elif args.pose_mode in ("quat", "ortho6d"):
                    pose_trans_matrix, rest_joints = pose_rot_to_global(
                        pose_trans,
                        gt.joints.nan_to_num(nan=0.0),
                        torch.tensor(KINEMATIC_TREE.parent_indices),
                        gt.rest_joints.nan_to_num(nan=0.0)[:, 0] - gt.joints.nan_to_num(nan=0.0)[:, 0],
                    )
                else:
                    pose_trans_matrix = to_pose_matrix(
                        pose_trans,
                        input_mode=args.pose_mode.replace("ortho6d", "matrix"),
                        source=gt.joints.nan_to_num(nan=0.0),
                    )
                    rest_joints = apply_transform(gt.joints.nan_to_num(nan=0.0), pose_trans_matrix)
                rest_joints_tail = apply_transform(gt.joints_tail.nan_to_num(nan=0.0), pose_trans_matrix)
                if args.use_pose_rest_loss:
                    loss_pose_rest = criterion(rest_joints[gt.joints_mask], gt.rest_joints[gt.joints_mask])
                    loss += 1e-1 * loss_pose_rest
                    loss_value_dict["loss/pose_rest"] = loss_pose_rest.item()
                # hips_transform = Transform3d(
                #     matrix=PoseData(
                #         joints=rest_joints.detach(), joints_tail=rest_joints_tail.detach()
                #     ).hips_transform.transpose(-1, -2)
                # )
                # rest_joints = hips_transform.transform_points(rest_joints)
                # rest_joints_tail = hips_transform.transform_points(rest_joints_tail)
                vis_data["pose_rest_joints"] = torch.cat([rest_joints, rest_joints_tail], dim=1)
                if args.use_pose_rest_prior_loss:
                    loss_rest_prior = rest_prior_loss_fn(rest_joints, rest_joints_tail)
                    loss += 1e-2 * loss_rest_prior
                    loss_value_dict["loss/pose_rest_prior"] = loss_rest_prior.item()
                if args.use_pose_connect_loss:
                    loss_pose_connect = connect_loss_fn(rest_joints, rest_joints_tail)
                    loss += 1e-2 * loss_pose_connect
                    loss_value_dict["loss/pose_connect"] = loss_pose_connect.item()
                if args.use_pose_adv_loss:
                    from model import adv_loss_g
                    from util.dataset_mixamo import keep_exists

                    model_D = get_discriminator(gt.device, args.distributed, (args.gpu,))[0]
                    model_D.eval()
                    rest_joints_real = torch.cat((gt.rest_joints, gt.rest_joints_tail), dim=-1)
                    # rest_joints_real = keep_exists(rest_joints_real)
                    rest_joints_real = rest_joints_real.nan_to_num(nan=0.0)
                    rest_joints_fake = torch.cat((rest_joints, rest_joints_tail), dim=-1)
                    # rest_joints_fake = keep_exists(rest_joints_fake)
                    rest_joints_fake = rest_joints_fake.nan_to_num(nan=0.0)
                    gt.d_real, gt.d_fake = rest_joints_real.detach(), rest_joints_fake.detach()
                    loss_G = adv_loss_g(model_D(rest_joints_fake, mask=gt.joints_mask))
                    loss += 1e-4 * loss_G
                    loss_value_dict["loss/pose_adv_g"] = loss_G.item()
        else:
            assert args.pose_attn_causal_discrete
            from model import tokenize

            if args.pose_mode == "ortho6d":
                pose_gt = gt.pose_p2r_ortho6d
            else:
                raise NotImplementedError(f"{args.pose_mode=}")
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = pose_trans.shape[-1]
            pose_logits = pose_trans[gt.joints_mask]
            labels = tokenize(pose_gt[gt.joints_mask], num_discrete=vocab_size)
            loss_pose = loss_fct(pose_logits.view(-1, vocab_size), labels.view(-1))
            loss += 1e-1 * loss_pose
            loss_value_dict["loss/pose"] = loss_pose.item()

    assert isinstance(loss, torch.Tensor), "No loss"
    loss_value = loss.item()
    loss_value_dict["loss"] = loss_value

    if not math.isfinite(loss_value):
        raise RuntimeError(f"Loss is NaN: {loss_value_dict}")

    return loss, loss_value_dict, vis_data


@robust_dataloader
def train_one_epoch(
    model: PCAE,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    max_norm: float = 0,
    log_writer: SummaryWriter = None,
    args=None,
    report_every: int = None,
    data_loader_val: DataLoader = None,
):
    accum_iter = args.accum_iter
    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    optimizer.zero_grad()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" | ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.2e}"))
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, 50, f"Epoch: [{epoch}]")):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        data: PoseData
        # Inputs
        if args.aug_rotation:
            assert not (args.predict_joints and args.use_joints_rest_prior_loss)
            assert not (args.predict_pose_trans and args.use_pose_rest_prior_loss)
            rotate = Rotate(R=random_rotations(len(data)))
        else:
            rotate = Transform3d(matrix=data.hips_transform.transpose(-1, -2))
        pts = rotate.transform_points(data.pts)
        norm = get_normalize_transform(pts, keep_ratio=True, recenter=args.aug_rotation)
        global_transform = rotate.compose(norm)
        if args.aug_rotation:
            global_transform_rest = global_transform
        else:
            global_transform_rest = Transform3d(matrix=data.hips_transform_rest.transpose(-1, -2)).compose(norm)
        pts = norm.transform_points(pts)
        if args.input_normal:
            pts_normal = F.normalize(global_transform.transform_normals(data.pts_normal), dim=-1)
            if args.drop_normal_ratio > 0:
                drop_normal = torch.rand(data.pts_normal.shape[0], device=pts_normal.device) < args.drop_normal_ratio
                pts_normal[drop_normal] = torch.zeros_like(pts_normal[drop_normal])
            pts = torch.cat([pts, pts_normal], dim=-1)
        pts = pts.to(device, non_blocking=True)
        if args.predict_bw:
            verts = global_transform.transform_points(data.verts)
            if args.input_normal:
                verts_normal = F.normalize(global_transform.transform_normals(data.verts_normal), dim=-1)
                if args.drop_normal_ratio > 0:
                    verts_normal[drop_normal] = torch.zeros_like(verts_normal[drop_normal])
                verts = torch.cat([verts, verts_normal], dim=-1)
            verts = verts.to(device, non_blocking=True)
        else:
            verts = None

        # Ground truth
        gt = GT(data, global_transform, global_transform_rest, device)  # cache some values to avoid recomputing
        # import trimesh; trimesh.Scene([trimesh.PointCloud(verts[-1].cpu().numpy()), trimesh.PointCloud(gt.joints[-1].nan_to_num().cpu().numpy()), trimesh.PointCloud(gt.joints_tail[-1].nan_to_num().cpu().numpy())]).export("test.glb")

        # Forward
        with torch.amp.autocast("cuda", enabled=args.ae_type == "hy3d2.1"):
            model.train()
            joints_gt = pose_gt = None
            if (
                (args.predict_joints and args.joints_attn_causal)
                or (args.predict_bw and args.bw_input_joints)
                or (args.predict_pose_trans and args.pose_input_joints)
            ):
                if args.predict_joints_tail or args.bw_input_joints or args.pose_input_joints:
                    joints_gt = torch.cat((gt.joints, gt.joints_tail), dim=-1).nan_to_num(nan=0.0)
                else:
                    joints_gt = gt.joints.nan_to_num(nan=0.0)
            if args.predict_pose_trans and args.pose_attn_causal:
                if args.pose_mode == "local_quat":
                    pose_gt = gt.pose_p2r_local_quat
                elif args.pose_mode == "local_ortho6d":
                    pose_gt = gt.pose_p2r_local_ortho6d
                elif args.pose_mode == "quat":
                    pose_gt = gt.pose_p2r_quat
                elif args.pose_mode == "ortho6d":
                    pose_gt = gt.pose_p2r_ortho6d
                elif args.pose_mode == "transl_quat":
                    pose_gt = gt.pose_p2r_transl_quat
                elif args.pose_mode == "dual_quat":
                    pose_gt = gt.pose_p2r_dualquat
                elif args.pose_mode == "transl_ortho6d":
                    pose_gt = gt.pose_p2r_transl_ortho6d
                elif args.pose_mode == "target_quat":
                    pose_gt = gt.pose_p2r_target_quat
                elif args.pose_mode == "target_ortho6d":
                    pose_gt = gt.pose_p2r_target_ortho6d
                pose_gt = pose_gt.nan_to_num(nan=0.0)
            output = model(pts, verts, joints=joints_gt, pose=pose_gt)
            loss, loss_value_dict, _ = get_loss(output, gt, criterion, args)

        for k, v in loss_value_dict.items():
            loss_value_dict[k] = misc.all_reduce_mean(v)

        # optimizer.zero_grad()
        # loss.backward()
        # for name, param in model.named_parameters():
        #     if param.requires_grad and not torch.isfinite(param).all():
        #         print(f"Parameter {name} is not finite, fixing it")
        #         param.requires_grad = False
        #         param.nan_to_num_(nan=0.0)
        #         param.requires_grad = True
        # optimizer.step()
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            named_parameters=model.named_parameters(),
        )
        # # print unused params
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(name)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if discriminator:
            model_D, optimizer_D = get_discriminator()  # here model_D still has grad (from g_loss)
            optimizer_D.zero_grad()
            loss_D = get_loss_discriminator(model_D, gt)
            loss_D_value = loss_D.item()
            loss_value_dict["loss/pose_adv_d"] = misc.all_reduce_mean(loss_D_value)
            if (epoch == 0 and data_iter_step < 20) or (
                data_iter_step % 50 == 0 and loss_value_dict["loss/pose_adv_g"] < math.log(2)
            ):
                loss_D.backward()
                optimizer_D.step()
                optimizer_D.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value_dict["loss"])

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if (data_iter_step + 1) % accum_iter == 0:
            # # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
            # step = int((data_iter_step / len(data_loader) + epoch) * 1000)
            step = data_iter_step + epoch * len(data_loader)

            if log_writer is not None and misc.is_main_process():
                for k, v in loss_value_dict.items():
                    log_writer.add_scalar(k, v, step)
                log_writer.add_scalar("lr", max_lr, step)

            if report_every is not None:
                if report_every < 1:
                    report_every = int(len(data_loader) * report_every)
                else:
                    report_every = int(report_every)
                report_every = max(1, report_every)
                if data_iter_step % report_every == 0:
                    # if args.output_dir and data_iter_step != 0:
                    #     misc.save_model(
                    #         args=args,
                    #         model=model,
                    #         model_without_ddp=(
                    #             model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    #         ),
                    #         optimizer=optimizer,
                    #         loss_scaler=loss_scaler,
                    #         epoch=f"step{step:06}",
                    #     )
                    test_stats, vis_data = evaluate(data_loader_val, model, device, args)
                    log_stats = {
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        "step": step,
                    }
                    if args.log_dir and misc.is_main_process():
                        if log_writer is not None:
                            for k, v in log_stats.items():
                                if k in ("step",):
                                    continue
                                log_writer.add_scalar(f"eval/{k.removeprefix('test_')}", v, step)
                            if vis_data:
                                log_writer.add_mesh(
                                    "eval/vis/pose_rest_joints",
                                    vertices=vis_data["pose_rest_joints"][:5].cpu(),
                                    global_step=step,
                                )
                        with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
@robust_dataloader
def evaluate(data_loader: DataLoader, model: PCAE, device: torch.device, args):
    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter=" | ")
    model.eval()
    vis_data_list = []
    for data in metric_logger.log_every(data_loader, 50, "Test:"):
        data: PoseData
        rotate = Transform3d(matrix=data.hips_transform.transpose(-1, -2))
        pts = rotate.transform_points(data.pts)
        norm = get_normalize_transform(pts, keep_ratio=True, recenter=False)
        global_transform = rotate.compose(norm)
        global_transform_rest = Transform3d(matrix=data.hips_transform_rest.transpose(-1, -2)).compose(norm)
        pts = global_transform.transform_points(data.pts)
        if args.input_normal:
            pts_normal = F.normalize(global_transform.transform_normals(data.pts_normal), dim=-1)
            pts = torch.cat([pts, pts_normal], dim=-1)
        pts = pts.to(device, non_blocking=True)
        if args.predict_bw:
            verts = global_transform.transform_points(data.verts)
            if args.input_normal:
                verts_normal = F.normalize(global_transform.transform_normals(data.verts_normal), dim=-1)
                verts = torch.cat([verts, verts_normal], dim=-1)
            verts = verts.to(device, non_blocking=True)
        else:
            verts = None
        gt = GT(data, global_transform, global_transform_rest, device)
        with torch.amp.autocast("cuda", enabled=args.ae_type == "hy3d2.1"):
            output = model(
                pts,
                verts,
                joints=(
                    torch.cat((gt.joints.nan_to_num(nan=0.0), gt.joints_tail.nan_to_num(nan=0.0)), dim=-1)
                    if args.bw_input_joints or args.pose_input_joints
                    else None
                ),
            )
            _, loss_value_dict, vis_data = get_loss(output, gt, criterion, args)
        metric_logger.update(**loss_value_dict)
        vis_data_list.append(vis_data)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"* loss {metric_logger.loss.global_avg:.3f}")
    vis_data = {k: torch.cat([d[k] for d in vis_data_list], dim=0) for k in vis_data_list[0].keys()}
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, vis_data
