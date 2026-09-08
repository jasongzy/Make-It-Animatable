import warnings
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch_cluster import fps

from models_ae import Attention, DiagonalGaussianDistribution, FeedForward, PreNorm, create_autoencoder
from util.dataset_mixamo import Joint
from util.utils import find_ckpt

Output = NamedTuple(
    "Output",
    [("bw", torch.Tensor), ("joints", torch.Tensor), ("global_trans", torch.Tensor), ("pose_trans", torch.Tensor)],
)


class Embedder3D(nn.Module):
    def __init__(self, dim=48, concat_input=True):
        super().__init__()

        assert dim % 6 == 0
        self.embedding_dim = dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
                torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
                torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
            ]
        )
        self.register_buffer("basis", e, persistent=False)  # 3 x 16
        self.basis: torch.Tensor
        self.concat_input = concat_input

    def embed(self, xyz: torch.Tensor):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, `dim]
        """
        projections = torch.einsum("bnd,de->bne", xyz, self.basis)
        return torch.cat([projections.sin(), projections.cos()], dim=2)

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, `dim`(+3)]
        """
        embeddings = self.embed(xyz)
        if self.concat_input:
            embeddings = torch.cat([embeddings, xyz], dim=-1)
        return embeddings


class JointsEmbedder(nn.Module):
    def __init__(self, include_tail=False, embed_dim=48, out_dim=512, concat_input=True, out_mlp=True):
        super().__init__()
        self.embed = Embedder3D(embed_dim, concat_input=concat_input)
        self.point_num = 2 if include_tail else 1
        self.embedding_dim = self.point_num * (embed_dim + (3 if concat_input else 0))
        self.out_mlp = out_mlp
        if self.out_mlp:
            self.mlp = nn.Sequential(nn.Linear(self.embedding_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, joints: torch.Tensor):
        """
        Args:
            joints: [B, N, D]
        Returns:
            [B, N, `out_dim` if `out_mlp` else `embedding_dim`]
        """
        B, N, D = joints.shape
        assert D == self.point_num * 3
        out = self.embed(joints.view(B, N * self.point_num, 3)).view(B, N, self.embedding_dim)
        return self.mlp(out) if self.out_mlp else out


class TransformMLP(nn.Module):
    def __init__(self, in_dim: int, transl_dim=3, rotation_dim=4, scaling_dim=3):
        super().__init__()
        self.transl_dim = transl_dim
        self.rotation_dim = rotation_dim
        self.scaling_dim = scaling_dim
        if self.transl_dim > 0:
            self.transl_mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, transl_dim))
        if self.rotation_dim > 0:
            if self.rotation_dim == 4:  # quaternions
                self.rotation_mlp_scalar = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, 1))
                self.rotation_mlp_vector = nn.Sequential(
                    nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, rotation_dim - 1)
                )
            else:
                self.rotation_mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, rotation_dim))
        if self.scaling_dim > 0:
            self.scaling_mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, scaling_dim))

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: [B, N, `in_dim`]
        Returns:
            [B, N, `transl_dim` + `rotation_dim` + `scaling_dim`]
        """
        empty = torch.empty(*feat.shape[:2], 0, device=feat.device)
        transl = self.transl_mlp(feat) if self.transl_dim > 0 else empty
        if self.rotation_dim > 0:
            if self.rotation_dim == 4:
                # # Make sure the scalar part is positive
                # scalar_sign = rotation[..., :1].sign().detach()
                # scalar_sign[scalar_sign == 0] = 1
                # rotation = rotation * scalar_sign
                rotation_scalar = torch.exp(self.rotation_mlp_scalar(feat))
                rotation_vector = self.rotation_mlp_vector(feat)
                rotation = F.normalize(torch.cat([rotation_scalar, rotation_vector], dim=-1), dim=-1)
            else:
                rotation = self.rotation_mlp(feat)
                if self.rotation_dim == 3:
                    rotation = F.normalize(rotation, dim=-1)
        else:
            rotation = empty
        scaling = torch.exp(self.scaling_mlp(feat)) if self.scaling_dim > 0 else empty
        return torch.cat([transl, rotation, scaling], dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, feat_dim: int, dim_head: int, heads: int):
        super().__init__()
        self.cross_attn = PreNorm(
            feat_dim,
            Attention(feat_dim, feat_dim, heads=heads, dim_head=dim_head),
            context_dim=feat_dim,
        )
        self.ff = PreNorm(feat_dim, FeedForward(feat_dim))

    def forward(self, query: torch.Tensor | list[torch.Tensor], context: torch.Tensor = None):
        """
        Args:
            query: [B, N1, D]
            context: [B, N2, D]
        Returns:
            [B, N1, D]
        """
        if isinstance(query, (list, tuple)):
            flag_cat = True
            query_shapes = [x.shape[1] for x in query]
            query = torch.cat(query, dim=1)
        else:
            flag_cat = False
        logits = self.cross_attn(query, context=context)
        logits = logits + self.ff(logits)
        return torch.split(logits, query_shapes, dim=1) if flag_cat else logits


class JointsAttention(nn.Module):
    def __init__(self, feat_dim: int, heads=8, dim_head=64, masked=True, kinematic_tree: Joint = None, *args, **xargs):
        super().__init__()

        self.norm_pre = nn.LayerNorm(feat_dim)
        self.attn = Attention(query_dim=feat_dim, heads=heads, dim_head=dim_head, *args, **xargs)
        self.norm_after = nn.LayerNorm(feat_dim)

        if masked:
            assert kinematic_tree is not None
            mask = torch.zeros((len(kinematic_tree), len(kinematic_tree)), dtype=torch.bool)
            for joint in kinematic_tree:
                mask[joint.index, joint.index] = True
                for parent in joint.parent_recursive:
                    mask[joint.index, parent.index] = True
            self.register_buffer("mask", mask, persistent=False)
            self.mask: torch.Tensor
        else:
            self.mask = None

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: [B, N, D]
        Returns:
            [B, N, D]
        """
        out = self.attn(
            self.norm_pre(feat), mask=None if self.mask is None else self.mask.expand(feat.shape[0], -1, -1)
        )
        out = out + feat
        out = self.norm_after(out)
        return out


class InputAttention(nn.Module):
    def __init__(self, feat_dim: int, heads=8, dim_head=64, *args, **xargs):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.norm_context = nn.LayerNorm(feat_dim)
        self.attn = Attention(query_dim=feat_dim, heads=heads, dim_head=dim_head, *args, **xargs)
        nn.init.zeros_(self.attn.to_out.weight)
        nn.init.zeros_(self.attn.to_out.bias)

    def forward(self, feat: torch.Tensor, context: torch.Tensor, return_score=False):
        """
        Args:
            feat: [..., D]
            context: [..., k, D]
        Returns:
            [..., D]
        """
        sh = feat.shape
        feat = feat.view(-1, 1, sh[-1])
        context = context.view(-1, context.shape[-2], context.shape[-1])
        context = torch.cat([feat, context], dim=-2)
        out = self.attn(self.norm(feat), self.norm_context(context), return_score=return_score)
        if return_score:
            out, attn_score = out
        out = out + feat
        out = out.reshape(*sh)
        if return_score:
            attn_score = attn_score.reshape(*sh[:-1], -1, context.shape[-2])
            return out, attn_score
        return out


class PCAE(nn.Module):
    def __init__(
        self,
        ae_type="vecset",
        N=2048,
        input_normal=False,
        input_attention=False,
        num_latents=512,
        deterministic=True,
        hierarchical_ratio=0.0,
        output_dim=52,
        output_actvn="softmax",
        output_log=False,
        kinematic_tree: Joint = None,
        tune_decoder_self_attn=True,
        tune_decoder_cross_attn=True,
        sharing_embed=tuple(),
        predict_bw=True,
        bw_dot=False,
        bw_input_joints=False,
        predict_joints=False,
        predict_joints_tail=False,
        joints_attn=False,
        joints_attn_masked=True,
        joints_attn_causal=False,
        joints_attn_causal_discrete=False,
        predict_global_trans=False,
        predict_pose_trans=False,
        pose_mode="dual_quat",
        pose_input_joints=False,
        pose_attn=False,
        pose_attn_masked=True,
        pose_attn_causal=False,
        pose_attn_causal_discrete=False,
        grid_density=128,
    ):
        super().__init__()

        self.ae_type = ae_type
        self.N = N
        self.hierarchical_ratio = float(hierarchical_ratio)
        assert 0 <= hierarchical_ratio < 1.0, f"{hierarchical_ratio=} must be in [0, 1)"

        if self.ae_type == "vecset":
            self.base = create_autoencoder(dim=512, M=num_latents, N=self.N, latent_dim=8, deterministic=deterministic)
            self.embed_dim = self.base.point_embed.mlp.out_features
            self.feat_dim = self.base.decoder_cross_attn.fn.to_out.out_features

            self.input_dims = [3]
            self.input_normal = input_normal
            if self.input_normal:
                self.input_dims.append(3)
                self.normal_embed = JointsEmbedder(out_dim=self.embed_dim)
                nn.init.zeros_(self.normal_embed.mlp[-1].weight)
                nn.init.zeros_(self.normal_embed.mlp[-1].bias)
            else:
                self.input_dims.append(0)
            self.input_dim = sum(self.input_dims)
            if self.input_dim == self.input_dims[0]:
                self.input_attention = False
            else:
                self.input_attention = input_attention
            if self.input_attention:
                self.input_attn = InputAttention(self.embed_dim)

        elif self.ae_type == "hy3d2.1":
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
            import logging

            from util.Hunyuan3D_21.hy3dshape.hy3dshape.models.autoencoders import ShapeVAE

            logging.getLogger("hy3dgen.shapgen").setLevel(logging.WARNING)

            pc_sharpedge_size = int(self.N * self.hierarchical_ratio)
            pc_size = self.N - pc_sharpedge_size
            self.base = ShapeVAE.from_pretrained(
                "tencent/Hunyuan3D-2.1",
                device="cpu",
                # dtype=torch.float32,
                use_safetensors=False,
                variant="fp16",
                pc_size=pc_size,
                pc_sharpedge_size=pc_sharpedge_size,
                downsample_ratio=self.N // 4096,
            )
            self.embed_dim = 1024
            self.feat_dim = 1024
            self.num_latents = 4096
            self.input_normal = input_normal
            assert self.input_normal
            self.input_dims = [3, 3]
            self.input_dim = 6

            old_query_proj = self.base.geo_decoder.query_proj
            new_query_proj = nn.Linear(self.base.geo_decoder.fourier_embedder.out_dim + 3, old_query_proj.out_features)
            with torch.no_grad():
                new_query_proj.weight[:, :-3] = old_query_proj.weight
                nn.init.zeros_(new_query_proj.weight[:, -3:])
                new_query_proj.bias = old_query_proj.bias
            self.base.geo_decoder.query_proj = new_query_proj

            self.base.geo_decoder.output_proj = nn.Linear(
                self.base.geo_decoder.output_proj.in_features, self.base.geo_decoder.output_proj.in_features
            )

        else:
            raise NotImplementedError(f"{self.ae_type=}")

        self.output_dim = output_dim
        assert len(sharing_embed) != 1, "len(sharing_embed) must be 0 or > 1"
        self.sharing_embed = sharing_embed
        self.shared_embed = None

        self.predict_bw = predict_bw
        if self.predict_bw:
            assert not bw_input_joints or (bw_input_joints and bw_dot), "bw_input_joints requires bw_dot"
            self.bw_dot = bw_dot
            if self.bw_dot:
                self.bw_embed = self.get_embed("bw")
                self.bw_cross_attn = CrossAttention(self.feat_dim, dim_head=self.feat_dim, heads=1)
                self.bw_proj_queries = nn.Sequential(
                    nn.LayerNorm(self.feat_dim), nn.Linear(self.feat_dim, self.feat_dim)
                )
                self.bw_proj_bones = nn.Sequential(nn.LayerNorm(self.feat_dim), nn.Linear(self.feat_dim, self.feat_dim))
            else:
                self.bw_head = nn.Linear(self.feat_dim, self.output_dim)
            self.bw_input_joints = bw_input_joints
            if self.bw_input_joints:
                self.joints_embedder = JointsEmbedder(include_tail=True, out_dim=self.embed_dim)
            output_actvn = output_actvn.lower()
            if output_actvn == "softmax":
                # self.actvn = nn.LogSoftmax(dim=-1) if self.output_actvn_log else nn.Sigmoid(dim=-1)
                self.actvn = nn.Softmax(dim=-1)
            elif output_actvn == "sigmoid":
                # self.actvn = nn.LogSigmoid() if self.output_actvn_log else nn.Sigmoid()
                self.actvn = nn.Sigmoid()
            elif output_actvn == "relu":
                self.actvn = nn.ReLU()
            elif output_actvn == "softplus":
                self.actvn = nn.Softplus()
            else:
                raise ValueError(f"Invalid activation: {output_actvn}")
            self.output_actvn_log = output_log

        self.tune_decoder_self_attn = tune_decoder_self_attn
        self.tune_decoder_cross_attn = tune_decoder_cross_attn

        self.predict_joints = predict_joints
        self.predict_joints_tail = predict_joints_tail
        if self.predict_joints:
            joints_dim = 6 if self.predict_joints_tail else 3
            assert not (joints_attn and joints_attn_causal), "Conflict arguments: joints_attn & joints_attn_causal"
            assert not (
                not joints_attn_causal and joints_attn_causal_discrete
            ), "Conflict arguments: !joints_attn_causal & joints_attn_causal_discrete"
            self.joints_attn_causal = joints_attn_causal
            self.joints_attn_causal_discrete = joints_attn_causal_discrete
            if self.joints_attn_causal and self.joints_attn_causal_discrete:
                self.joints_embed = nn.Parameter(torch.randn(1, self.output_dim * joints_dim, self.embed_dim))
            else:
                self.joints_embed = self.get_embed("joints")
            if joints_attn:
                self.joints_head = nn.Sequential(
                    JointsAttention(self.feat_dim, masked=joints_attn_masked, kinematic_tree=kinematic_tree),
                    nn.Linear(self.feat_dim, joints_dim),
                )
            elif self.joints_attn_causal:
                if self.joints_attn_causal_discrete:
                    assert self.predict_joints_tail
                    self.joints_head = JointsAttentionCausalDiscrete(
                        self.feat_dim,
                        kinematic_tree=kinematic_tree,
                        out_type="joints",
                        out_dim=joints_dim,
                        query_type="embedding",
                    )
                else:
                    self.joints_head = JointsAttentionCausal(
                        self.feat_dim,
                        kinematic_tree=kinematic_tree,
                        out_type="joints",
                        include_joints_tail=self.predict_joints_tail,
                        out_dim=joints_dim,
                        query_type="embedding",
                    )
            else:
                self.joints_head = nn.Linear(self.feat_dim, joints_dim)

        self.predict_global_trans = predict_global_trans
        if self.predict_global_trans:
            self.global_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.global_head = TransformMLP(self.feat_dim, transl_dim=3, rotation_dim=4, scaling_dim=1)

        self.predict_pose_trans = predict_pose_trans
        assert pose_mode in (
            "transl_quat",  # 3 + 4
            "dual_quat",  # 4 + 4
            "transl_ortho6d",  # 3 + 6
            "target_quat",  # 3 + 4
            "target_ortho6d",  # 3 + 6
            "quat",  # 4
            "ortho6d",  # 6
            "local_quat",  # 4
            "local_ortho6d",  # 6
        ), f"Invalid {pose_mode=}"
        self.pose_mode = pose_mode
        self.pose_input_joints = pose_input_joints
        if self.predict_pose_trans:
            if "local" in self.pose_mode or self.pose_mode in ("quat", "ortho6d"):
                rotation_dim = pose_dim = 6 if "ortho6d" in self.pose_mode else 4
                self.pose_head = TransformMLP(self.feat_dim, transl_dim=0, rotation_dim=rotation_dim, scaling_dim=0)
            else:
                transl_dim = 4 if self.pose_mode == "dual_quat" else 3
                rotation_dim = 6 if "ortho6d" in self.pose_mode else 4
                pose_dim = transl_dim + rotation_dim
                self.pose_head = TransformMLP(
                    self.feat_dim, transl_dim=transl_dim, rotation_dim=rotation_dim, scaling_dim=0
                )
            if self.pose_input_joints:
                self.joints_embedder = JointsEmbedder(include_tail=True, out_dim=self.embed_dim)
            assert not (pose_attn and pose_attn_causal), "Conflict arguments: pose_attn & pose_attn_causal"
            assert not (
                not pose_attn_causal and pose_attn_causal_discrete
            ), "Conflict arguments: !pose_attn_causal & pose_attn_causal_discrete"
            assert not (
                self.pose_input_joints and pose_attn_causal_discrete
            ), "Conflict arguments: pose_input_joints & pose_attn_causal_discrete"
            self.pose_attn_causal = pose_attn_causal
            self.pose_attn_causal_discrete = pose_attn_causal_discrete
            if self.pose_attn_causal and self.pose_attn_causal_discrete:
                self.pose_embed = nn.Parameter(torch.randn(1, self.output_dim * pose_dim, self.embed_dim))
            else:
                self.pose_embed = self.get_embed("pose")
            if pose_attn:
                self.pose_head = nn.Sequential(
                    JointsAttention(self.feat_dim, masked=pose_attn_masked, kinematic_tree=kinematic_tree),
                    self.pose_head,
                )
            elif self.pose_attn_causal:
                if self.pose_attn_causal_discrete:
                    self.pose_head = JointsAttentionCausalDiscrete(
                        self.feat_dim,
                        kinematic_tree=kinematic_tree,
                        out_type="pose",
                        out_dim=pose_dim,
                        query_type="embedding",
                    )
                else:
                    self.pose_head = JointsAttentionCausal(
                        self.feat_dim,
                        kinematic_tree=kinematic_tree,
                        out_type="pose",
                        out_dim=pose_dim,
                        rotation_dim=rotation_dim,
                        query_type="embedding",
                    )

        self.grid_density = grid_density
        self.grid = None

    def get_embed(self, name: str):
        if name in self.sharing_embed:
            if self.shared_embed is None:
                self.shared_embed = nn.Parameter(torch.randn(1, self.output_dim, self.embed_dim))
            return self.shared_embed
        else:
            return nn.Parameter(torch.randn(1, self.output_dim, self.embed_dim))

    def adapt_ckpt(self, ckpt: dict[str, torch.Tensor]):
        def access_attr(obj, attr: str):
            for k in attr.split("."):
                obj = getattr(obj, k)
            return obj

        params2replace = []
        if self.predict_bw:
            params2replace.extend(["bw_head.weight", "bw_head.bias"])
        for k in params2replace:
            if k in ckpt:
                ckpt_param = ckpt[k]
                model_param = access_attr(self, k)
                if ckpt_param.shape != model_param.shape:
                    print(
                        f"Size mismatch for {k}: {ckpt_param.shape} from checkpoint vs {model_param.shape} from model. Ignoring it."
                    )
                    ckpt[k] = model_param.to(ckpt_param)

        params2remove = [
            "normal_embed.embed.basis",
            "joints_embedder.embed.basis",
            "joints_head.encoder.embed.basis",
            "joints_head.0.mask",
            "pose_head.0.mask",
        ]
        params2remove.extend(
            [
                f"{l}.{m}"
                for l in ("joints_head", "pose_head")
                for m in ("mask_attn", "mask_parent", "tree_levels_mask", "embed.basis")
            ]
        )
        for k in params2remove:
            if k in ckpt:
                print(f"Removing deprecated params {k} from checkpoint.")
                del ckpt[k]

        params2partial = []
        if self.predict_joints:
            params2partial.append("joints_embed")
        if self.predict_pose_trans:
            params2partial.append("pose_embed")
        for k in params2partial:
            if k in ckpt:
                ckpt_param = ckpt[k]
                model_param = access_attr(self, k)
                if ckpt_param.shape != model_param.shape:
                    assert ckpt_param.shape[0] == model_param.shape[0] and ckpt_param.shape[-1] == model_param.shape[-1]
                    print(
                        f"Size mismatch for {k}: {ckpt_param.shape} from checkpoint vs {model_param.shape} from model. Partially loading it."
                    )
                    if ckpt_param.shape[1] < model_param.shape[1]:
                        ckpt_param_new = model_param.clone().detach()
                        ckpt_param_new[:, : ckpt_param.shape[1]] = ckpt_param
                        ckpt[k] = ckpt_param_new.to(ckpt_param)
                    else:
                        ckpt[k] = ckpt_param[:, : model_param.shape[1]]

        return ckpt

    def load(self, pth_path: str, epoch=-1, strict=True, adapt=True):
        pth_path = find_ckpt(pth_path, epoch=epoch)
        checkpoint = torch.load(pth_path, map_location="cpu")
        model_state_dict = checkpoint["model"]
        if adapt:
            model_state_dict = self.adapt_ckpt(model_state_dict)
        self.load_state_dict(model_state_dict, strict=strict)
        print(f"Loaded model from {pth_path}")
        return self

    def load_base(self, pth_path: str):
        if self.ae_type == "vecset":
            self.base.load_state_dict(torch.load(pth_path, map_location="cpu")["model"], strict=True)
            print(f"Loaded base model from {pth_path}")
        return self

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        tune_module_list: list[nn.Module] = []
        if self.tune_decoder_self_attn:
            if self.ae_type == "vecset":
                tune_module_list.append(self.base.layers)
            elif self.ae_type == "hy3d2.1":
                tune_module_list.append(self.base.transformer)
        if self.tune_decoder_cross_attn:
            if self.ae_type == "vecset":
                tune_module_list.append(self.base.decoder_cross_attn)
                if self.base.decoder_ff is not None:
                    tune_module_list.append(self.base.decoder_ff)
            elif self.ae_type == "hy3d2.1":
                tune_module_list.append(self.base.geo_decoder)
        for module in tune_module_list:
            if self.ae_type == "hy3d2.1":
                module.to(dtype=torch.float32)
            for param in module.parameters():
                param.requires_grad = True
            if self.ae_type == "hy3d2.1" and not self.predict_bw:
                for param in self.base.geo_decoder.query_proj.parameters():
                    param.requires_grad = False
        return self

    def fps(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc: [B, `self.N`, D]
        Returns:
            [B, `self.base.num_latents`, D]
        """
        B, N, D = pc.shape
        assert N == self.base.num_inputs
        assert D == self.input_dim

        # flattened = pc.view(B * N, D)
        # batch = torch.arange(B).to(pc.device)
        # batch = torch.repeat_interleave(batch, N)
        # pos = flattened
        # ratio = 1.0 * self.base.num_latents / N
        # idx = fps(pos, batch, ratio=ratio)
        # sampled_pc = pos[idx]
        # sampled_pc = sampled_pc.view(B, -1, 3)

        N_hier = int(N * self.hierarchical_ratio)
        N_pc = [N - N_hier, N_hier]
        num_latents_hier = int(self.base.num_latents * self.hierarchical_ratio)
        N_latents = [self.base.num_latents - num_latents_hier, num_latents_hier]
        latents_begin_idx = [0, self.base.num_latents - num_latents_hier]

        sampled_pc = torch.empty(B, self.base.num_latents, D, dtype=pc.dtype, device=pc.device)
        for i, pc_ in enumerate(pc.split(N_pc, dim=1)):
            N_ = pc_.shape[1]
            if N_ == 0:
                continue
            batch = torch.repeat_interleave(torch.arange(B).to(pc.device), N_)
            pos = pc_.reshape(-1, D)
            idx = fps(pos[:, :3], batch, ratio=1.0 * N_latents[i] / N_)
            sampled_pc[:, latents_begin_idx[i] : latents_begin_idx[i] + N_latents[i], :] = pos[idx].view(B, -1, D)[
                :, : N_latents[i], :
            ]
        # import trimesh; trimesh.Trimesh(sampled_pc[0].cpu()).export("sample.ply")  # fmt: skip

        return sampled_pc

    def embed(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc: [B, N, D]
        Returns:
            [B, N, 512]
        """
        extra_embeddings = []
        if self.input_dim > 3:
            pc, normal = pc.split(self.input_dims, dim=-1)
            if normal.shape[-1] > 0:
                extra_embeddings.append(self.normal_embed(normal))

        pc_embeddings = self.base.point_embed(pc)

        if extra_embeddings:
            if self.input_attention:
                pc_embeddings = self.input_attn(pc_embeddings, torch.stack(extra_embeddings, dim=-2))
                # pc_embeddings, attn_score = self.input_attn(
                #     pc_embeddings, torch.stack(extra_embeddings, dim=-2), return_score=True
                # )
                # if pc.shape[1] > 512:
                #     import matplotlib
                #     import trimesh  # fmt: skip
                #     cmap = matplotlib.colormaps.get_cmap("plasma")
                #     vis_head = 0
                #     trimesh.PointCloud(pc[0].cpu().numpy(), colors=cmap(attn_score[0, :, vis_head, -1].cpu().numpy())[:, :3], process=False).export("normal_attn.glb")  # fmt: skip
            else:
                pc_embeddings = pc_embeddings + torch.stack(extra_embeddings, dim=0).sum(0)

        return pc_embeddings

    def encode(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc: [B, `self.N`, 3]
        Returns:
            [B, 512, 512]
        """
        if self.ae_type == "vecset":
            # _, x = self.base.encode(pc)

            sampled_pc = self.fps(pc)
            sampled_pc_embeddings = self.embed(sampled_pc)
            pc_embeddings = self.embed(pc)
            cross_attn, cross_ff = self.base.cross_attend_blocks
            x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
            x = cross_ff(x) + x

            if hasattr(self.base, "mean_fc"):
                mean = self.base.mean_fc(x)
                logvar = self.base.logvar_fc(x)
                posterior = DiagonalGaussianDistribution(mean, logvar)
                x = posterior.sample()

        elif self.ae_type == "hy3d2.1":
            pc = torch.cat([pc, torch.zeros_like(pc[..., :1])], dim=-1)
            x, fps_idx = self.base.encode(pc, fps_idx=None, sample_posterior=True)

        return x

    def decode(self, x: torch.Tensor, queries: torch.Tensor, learnable_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, 512, 3]
            queries: [B, N, 3]
        Returns:
            [B, N, 512]
        """
        if self.ae_type == "vecset":
            # o = self.base.decode(x, queries)
            if hasattr(self.base, "proj"):
                x = self.base.proj(x)
            for self_attn, self_ff in self.base.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x
            # cross attend from decoder queries to latents
            queries_embeddings = self.embed(queries)
            if learnable_embeddings is not None:
                queries_embeddings = torch.cat((queries_embeddings, learnable_embeddings), dim=1)
            latents = self.base.decoder_cross_attn(queries_embeddings, context=x)
            # optional decoder feedforward
            if self.base.decoder_ff is not None:
                latents = latents + self.base.decoder_ff(latents)

        elif self.ae_type == "hy3d2.1":
            x = self.base.decode(x)
            if queries.shape[-2] > 0:
                queries, normals = queries.split(self.input_dims, dim=-1)
                geo_decoder = self.base.geo_decoder
                queries_embeddings = geo_decoder.fourier_embedder(queries).to(x.dtype)
                queries_embeddings = geo_decoder.query_proj(torch.cat([queries_embeddings, normals], dim=-1))
                if learnable_embeddings is not None:
                    queries_embeddings = torch.cat((queries_embeddings, learnable_embeddings), dim=1)
            else:
                queries_embeddings = learnable_embeddings
            latents = self.base.geo_decoder(query_embeddings=queries_embeddings, latents=x)

        return latents

    def forward_base(self, pc: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """encode + decode + occupancy mlp
        Args:
            pc: [B, `self.N`, 3]
            queries: [B, N2, 3]
        Returns:
            [B, N2, 1]
        """
        if self.tune_decoder_cross_attn:
            warnings.warn(
                "Decoder cross-attn layers are tuned, so the output of occupancy MLP is not reliable anymore."
            )
        return self.base.forward(pc, queries)["logits"].unsqueeze(-1)

    def forward(
        self, pc: torch.Tensor, queries: torch.Tensor = None, joints: torch.Tensor = None, pose: torch.Tensor = None
    ):
        """
        Args:
            pc: [B, `self.N`, 3]
            queries: [B, N2, 3]
        Returns:
            [B, N2, `self.output_dim`]
        """
        if pc.shape[-1] > self.input_dim:
            pc = pc[..., : self.input_dim]
        x = self.encode(pc)

        learnable_embeddings = (
            self.bw_embed if self.predict_bw and self.bw_dot else None,
            self.joints_embed if self.predict_joints else None,
            self.global_embed if self.predict_global_trans else None,
            self.pose_embed if self.predict_pose_trans else None,
        )
        learnable_embeddings_length = [0 if x is None else x.shape[1] for x in learnable_embeddings]
        learnable_embeddings = [x for x in learnable_embeddings if x is not None]
        if learnable_embeddings:
            learnable_embeddings = torch.cat(learnable_embeddings, dim=1)
            learnable_embeddings = learnable_embeddings.expand(pc.shape[0], -1, -1).clone()
        else:
            learnable_embeddings = None
        if queries is None:
            assert learnable_embeddings is not None, "Nothing to predict"
            queries = torch.empty_like(pc[:, :0])  # placeholder
        elif queries.shape[-1] > self.input_dim:
            queries = queries[..., : self.input_dim]

        if self.predict_pose_trans and self.pose_input_joints:
            assert joints is not None and joints.shape[:-1] == (pc.shape[0], self.pose_embed.shape[1])
            joints_embed = self.joints_embedder(joints)
            pose_embed_length = learnable_embeddings_length[3]
            learnable_embeddings[:, -pose_embed_length:] = learnable_embeddings[:, -pose_embed_length:] + joints_embed
        if self.predict_bw and self.bw_input_joints:
            assert joints is not None and joints.shape[:-1] == (pc.shape[0], self.bw_embed.shape[1])
            joints_embed = self.joints_embedder(joints)
            bw_embed_length = learnable_embeddings_length[0]
            learnable_embeddings[:, :bw_embed_length] = learnable_embeddings[:, :bw_embed_length] + joints_embed

        logits = self.decode(x, queries, learnable_embeddings)
        if learnable_embeddings is not None:
            logits, logits_bw, logits_joints, logits_global, logits_pose = torch.split(
                logits, [queries.shape[1]] + learnable_embeddings_length, dim=1
            )

        if self.predict_bw:
            if self.bw_dot:
                logits, logits_bw = self.bw_cross_attn((logits, logits_bw), context=logits_bw)
                logits = self.bw_proj_queries(logits)
                logits_bw = self.bw_proj_bones(logits_bw)
                bw = torch.einsum("bnd,bkd->bnk", logits, logits_bw)
            else:
                bw: torch.Tensor = self.bw_head(logits)
            bw = self.actvn(bw)
            if not isinstance(self.actvn, nn.Softmax):
                bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)
            if self.output_actvn_log and self.training:
                bw = torch.log(bw)
        else:
            bw = None

        if self.predict_joints:
            if self.joints_attn_causal:
                if self.joints_attn_causal_discrete:
                    logits_joints = logits_joints.reshape(logits_joints.shape[0], self.output_dim, -1, self.embed_dim)
                joints = self.joints_head(logits_joints, out_gt=joints)
            else:
                joints = self.joints_head(logits_joints)
        else:
            joints = None

        global_trans = self.global_head(logits_global) if self.predict_global_trans else None

        if self.predict_pose_trans:
            if self.pose_attn_causal:
                if self.pose_attn_causal_discrete:
                    logits_pose = logits_pose.reshape(logits_pose.shape[0], self.output_dim, -1, self.embed_dim)
                pose_trans = self.pose_head(logits_pose, out_gt=pose)
            else:
                pose_trans = self.pose_head(logits_pose)
        else:
            pose_trans = None

        return Output(bw, joints, global_trans, pose_trans)

    def get_grid(self):
        if self.grid is None:
            x = np.linspace(-1, 1, self.grid_density + 1)
            y = np.linspace(-1, 1, self.grid_density + 1)
            z = np.linspace(-1, 1, self.grid_density + 1)
            xv, yv, zv = np.meshgrid(x, y, z)
            self.grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None]
        return self.grid

    @torch.no_grad()
    def to_mesh(self, pc: torch.Tensor):
        """forward_base + marching cubes
        Args:
            pc: [B, N, 3]
        """
        if self.tune_decoder_cross_attn:
            warnings.warn(
                "Decoder cross-attn layers are tuned, so the output of occupancy MLP is not reliable anymore."
            )
        import mcubes
        import trimesh

        output = self.forward_base(pc, self.get_grid().to(pc.device))
        mesh_list: list[trimesh.Trimesh] = []
        for volume in (
            output.view(-1, self.grid_density + 1, self.grid_density + 1, self.grid_density + 1)
            .permute(0, 2, 1, 3)
            .cpu()
            .numpy()
        ):
            verts, faces = mcubes.marching_cubes(volume, 0)
            verts *= 2.0 / self.grid_density
            verts -= 1.0
            mesh = trimesh.Trimesh(verts, faces)
            mesh_list.append(mesh)
        return mesh_list


class JointsDiscriminator(nn.Module):
    def __init__(self, joints_num=22 * 2, feat_dim=512):
        super().__init__()
        self.embed = JointsEmbedder(include_tail=True, out_mlp=False)
        self.mlp = nn.Sequential(
            nn.Linear(joints_num * self.embed.embedding_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim // 2, 1),
            # nn.Sigmoid(),
        )

    def forward(self, joints: torch.Tensor, mask=None):
        """
        Args:
            joints: [B, N, 3]
        Returns:
            [B, 1]
        """
        B, N, D = joints.shape
        if D == 6:
            joints = joints.view(B, N * 2, 3)
        elif D != 3:
            raise ValueError(f"Invalid joints shape: {joints.shape}")
        joints_embed = self.embed(joints)
        validity = self.mlp(joints_embed.view(B, -1))
        return validity

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class Transformer(nn.Module):
    def __init__(self, dim: int, depth=8, heads=8, dropout=0.1, norm_first=True, zero_init=False):
        super().__init__()

        warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

        # self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(
        #         nn.ModuleList(
        #             [
        #                 nn.LayerNorm(dim),
        #                 Attention(dim, heads=heads, dim_head=dim_head),
        #                 nn.LayerNorm(dim),
        #                 FeedForward(dim),
        #             ]
        #         )
        #     )
        # if zero_init:
        #     for _, attn, _, ff in self.layers:
        #         nn.init.zeros_(attn.to_out.weight)
        #         nn.init.zeros_(attn.to_out.bias)
        #         for m in ff.net:
        #             if isinstance(m, nn.Linear):
        #                 nn.init.zeros_(m.weight)
        #                 nn.init.zeros_(m.bias)

        layer = nn.TransformerEncoderLayer(
            dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=norm_first
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=depth)
        if zero_init:
            for layer in self.layers.layers:
                layer: nn.TransformerEncoderLayer
                nn.init.zeros_(layer.self_attn.out_proj.weight)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [B, N, D]
            mask: [N, N] bool (False means don't attend)
        Returns:
            [B, N, D]
        """
        # for norm1, attn, norm2, ff in self.layers:
        #     x = attn(norm1(x), mask=mask) + x
        #     x = ff(norm2(x)) + x
        # assert mask.dtype is torch.bool
        x = self.layers(x, mask=~mask)
        return x


class JointsDiscriminatorAttn(nn.Module):
    def __init__(self, num_joints=52 * 2, feat_dim=512, depth=8):
        super().__init__()
        self.embed = JointsEmbedder(include_tail=True, out_dim=feat_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints + 1, feat_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.transformer = Transformer(feat_dim, depth=depth)
        self.cls_head = nn.Linear(feat_dim, 1)

    def forward(self, joints: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            joints: [B, N, 6]
        Returns:
            [B, 1]
        """
        B, N, D = joints.shape
        assert D == 6
        joints_embed = self.embed(joints)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat((cls_tokens, joints_embed), dim=1)
        x += self.pos_embedding
        if mask is not None:
            assert mask.shape == (B, N, 2)
            mask = mask.view(B, N * 2, 1)
            mask = torch.cat((mask, torch.zeros_like(mask)[:, :1, :]), dim=1)  # cls_token
            mask = mask.squeeze(-1).unsqueeze(1).expand(-1, N * 2 + 1, -1)
        x = self.transformer(x, mask)
        x = x[:, 0]
        return self.cls_head(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


def adv_loss_d(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    B_real = logits_real.shape[0]
    B_fake = logits_fake.shape[0]
    loss_real = (F.softplus(-logits_real)).mean()
    loss_fake = (F.softplus(logits_fake)).mean()
    return (B_real / (B_real + B_fake)) * loss_real + (B_fake / (B_real + B_fake)) * loss_fake


def adv_loss_g(logits_fake: torch.Tensor) -> torch.Tensor:
    return (F.softplus(-logits_fake)).mean()


class JointsAttentionCausal(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        kinematic_tree: Joint,
        heads=8,
        depth=8,
        out_type="pose",
        include_joints_tail=False,
        out_dim=8,
        rotation_dim=4,
        query_type="embedding",
        zero_init=False,
    ):
        super().__init__()

        self.transformer = Transformer(feat_dim, depth=depth, heads=heads, norm_first=True, zero_init=zero_init)
        self.out_type = out_type
        self.out_dim = out_dim
        if self.out_type == "joints":
            self.encoder = JointsEmbedder(include_tail=include_joints_tail, out_dim=feat_dim)
            self.decoder = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, self.out_dim))
        elif self.out_type == "pose":
            self.encoder = nn.Sequential(nn.Linear(self.out_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))
            if zero_init:
                nn.init.zeros_(self.encoder[-1].weight)
                nn.init.zeros_(self.encoder[-1].bias)
            self.decoder = TransformMLP(
                feat_dim, transl_dim=self.out_dim - rotation_dim, rotation_dim=rotation_dim, scaling_dim=0
            )
        else:
            raise NotImplementedError(f"{self.out_type=}")
        self.query_type = query_type

        mask_attn = torch.zeros((len(kinematic_tree), len(kinematic_tree)), dtype=torch.bool)
        self.register_buffer("mask_attn", mask_attn, persistent=False)
        self.mask_attn: torch.Tensor

        mask_parent = mask_attn.clone()
        for joint in kinematic_tree:
            mask_attn[joint.index, joint.index] = True
            for parent in joint.parent_recursive:
                mask_attn[joint.index, parent.index] = True
            if joint.parent is not None:
                mask_parent[joint.index, joint.parent.index] = True
        self.register_buffer("mask_parent", mask_parent, persistent=False)
        self.mask_parent: torch.Tensor
        tree_levels_mask = torch.tensor(kinematic_tree.tree_levels_mask)
        self.register_buffer("tree_levels_mask", tree_levels_mask, persistent=False)
        self.tree_levels_mask: torch.Tensor

    def _forward(self, feat: torch.Tensor, out_gt: torch.Tensor = None):
        B, N, _ = feat.shape
        if self.out_type == "pose" and out_gt.shape[-1] == 6:
            from util.utils import matrix_to_ortho6d, ortho6d_to_matrix

            out_gt = matrix_to_ortho6d(ortho6d_to_matrix(out_gt))
        out_gt_feat: torch.Tensor = self.encoder(out_gt)  # B, N, D
        out_gt_feat = out_gt_feat.unsqueeze(1).expand(-1, N, -1, -1).clone()  # B, (N), N, D
        out_gt_feat[~self.mask_parent.expand(B, -1, -1)] = 0
        out_gt_feat = out_gt_feat.sum(-2)  # B, (N), D
        if self.query_type == "embedding":
            in_feat = out_gt_feat + feat
        else:
            raise NotImplementedError(f"{self.query_type=}")
        in_feat_attn = self.transformer(in_feat, mask=self.mask_attn)
        # out = self.decoder(feat + in_feat_attn)
        out = self.decoder(in_feat_attn)
        return out

    def forward(self, feat: torch.Tensor, out_gt: torch.Tensor = None):
        """
        Args:
            feat: [B, N, D]
        Returns:
            [B, N, `self.out_dim`]
        """
        B, N, _ = feat.shape

        if self.training:
            assert out_gt is not None and out_gt.shape == (B, N, self.out_dim)
            # # Avoid overfitting
            # if self.out_type == "joints":
            #     out_gt += torch.randn_like(out_gt) * 5e-2
            # elif self.out_type == "pose":
            #     from pytorch3d.transforms import euler_angles_to_matrix

            #     rand_rot = (torch.randn_like(out_gt[..., :3]) * 30) / 180 * torch.pi
            #     rand_rot = euler_angles_to_matrix(rand_rot, "XYZ")
            #     if out_gt.shape[-1] == 6:
            #         from util.utils import matrix_to_ortho6d, ortho6d_to_matrix

            #         out_gt = matrix_to_ortho6d(rand_rot @ ortho6d_to_matrix(out_gt))
            #     elif out_gt.shape[-1] == 4:
            #         from util.utils import matrix_to_quat, quat_to_matrix

            #         out_gt = matrix_to_quat(rand_rot @ quat_to_matrix(out_gt))
            #     else:
            #         raise NotImplementedError
            out = self._forward(feat, out_gt)
        else:
            out = torch.zeros((B, N, self.out_dim), dtype=feat.dtype, device=feat.device)
            for mask in self.tree_levels_mask:
                if not any(mask):
                    continue
                out_ = self._forward(feat, out)
                out[mask.expand(B, -1)] = out_[mask.expand(B, -1)].to(out.dtype)
        # assert out.isfinite().all()
        return out


def tokenize(coords: torch.Tensor, coord_continuous_range: tuple[float, float] = (-1.0, 1.0), num_discrete=256):
    low, high = coord_continuous_range
    normalized = (coords - low) / (high - low)  # (coords + 1) / 2, to [0, 1)
    scaled = normalized * num_discrete  # to [0, num_discrete)
    tokens = torch.floor(scaled).to(torch.long)
    tokens = torch.clamp(tokens, 0, num_discrete - 1)  # to [0, num_discrete - 1]
    return tokens


def detokenize(ids: torch.Tensor, coord_continuous_range: tuple[float, float] = (-1.0, 1.0), num_discrete=256):
    low, high = coord_continuous_range
    assert high > low
    assert (ids >= 0).all() and (ids < num_discrete).all()
    coords = ids.float()
    coords /= num_discrete
    coords = coords * (high - low) + low
    assert (coords >= low).all() and (coords < high).all()
    return coords


class JointsAttentionCausalDiscrete(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        kinematic_tree: Joint,
        heads=8,
        depth=8,
        out_type="joints",
        out_dim=6,
        query_type="embedding",
        zero_init=False,
    ):
        super().__init__()

        self.transformer = Transformer(feat_dim, depth=depth, heads=heads, norm_first=True, zero_init=zero_init)
        self.out_type = out_type
        self.out_dim = out_dim
        if self.out_type in ("joints", "pose"):
            self.coord_continuous_range = (-1.0, 1.0)
            self.num_discrete = 256
            self.embed_tokens = nn.Embedding(self.num_discrete, feat_dim)
            self.head = nn.Linear(feat_dim, self.num_discrete, bias=False)
        else:
            raise NotImplementedError(f"{self.out_type=}")
        self.query_type = query_type

        mask_attn = torch.zeros((len(kinematic_tree), len(kinematic_tree)), dtype=torch.bool)
        self.register_buffer("mask_attn", mask_attn, persistent=False)
        self.mask_attn: torch.Tensor

        mask_parent = mask_attn.clone()
        for joint in kinematic_tree:
            mask_attn[joint.index, joint.index] = True
            for parent in joint.parent_recursive:
                mask_attn[joint.index, parent.index] = True
            if joint.parent is not None:
                mask_parent[joint.index, joint.parent.index] = True
        self.register_buffer("mask_parent", mask_parent, persistent=False)
        self.mask_parent: torch.Tensor
        tree_levels_mask = torch.tensor(kinematic_tree.tree_levels_mask)
        self.register_buffer("tree_levels_mask", tree_levels_mask, persistent=False)
        self.tree_levels_mask: torch.Tensor

    def _forward(self, feat: torch.Tensor, out_gt_ids: torch.Tensor = None, out_gt_mask: torch.Tensor = None):
        """
        Args:
            feat: [B, N, `self.out_dim`, D]
            out_gt_ids: [B, N, `self.out_dim`] int
            out_gt_mask: [B, N] bool
        Returns:
            next_tokens_logits: [B, N, `self.out_dim`, `self.num_discrete`]
            next_tokens: [B, N, `self.out_dim`] int
        """
        B, N, _, D = feat.shape
        num_tokens_per_joint = out_gt_ids.shape[-1]
        out_gt_feat: torch.Tensor = self.embed_tokens(out_gt_ids)  # B, N, 6, D
        if out_gt_mask is not None:
            out_gt_feat[~out_gt_mask] = 0

        out_gt_feat = out_gt_feat.unsqueeze(1).expand(-1, N, -1, -1, -1).clone()  # B, (N), N, 6, D
        out_gt_feat[~self.mask_parent.expand(B, -1, -1)] = 0
        out_gt_feat = out_gt_feat.sum(-3)  # B, (N), 6, D
        if self.query_type == "embedding":
            in_feat = out_gt_feat + feat
        else:
            raise NotImplementedError(f"{self.query_type=}")
        # mask = torch.triu(
        #     torch.ones((N * self.out_dim, N * self.out_dim), dtype=torch.bool, device=feat.device), diagonal=1
        # )
        mask = torch.kron(
            self.mask_attn,
            torch.ones((num_tokens_per_joint, num_tokens_per_joint), dtype=bool, device=self.mask_attn.device),
        )

        in_feat_attn = self.transformer(in_feat.reshape(B, -1, D), mask=mask)  # B, N * 6, D
        next_tokens_logits: torch.Tensor = self.head(in_feat_attn)  # B, N * 6, 256
        next_tokens_logits = next_tokens_logits.reshape(B, N, num_tokens_per_joint, self.num_discrete)  # B, N, 6, 256
        next_tokens = torch.argmax(next_tokens_logits, dim=-1)
        return next_tokens_logits, next_tokens

    def forward(self, feat: torch.Tensor, out_gt: torch.Tensor = None):
        """
        Args:
            feat: [B, N, `self.out_dim`, D]
        Returns:
            training: [B, N, `self.out_dim`, `self.num_discrete`]
            else: [B, N, `self.out_dim`]
        """
        B, N = feat.shape[:2]

        if self.training:
            assert out_gt is not None and out_gt.shape == (B, N, self.out_dim)
            if self.out_type == "pose" and out_gt.shape[-1] == 6:
                from util.utils import matrix_to_ortho6d, ortho6d_to_matrix

                out_gt = matrix_to_ortho6d(ortho6d_to_matrix(out_gt))
            out_gt_ids = tokenize(out_gt, self.coord_continuous_range, self.num_discrete)
            next_tokens_logits, next_tokens = self._forward(feat, out_gt_ids)
            return next_tokens_logits
        else:
            out_ids = torch.zeros((B, N, self.out_dim), dtype=int, device=feat.device)
            out_mask = torch.zeros((B, N), dtype=torch.bool, device=feat.device)
            for mask in self.tree_levels_mask:
                if not any(mask):
                    continue
                next_tokens_logits, next_tokens = self._forward(feat, out_ids, out_mask)
                out_ids[mask.expand(B, -1)] = next_tokens[mask.expand(B, -1)]
                out_mask[mask.expand(B, -1)] = True
            out = detokenize(out_ids, self.coord_continuous_range, self.num_discrete)
            return out
