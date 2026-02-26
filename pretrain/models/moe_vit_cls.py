"""Classification wrapper over this repo's MoE ViT encoder."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from timm.layers import trunc_normal_

from models.ckpt_vision_transformer_moe import VisionTransformerMoE


@dataclass
class MoEViTConfig:
    model_name: str
    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    distilled: bool
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float
    num_classes: int
    random_init: bool
    pos_embed_interp: bool
    align_corners: bool
    moe_mlp_ratio: float
    moe_experts: int
    moe_top_k: int
    moe_gate_type: str
    vmoe_noisy_std: float
    gate_dim: int
    gate_task_specific_dim: int
    multi_gate: bool
    world_size: int


class MoEViTForImageNet(nn.Module):
    def __init__(self, cfg: MoEViTConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = VisionTransformerMoE(
            model_name=cfg.model_name,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            num_classes=cfg.num_classes,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=None,
            representation_size=None,
            distilled=cfg.distilled,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            hybrid_backbone=None,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            pos_embed_interp=cfg.pos_embed_interp,
            random_init=cfg.random_init,
            align_corners=cfg.align_corners,
            act_layer=None,
            weight_init="",
            moe_mlp_ratio=cfg.moe_mlp_ratio,
            moe_experts=cfg.moe_experts,
            moe_top_k=cfg.moe_top_k,
            world_size=cfg.world_size,
            gate_dim=cfg.gate_dim,
            gate_return_decoupled_activation=False,
            moe_gate_type=cfg.moe_gate_type,
            vmoe_noisy_std=cfg.vmoe_noisy_std,
            gate_task_specific_dim=cfg.gate_task_specific_dim,
            multi_gate=cfg.multi_gate,
            num_tasks=-1,
            regu_experts_fromtask=False,
            num_experts_pertask=-1,
            gate_input_ahead=False,
            regu_sem=False,
            sem_force=False,
            regu_subimage=False,
            expert_prune=False,
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.head_dist = nn.Linear(cfg.embed_dim, cfg.num_classes) if cfg.distilled else None
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.head_dist is not None:
            trunc_normal_(self.head_dist.weight, std=0.02)
            nn.init.zeros_(self.head_dist.bias)

    def no_weight_decay(self):
        skip = set()
        no_wd = getattr(self.encoder, "no_weight_decay", None)
        if callable(no_wd):
            no_wd = no_wd()
        if no_wd is not None:
            skip.update({f"encoder.{name}" for name in no_wd})
        return skip

    def forward(self, x):
        encoder_out = self.encoder(x)
        if isinstance(encoder_out, tuple):
            tokens, cv_loss = encoder_out
        else:
            tokens, cv_loss = encoder_out, None

        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token output [B, N, C], got shape {tuple(tokens.shape)}")

        tokens = self.norm(tokens)
        logits_cls = self.head(tokens[:, 0])

        if self.head_dist is None:
            return {"logits": logits_cls, "cv_loss": cv_loss}

        if tokens.shape[1] < 2:
            raise RuntimeError(
                "Distilled model expects [cls, dist, ...] tokens, got shape "
                f"{tuple(tokens.shape)}"
            )

        logits_dist = self.head_dist(tokens[:, 1])
        if self.training:
            logits = (logits_cls, logits_dist)
        else:
            logits = (logits_cls + logits_dist) / 2
        return {"logits": logits, "cv_loss": cv_loss}
