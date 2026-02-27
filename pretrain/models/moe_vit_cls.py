"""Classification wrapper over this repo's MoE ViT encoder."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

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
    use_checkpointing: bool


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
            use_checkpointing=cfg.use_checkpointing,
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.head_dist = nn.Linear(cfg.embed_dim, cfg.num_classes) if cfg.distilled else None
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.head_dist is not None:
            trunc_normal_(self.head_dist.weight, std=0.02)
            nn.init.zeros_(self.head_dist.bias)

        # Warm-start cls/dist heads and pre-head norm from the same DeiT checkpoint
        # used for encoder initialization when random_init is disabled.
        if not cfg.random_init:
            self._warm_start_cls_and_dist_heads()

    @staticmethod
    def _unwrap_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            model_sd = checkpoint.get("model")
            if isinstance(model_sd, dict):
                return model_sd
            nested_sd = checkpoint.get("state_dict")
            if isinstance(nested_sd, dict):
                return nested_sd
        return checkpoint

    @staticmethod
    def _copy_param_if_match(module, attr_name, state_dict, key):
        if module is None:
            return False, f"{key}: module is None"
        if key not in state_dict:
            return False, f"{key}: missing in checkpoint"
        target = getattr(module, attr_name)
        source = state_dict[key]
        if tuple(target.shape) != tuple(source.shape):
            return False, (
                f"{key}: shape mismatch checkpoint {tuple(source.shape)} "
                f"!= target {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(source.to(dtype=target.dtype, device=target.device))
        return True, f"{key}: loaded"

    def _warm_start_cls_and_dist_heads(self):
        default_cfg = getattr(self.encoder, "default_cfg", None) or {}
        url = str(default_cfg.get("url", "") or "")
        if not url:
            print("[INIT] skip head warm-start (no pretrained URL in encoder.default_cfg)")
            return

        try:
            checkpoint = load_state_dict_from_url(url, map_location="cpu", progress=False)
            state_dict = self._unwrap_state_dict(checkpoint)
        except Exception as exc:
            print(f"[INIT] failed to load head warm-start checkpoint from {url}: {exc}")
            return

        loaded_msgs = []
        skipped_msgs = []

        for module, attr_name, key in (
            (self.norm, "weight", "norm.weight"),
            (self.norm, "bias", "norm.bias"),
            (self.head, "weight", "head.weight"),
            (self.head, "bias", "head.bias"),
        ):
            ok, msg = self._copy_param_if_match(module, attr_name, state_dict, key)
            (loaded_msgs if ok else skipped_msgs).append(msg)

        if self.head_dist is not None:
            for module, attr_name, key in (
                (self.head_dist, "weight", "head_dist.weight"),
                (self.head_dist, "bias", "head_dist.bias"),
            ):
                ok, msg = self._copy_param_if_match(module, attr_name, state_dict, key)
                (loaded_msgs if ok else skipped_msgs).append(msg)

        if loaded_msgs:
            print("[INIT] warm-started classifier params:", ", ".join(loaded_msgs))
        if skipped_msgs:
            print("[INIT] skipped classifier params:", ", ".join(skipped_msgs))

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
