"""Single-file Dense DeiT trainer for ImageNet pretraining/debugging.

This script is intentionally self-contained for Dense DeiT:
- Model classes live in this file (PatchEmbed/Attention/MLP/Block/DeiT).
- Distillation loss lives in this file.
- Training/eval loop lives in this file.

It reuses only shared infrastructure for:
- ImageNet dataset/loader construction (`pretrain.datasets`)
- Optimizer/scheduler builders (`pretrain.optim`)
- Distributed/logger helpers (`pretrain.utils`)
"""

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import yaml
from timm.data import Mixup
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import accuracy
from torch.hub import load_state_dict_from_url

from pretrain.datasets import build_imagenet_datasets, build_imagenet_loaders
from pretrain.optim import build_optimizer, build_scheduler
from pretrain.utils import MetricLogger, init_distributed_mode, is_main_process, set_seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"yes", "true", "t", "y", "1"}:
        return True
    if s in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _maybe_get(d, key):
    if isinstance(d, dict) and key in d:
        return d[key]
    return None


def _set_if_absent(dst, key, value):
    if key not in dst and value is not None:
        dst[key] = value


def _infer_dense_deit_name_from_strings(*values):
    for value in values:
        text = str(value or "").lower()
        if "tiny" in text:
            return "deit_tiny"
        if "small" in text:
            return "deit_small"
        if "base" in text:
            return "deit_base"
    return None


def normalize_config(cfg):
    """Normalize nested YAML config to argparse flat keys."""
    cfg = deepcopy(cfg or {})
    out = dict(cfg)

    _set_if_absent(out, "batch_size", _maybe_get(cfg, "trBatch"))
    _set_if_absent(out, "workers", _maybe_get(cfg, "nworkers"))
    _set_if_absent(out, "dataset_name", _maybe_get(cfg, "train_db_name"))
    _set_if_absent(out, "data_path", _maybe_get(cfg, "data_path"))
    _set_if_absent(out, "input_size", _maybe_get(cfg, "input_size"))
    _set_if_absent(out, "nb_classes", _maybe_get(cfg, "nb_classes"))

    _set_if_absent(out, "epochs", _maybe_get(cfg, "epochs"))
    _set_if_absent(out, "save_freq", _maybe_get(cfg, "save_freq"))
    _set_if_absent(out, "eval_freq", _maybe_get(cfg, "eval_freq"))
    _set_if_absent(out, "print_freq", _maybe_get(cfg, "print_freq"))
    _set_if_absent(out, "seed", _maybe_get(cfg, "seed"))
    _set_if_absent(out, "amp", _maybe_get(cfg, "amp"))
    _set_if_absent(out, "pin_mem", _maybe_get(cfg, "pin_mem"))
    _set_if_absent(out, "dist_eval", _maybe_get(cfg, "dist_eval"))

    opt_kwargs = _maybe_get(cfg, "optimizer_kwargs") or {}
    _set_if_absent(out, "opt", _maybe_get(cfg, "optimizer"))
    _set_if_absent(out, "lr", _maybe_get(opt_kwargs, "lr"))
    _set_if_absent(out, "momentum", _maybe_get(opt_kwargs, "momentum"))
    _set_if_absent(out, "weight_decay", _maybe_get(opt_kwargs, "weight_decay"))
    _set_if_absent(out, "opt_betas", _maybe_get(opt_kwargs, "opt_betas"))
    _set_if_absent(out, "opt_eps", _maybe_get(opt_kwargs, "opt_eps"))

    sched_kwargs = _maybe_get(cfg, "scheduler_kwargs") or {}
    _set_if_absent(out, "warmup_epochs", _maybe_get(sched_kwargs, "warmup_epochs"))
    _set_if_absent(out, "min_lr", _maybe_get(sched_kwargs, "min_lr"))
    _set_if_absent(out, "unscale_lr", _maybe_get(sched_kwargs, "unscale_lr"))

    bn_kwargs = _maybe_get(cfg, "backbone_kwargs") or {}
    dense_name = _infer_dense_deit_name_from_strings(
        _maybe_get(out, "model"),
        _maybe_get(bn_kwargs, "model_name"),
    )
    if dense_name is not None:
        out["model"] = dense_name
    elif str(_maybe_get(out, "model") or "").startswith("moe_vit_"):
        # Safe fallback for configs copied from MoE presets.
        out["model"] = str(out["model"]).replace("moe_vit_", "deit_")

    _set_if_absent(out, "patch_size", _maybe_get(bn_kwargs, "patch_size"))
    _set_if_absent(out, "in_chans", _maybe_get(bn_kwargs, "in_chans"))
    _set_if_absent(out, "embed_dim", _maybe_get(bn_kwargs, "embed_dim"))
    _set_if_absent(out, "depth", _maybe_get(bn_kwargs, "depth"))
    _set_if_absent(out, "num_heads", _maybe_get(bn_kwargs, "num_heads"))
    _set_if_absent(out, "mlp_ratio", _maybe_get(bn_kwargs, "mlp_ratio"))
    _set_if_absent(out, "qkv_bias", _maybe_get(bn_kwargs, "qkv_bias"))
    _set_if_absent(out, "drop", _maybe_get(bn_kwargs, "drop_rate"))
    _set_if_absent(out, "attn_drop_rate", _maybe_get(bn_kwargs, "attn_drop_rate"))
    _set_if_absent(out, "drop_path", _maybe_get(bn_kwargs, "drop_path_rate"))
    _set_if_absent(out, "distilled", _maybe_get(bn_kwargs, "distilled"))

    aug_kwargs = _maybe_get(cfg, "aug_kwargs") or {}
    _set_if_absent(out, "color_jitter", _maybe_get(aug_kwargs, "color_jitter"))
    _set_if_absent(out, "aa", _maybe_get(aug_kwargs, "aa"))
    _set_if_absent(out, "train_interpolation", _maybe_get(aug_kwargs, "train_interpolation"))
    _set_if_absent(out, "reprob", _maybe_get(aug_kwargs, "reprob"))
    _set_if_absent(out, "remode", _maybe_get(aug_kwargs, "remode"))
    _set_if_absent(out, "recount", _maybe_get(aug_kwargs, "recount"))
    _set_if_absent(out, "repeated_aug", _maybe_get(aug_kwargs, "repeated_aug"))
    _set_if_absent(out, "mixup", _maybe_get(aug_kwargs, "mixup"))
    _set_if_absent(out, "cutmix", _maybe_get(aug_kwargs, "cutmix"))
    _set_if_absent(out, "cutmix_minmax", _maybe_get(aug_kwargs, "cutmix_minmax"))
    _set_if_absent(out, "mixup_prob", _maybe_get(aug_kwargs, "mixup_prob"))
    _set_if_absent(out, "mixup_switch_prob", _maybe_get(aug_kwargs, "mixup_switch_prob"))
    _set_if_absent(out, "mixup_mode", _maybe_get(aug_kwargs, "mixup_mode"))
    _set_if_absent(out, "smoothing", _maybe_get(aug_kwargs, "smoothing"))

    dist_kwargs = _maybe_get(cfg, "distillation_kwargs") or {}
    _set_if_absent(out, "distillation_type", _maybe_get(dist_kwargs, "distillation_type"))
    _set_if_absent(out, "teacher_model", _maybe_get(dist_kwargs, "teacher_model"))
    _set_if_absent(out, "teacher_path", _maybe_get(dist_kwargs, "teacher_path"))
    _set_if_absent(out, "distillation_alpha", _maybe_get(dist_kwargs, "distillation_alpha"))
    _set_if_absent(out, "distillation_tau", _maybe_get(dist_kwargs, "distillation_tau"))
    return out


def _resolve_env_config_path(config_path):
    if not config_path:
        return ""
    if config_path == "configs/path_env.yml" and not os.path.exists(config_path) and os.path.exists("configs/env.yml"):
        return "configs/env.yml"
    return config_path


def _resolve_data_path_from_env(args):
    env_cfg_path = _resolve_env_config_path(getattr(args, "config_path", ""))
    if not env_cfg_path or not os.path.exists(env_cfg_path):
        return

    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f) or {}

    hf_token = str(env_cfg.get("huggingface_access_token", "") or "").strip()
    if hf_token:
        hf_token = os.path.expandvars(os.path.expanduser(hf_token)).strip()
        if hf_token and "$" not in hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    if getattr(args, "data_path", ""):
        return
    dataset_roots = env_cfg.get("dataset_roots", {}) or {}
    data_path = dataset_roots.get(args.dataset_name, "")
    if data_path:
        args.data_path = os.path.expandvars(os.path.expanduser(str(data_path)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise RuntimeError(
                f"Input size ({h}x{w}) does not match model size "
                f"({self.img_size[0]}x{self.img_size[1]})."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DenseDeiT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.distilled = bool(distilled)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        num_tokens = 2 if self.distilled else 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes) if self.distilled else None

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {"pos_embed", "cls_token"}
        if self.dist_token is not None:
            names.add("dist_token")
        return names

    def forward_features(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        if self.dist_token is not None:
            dist = self.dist_token.expand(b, -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        else:
            x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        tokens = self.forward_features(x)
        logits = self.head(tokens[:, 0])
        if self.head_dist is None:
            return logits

        logits_dist = self.head_dist(tokens[:, 1])
        if self.training:
            return logits, logits_dist
        return (logits + logits_dist) * 0.5


MODEL_SPECS = {
    "deit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0},
    "deit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
    "deit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
}


def build_dense_deit_model(args):
    if args.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model}'. Expected one of: {list(MODEL_SPECS.keys())}")
    spec = MODEL_SPECS[args.model]
    embed_dim = int(args.embed_dim) if args.embed_dim > 0 else int(spec["embed_dim"])
    depth = int(args.depth) if args.depth > 0 else int(spec["depth"])
    num_heads = int(args.num_heads) if args.num_heads > 0 else int(spec["num_heads"])
    mlp_ratio = float(args.mlp_ratio) if args.mlp_ratio > 0 else float(spec["mlp_ratio"])
    return DenseDeiT(
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        num_classes=args.nb_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=bool(args.qkv_bias),
        distilled=bool(args.distilled),
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path,
    )


class DistillationLoss(nn.Module):
    """DeiT-style distillation loss."""

    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = str(distillation_type)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.last_stats = {}

    def forward(self, inputs, outputs, labels):
        if not isinstance(outputs, tuple):
            outputs = (outputs, None)

        outputs_cls, outputs_dist = outputs
        base_loss = self.base_criterion(outputs_cls, labels)
        base_loss_val = float(base_loss.detach().item())

        if self.distillation_type == "none":
            self.last_stats = {
                "loss_base": base_loss_val,
                "loss_distill": 0.0,
                "loss_total_no_aux": base_loss_val,
            }
            return base_loss

        if outputs_dist is None:
            raise ValueError(
                "Distillation is enabled but model did not return distillation logits. "
                "Set --distilled true for soft/hard distillation."
            )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            t = self.tau
            distill_loss = F.kl_div(
                F.log_softmax(outputs_dist / t, dim=1),
                F.log_softmax(teacher_outputs / t, dim=1),
                reduction="sum",
                log_target=True,
            ) * (t * t) / outputs_dist.numel()
        elif self.distillation_type == "hard":
            distill_loss = F.cross_entropy(outputs_dist, teacher_outputs.argmax(dim=1))
        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")

        total_loss = base_loss * (1.0 - self.alpha) + distill_loss * self.alpha
        self.last_stats = {
            "loss_base": base_loss_val,
            "loss_distill": float(distill_loss.detach().item()),
            "loss_total_no_aux": float(total_loss.detach().item()),
        }
        return total_loss


def build_criterion(args, teacher_model=None):
    if args.mixup > 0.0 or args.cutmix > 0.0 or args.cutmix_minmax is not None:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = nn.CrossEntropyLoss()
    return DistillationLoss(
        base_criterion=base_criterion,
        teacher_model=teacher_model,
        distillation_type=args.distillation_type,
        alpha=args.distillation_alpha,
        tau=args.distillation_tau,
    )


def _strip_common_state_dict_prefix(state_dict, model_keys):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    for prefix in ("module.", "model.", "model.module."):
        prefixed = [k for k in state_dict.keys() if isinstance(k, str) and k.startswith(prefix)]
        if len(prefixed) < max(1, int(0.8 * len(state_dict))):
            continue
        stripped = {}
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith(prefix):
                stripped[k[len(prefix) :]] = v
            else:
                stripped[k] = v
        overlap_before = sum(1 for k in state_dict.keys() if k in model_keys)
        overlap_after = sum(1 for k in stripped.keys() if k in model_keys)
        if overlap_after > overlap_before:
            if is_main_process():
                print(
                    f"[Teacher] stripped checkpoint key prefix '{prefix}' "
                    f"(overlap {overlap_before} -> {overlap_after})"
                )
            return stripped
    return state_dict


def build_teacher_model(args, device):
    if args.distillation_type == "none":
        return None
    if not args.teacher_path:
        raise ValueError("--teacher-path is required when distillation is enabled")

    teacher = create_model(args.teacher_model, pretrained=False, num_classes=args.nb_classes)
    if args.teacher_path.startswith("http://") or args.teacher_path.startswith("https://"):
        checkpoint = load_state_dict_from_url(args.teacher_path, map_location="cpu", progress=True)
    else:
        checkpoint = torch.load(args.teacher_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        teacher_state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_ema" in checkpoint:
        teacher_state = checkpoint["model_ema"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        teacher_state = checkpoint["state_dict"]
    else:
        teacher_state = checkpoint

    model_keys = set(teacher.state_dict().keys())
    teacher_state = _strip_common_state_dict_prefix(teacher_state, model_keys)
    load_info = teacher.load_state_dict(teacher_state, strict=False)
    missing = list(getattr(load_info, "missing_keys", []) or [])
    unexpected = list(getattr(load_info, "unexpected_keys", []) or [])
    total_keys = len(model_keys)
    loaded_keys = max(0, total_keys - len(missing))
    loaded_ratio = float(loaded_keys) / float(max(total_keys, 1))

    if is_main_process():
        print(
            "[Teacher] load summary: "
            f"loaded={loaded_keys}/{total_keys} ({loaded_ratio:.1%}), "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[Teacher] sample missing keys: {missing[:10]}")
        if unexpected:
            print(f"[Teacher] sample unexpected keys: {unexpected[:10]}")
    if loaded_ratio < 0.5:
        raise RuntimeError(
            "Teacher checkpoint appears incompatible with teacher model "
            f"({loaded_keys}/{total_keys} keys loaded)."
        )

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _save_checkpoint(args, epoch, model, optimizer, scheduler, scaler, best_acc1, is_best=False):
    periodic = ((epoch + 1) % int(args.save_freq) == 0)
    if not periodic and not bool(is_best):
        return
    os.makedirs(args.output_dir, exist_ok=True)
    state = {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_acc1": float(best_acc1),
        "args": vars(args),
    }
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if is_main_process():
        torch.save(state, latest_path)
        if periodic:
            torch.save(state, os.path.join(args.output_dir, f"checkpoint_{epoch + 1:04d}.pth"))
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "checkpoint_best.pth"))


def _auto_resume(args, model, optimizer=None, scheduler=None, scaler=None):
    if not args.resume:
        return 0, 0.0
    resume_path = args.resume
    if os.path.isdir(resume_path):
        resume_path = os.path.join(resume_path, "checkpoint_latest.pth")
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    if is_main_process():
        print(f"Resuming from: {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu")
    _unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_acc1", 0.0))


def _resolve_output_dir(args):
    base = os.path.expandvars(os.path.expanduser(str(args.output_dir)))
    if args.resume:
        args.output_dir = base
        return
    leaf = os.path.basename(os.path.normpath(base))
    already_timestamped = (
        len(leaf) == 9 and leaf[4] == "_" and leaf[:4].isdigit() and leaf[5:].isdigit()
    )
    if already_timestamped:
        args.output_dir = base
        return
    args.output_dir = os.path.join(base, datetime.now().strftime("%m%d_%H%M"))


def _resolve_ddp_find_unused_parameters(args):
    # Distilled head can be unused if distillation is disabled.
    return bool(args.distilled) and str(args.distillation_type).lower() == "none"


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scaler, mixup_fn, args):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
            criterion_stats = getattr(criterion, "last_stats", None)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
        if isinstance(criterion_stats, dict):
            metric_logger.update(**criterion_stats)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    loss_sum = 0.0
    correct1_sum = 0.0
    correct5_sum = 0.0
    sample_count = 0

    for images, target in metric_logger.log_every(data_loader, args.print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        bs = int(images.shape[0])
        loss_sum += float(loss.item() * bs)
        correct1_sum += float(acc1.item() * bs / 100.0)
        correct5_sum += float(acc5.item() * bs / 100.0)
        sample_count += bs

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=bs)
        metric_logger.meters["acc5"].update(acc5.item(), n=bs)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor(
            [loss_sum, correct1_sum, correct5_sum, float(sample_count)],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        loss_sum, correct1_sum, correct5_sum, sample_count = stats.tolist()

    sample_count = max(sample_count, 1.0)
    acc1_global = 100.0 * correct1_sum / sample_count
    acc5_global = 100.0 * correct5_sum / sample_count
    loss_global = loss_sum / sample_count

    print(
        "* Acc@1 {top1:.3f} Acc@5 {top5:.3f} loss {loss:.4f}".format(
            top1=acc1_global, top5=acc5_global, loss=loss_global
        )
    )
    return {"loss": loss_global, "acc1": acc1_global, "acc5": acc5_global}


def get_args_parser():
    parser = argparse.ArgumentParser("Dense DeiT ImageNet pretrain", add_help=False)
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--config-path", "--config-env", dest="config_path", default="configs/path_env.yml", type=str)
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--dataset-name", default="ImageNet1K", type=str)
    parser.add_argument("--output-dir", default="./output/pretrain_dense_deit", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--save-freq", default=10, type=int)
    parser.add_argument("--eval-freq", default=10, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dev-test", default=False, type=str2bool)

    parser.add_argument("--model", default="deit_small", choices=["deit_tiny", "deit_small", "deit_base"])
    parser.add_argument("--nb-classes", default=1000, type=int)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--patch-size", default=16, type=int)
    parser.add_argument("--in-chans", default=3, type=int)
    parser.add_argument("--embed-dim", default=-1, type=int)
    parser.add_argument("--depth", default=-1, type=int)
    parser.add_argument("--num-heads", default=-1, type=int)
    parser.add_argument("--mlp-ratio", default=-1.0, type=float)
    parser.add_argument("--qkv-bias", default=True, type=str2bool)
    parser.add_argument("--distilled", default=False, type=str2bool)
    parser.add_argument("--drop", default=0.0, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)
    parser.add_argument("--attn-drop-rate", default=0.0, type=float)

    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--opt-betas", default=None, nargs="+", type=float)
    parser.add_argument("--opt-eps", default=1e-8, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--unscale-lr", action="store_true")
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--clip-grad", default=None, type=float)

    parser.add_argument("--color-jitter", default=0.3, type=float)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--train-interpolation", default="bicubic", type=str)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)
    parser.add_argument("--repeated-aug", action="store_true", dest="repeated_aug")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")

    parser.add_argument("--mixup", default=0.8, type=float)
    parser.add_argument("--cutmix", default=1.0, type=float)
    parser.add_argument("--cutmix-minmax", default=None, type=float, nargs="+")
    parser.add_argument("--mixup-prob", default=1.0, type=float)
    parser.add_argument("--mixup-switch-prob", default=0.5, type=float)
    parser.add_argument("--mixup-mode", default="batch", type=str)
    parser.add_argument("--smoothing", default=0.1, type=float)

    parser.add_argument("--distillation-type", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--teacher-model", default="regnety_160", type=str)
    parser.add_argument(
        "--teacher-path",
        default="https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth",
        type=str,
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=1.0, type=float)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--pin-mem", action="store_true", dest="pin_mem")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.add_argument("--amp", default=True, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--dist-eval", action="store_true", dest="dist_eval")
    parser.add_argument("--no-dist-eval", action="store_false", dest="dist_eval")

    parser.set_defaults(
        pin_mem=True,
        dist_eval=False,
        repeated_aug=True,
    )
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", type=str)
    config_parser.add_argument("--config-path", "--config-env", dest="config_path", default="configs/path_env.yml", type=str)
    cfg_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        "Dense DeiT ImageNet pretrain",
        parents=[get_args_parser()],
    )
    if cfg_args.config:
        with open(cfg_args.config, "r", encoding="utf-8") as f:
            cfg = normalize_config(yaml.safe_load(f) or {})
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args.config = cfg_args.config
    args.config_path = _resolve_env_config_path(cfg_args.config_path)
    return args


def _dump_run_metadata(args):
    if not is_main_process():
        return
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "command": " ".join(sys.argv),
        "args": {k: v for k, v in vars(args).items()},
    }
    path = os.path.join(args.output_dir, "run_args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved run args to {path}")


def main():
    args = parse_args()
    if int(args.eval_freq) < 1:
        raise ValueError(f"--eval-freq must be >= 1, got {args.eval_freq}")
    init_distributed_mode(args)
    _resolve_data_path_from_env(args)

    if args.distillation_type != "none" and not args.distilled:
        raise ValueError("--distilled must be true when --distillation-type is soft/hard")

    if args.unscale_lr:
        effective_lr = args.lr
    else:
        effective_lr = args.lr * (args.batch_size * args.world_size) / 512.0
    args.lr = effective_lr

    device = torch.device(args.device)
    set_seed(args.seed, rank=args.rank)
    cudnn.benchmark = True

    _resolve_output_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process():
        print(f"Resolved output dir: {args.output_dir}")
    _dump_run_metadata(args)

    if args.data_path == "":
        raise ValueError(
            "--data-path is required (or set dataset_roots.<dataset_name> in "
            f"{args.config_path} and use --dataset-name)"
        )
    dataset_train, dataset_val, nb_classes = build_imagenet_datasets(args)
    args.nb_classes = nb_classes
    loader_train, loader_val = build_imagenet_loaders(dataset_train, dataset_val, args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = build_dense_deit_model(args)
    model.to(device)

    if args.distributed:
        ddp_find_unused = _resolve_ddp_find_unused_parameters(args)
        if is_main_process():
            print(
                "DDP find_unused_parameters:",
                ddp_find_unused,
                f"(distilled={bool(args.distilled)}, distillation_type={args.distillation_type})",
            )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=ddp_find_unused,
        )
    model_without_ddp = _unwrap_model(model)

    optimizer = build_optimizer(args, model_without_ddp)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_model = build_teacher_model(args, device)
    criterion = build_criterion(args, teacher_model=teacher_model)

    start_epoch, best_acc1 = _auto_resume(
        args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    if args.eval:
        test_stats = evaluate(loader_val, model, device, args)
        if is_main_process():
            print(f"Eval only: Acc@1={test_stats['acc1']:.3f} Acc@5={test_stats['acc5']:.3f}")
        return

    if args.dev_test:
        if is_main_process():
            print("Run one pre-flight validation (--dev-test=true)")
        _ = evaluate(loader_val, model, device, args)

    print("Start training (Dense DeiT)")
    start_time = time.time()
    log_path = os.path.join(args.output_dir, "log.txt")

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            mixup_fn=mixup_fn,
            args=args,
        )
        scheduler.step(epoch + 1)

        do_eval = ((epoch + 1) % int(args.eval_freq) == 0) or ((epoch + 1) == args.epochs)
        if do_eval:
            test_stats = evaluate(loader_val, model, device, args)
            acc1 = float(test_stats.get("acc1", 0.0))
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
        else:
            test_stats = {}
            is_best = False
            if is_main_process():
                print(f"Skip eval at epoch {epoch + 1} (eval_freq={args.eval_freq})")

        _save_checkpoint(
            args=args,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_acc1=best_acc1,
            is_best=is_best,
        )

        if is_main_process():
            log_stats = {
                "epoch": epoch,
                "n_parameters": sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad),
                "best_acc1": float(best_acc1),
                "eval_performed": bool(do_eval),
            }
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            if do_eval:
                log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            print(json.dumps(log_stats))

    total_time = time.time() - start_time
    total_time_str = str(time.strftime("%H:%M:%S", time.gmtime(total_time)))
    if is_main_process():
        print(f"Training time {total_time_str}")


if __name__ == "__main__":
    main()
