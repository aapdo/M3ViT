"""ImageNet pretrain entrypoint for MoE ViT (DeiT-inspired skeleton)."""

import argparse
import importlib
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone

import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.hub import load_state_dict_from_url
from timm.data import Mixup
from timm.models import create_model
try:
    import fmoe
except ImportError:
    fmoe = None
try:
    from timm.utils import ModelEma as TimmModelEma
except ImportError:
    try:
        from timm.utils import ModelEmaV2 as TimmModelEma
    except ImportError:
        TimmModelEma = None

from pretrain.datasets import build_imagenet_datasets, build_imagenet_loaders
from pretrain.engine import evaluate, train_one_epoch
from pretrain.models import build_criterion, build_model
from pretrain.optim import build_optimizer, build_scheduler
from pretrain.utils import init_distributed_mode, is_main_process, set_seed
from pretrain.utils.checkpoint import auto_resume, save_checkpoint
from utils.helpers import set_upcycling_runtime_options
from utils.tracing import patch_and_log_initializations, restore_original_initializations


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2bool_or_auto(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"auto", "none", ""}:
        return None
    return str2bool(v)


def _maybe_get(d, key):
    if isinstance(d, dict) and key in d:
        return d[key]
    return None


def _set_if_absent(dst, key, value):
    if key not in dst and value is not None:
        dst[key] = value


def normalize_config(cfg):
    """
    Normalize nested YAML style into argparse flat keys.

    Supports both:
    - flat DeiT-style keys (lr, epochs, batch_size, ...)
    - nested project-style keys (optimizer_kwargs, backbone_kwargs, aug_kwargs, ...)
    """
    cfg = deepcopy(cfg or {})
    out = dict(cfg)

    # Baseline-style dataloader keys.
    _set_if_absent(out, "batch_size", _maybe_get(cfg, "trBatch"))
    _set_if_absent(out, "workers", _maybe_get(cfg, "nworkers"))
    _set_if_absent(out, "dataset_name", _maybe_get(cfg, "train_db_name"))
    _set_if_absent(out, "eval_freq", _maybe_get(cfg, "eval_freq"))
    _set_if_absent(out, "dev_test", _maybe_get(cfg, "dev_test"))
    _set_if_absent(out, "log_initializations", _maybe_get(cfg, "log_initializations"))
    _set_if_absent(out, "deit_init_mode", _maybe_get(cfg, "deit_init_mode"))
    _set_if_absent(out, "deit_init_mode", _maybe_get(cfg, "init_mode"))
    _set_if_absent(out, "use_weight_scaling", _maybe_get(cfg, "use_weight_scaling"))
    _set_if_absent(
        out,
        "use_virtual_group_initialization",
        _maybe_get(cfg, "use_virtual_group_initialization"),
    )

    # Optimizer/scheduler sections.
    _set_if_absent(out, "opt", _maybe_get(cfg, "optimizer"))
    opt_kwargs = _maybe_get(cfg, "optimizer_kwargs") or {}
    _set_if_absent(out, "lr", _maybe_get(opt_kwargs, "lr"))
    _set_if_absent(out, "weight_decay", _maybe_get(opt_kwargs, "weight_decay"))
    _set_if_absent(out, "momentum", _maybe_get(opt_kwargs, "momentum"))
    _set_if_absent(out, "opt_betas", _maybe_get(opt_kwargs, "opt_betas"))
    _set_if_absent(out, "opt_betas", _maybe_get(opt_kwargs, "betas"))
    _set_if_absent(out, "opt_eps", _maybe_get(opt_kwargs, "opt_eps"))
    _set_if_absent(out, "opt_eps", _maybe_get(opt_kwargs, "eps"))
    _set_if_absent(out, "nesterov", _maybe_get(opt_kwargs, "nesterov"))

    sched_kwargs = _maybe_get(cfg, "scheduler_kwargs") or {}
    _set_if_absent(out, "min_lr", _maybe_get(sched_kwargs, "min_lr"))
    _set_if_absent(out, "warmup_epochs", _maybe_get(sched_kwargs, "warmup_epochs"))
    _set_if_absent(out, "unscale_lr", _maybe_get(sched_kwargs, "unscale_lr"))

    # Backbone/model section.
    bn_kwargs = _maybe_get(cfg, "backbone_kwargs") or {}
    _set_if_absent(out, "model_name", _maybe_get(bn_kwargs, "model_name"))
    _set_if_absent(out, "patch_size", _maybe_get(bn_kwargs, "patch_size"))
    _set_if_absent(out, "in_chans", _maybe_get(bn_kwargs, "in_chans"))
    _set_if_absent(out, "embed_dim", _maybe_get(bn_kwargs, "embed_dim"))
    _set_if_absent(out, "depth", _maybe_get(bn_kwargs, "depth"))
    _set_if_absent(out, "num_heads", _maybe_get(bn_kwargs, "num_heads"))
    _set_if_absent(out, "mlp_ratio", _maybe_get(bn_kwargs, "mlp_ratio"))
    _set_if_absent(out, "qkv_bias", _maybe_get(bn_kwargs, "qkv_bias"))
    _set_if_absent(out, "nb_classes", _maybe_get(bn_kwargs, "num_classes"))
    _set_if_absent(out, "drop", _maybe_get(bn_kwargs, "drop_rate"))
    _set_if_absent(out, "attn_drop_rate", _maybe_get(bn_kwargs, "attn_drop_rate"))
    _set_if_absent(out, "drop_path", _maybe_get(bn_kwargs, "drop_path_rate"))
    _set_if_absent(out, "random_init", _maybe_get(bn_kwargs, "random_init"))
    _set_if_absent(out, "deit_init_mode", _maybe_get(bn_kwargs, "deit_init_mode"))
    _set_if_absent(out, "deit_init_mode", _maybe_get(bn_kwargs, "init_mode"))
    _set_if_absent(out, "use_weight_scaling", _maybe_get(bn_kwargs, "use_weight_scaling"))
    _set_if_absent(
        out,
        "use_virtual_group_initialization",
        _maybe_get(bn_kwargs, "use_virtual_group_initialization"),
    )
    _set_if_absent(out, "pos_embed_interp", _maybe_get(bn_kwargs, "pos_embed_interp"))
    _set_if_absent(out, "align_corners", _maybe_get(bn_kwargs, "align_corners"))
    _set_if_absent(out, "moe_mlp_ratio", _maybe_get(bn_kwargs, "moe_mlp_ratio"))
    _set_if_absent(out, "moe_experts", _maybe_get(bn_kwargs, "moe_experts"))
    _set_if_absent(out, "moe_top_k", _maybe_get(bn_kwargs, "moe_top_k"))
    _set_if_absent(out, "moe_gate_type", _maybe_get(bn_kwargs, "moe_gate_type"))
    _set_if_absent(out, "vmoe_noisy_std", _maybe_get(bn_kwargs, "vmoe_noisy_std"))
    _set_if_absent(out, "gate_dim", _maybe_get(bn_kwargs, "gate_dim"))
    _set_if_absent(out, "gate_task_specific_dim", _maybe_get(bn_kwargs, "gate_task_specific_dim"))
    _set_if_absent(out, "multi_gate", _maybe_get(bn_kwargs, "multi_gate"))
    _set_if_absent(out, "moe_data_distributed", _maybe_get(cfg, "moe_data_distributed"))
    _set_if_absent(out, "moe_data_distributed", _maybe_get(bn_kwargs, "moe_data_distributed"))
    _set_if_absent(out, "use_checkpointing", _maybe_get(bn_kwargs, "use_checkpointing"))
    _set_if_absent(out, "distilled", _maybe_get(bn_kwargs, "distilled"))

    # img_size can be int or [H, W]. For pretraining we use a square crop.
    img_size = _maybe_get(bn_kwargs, "img_size")
    if "input_size" not in out and img_size is not None:
        if isinstance(img_size, (list, tuple)):
            out["input_size"] = int(img_size[0])
        else:
            out["input_size"] = int(img_size)

    # Augmentation section.
    aug_kwargs = _maybe_get(cfg, "aug_kwargs") or {}
    _set_if_absent(out, "color_jitter", _maybe_get(aug_kwargs, "color_jitter"))
    _set_if_absent(out, "aa", _maybe_get(aug_kwargs, "aa"))
    _set_if_absent(out, "train_interpolation", _maybe_get(aug_kwargs, "train_interpolation"))
    _set_if_absent(out, "reprob", _maybe_get(aug_kwargs, "reprob"))
    _set_if_absent(out, "remode", _maybe_get(aug_kwargs, "remode"))
    _set_if_absent(out, "recount", _maybe_get(aug_kwargs, "recount"))
    _set_if_absent(out, "mixup", _maybe_get(aug_kwargs, "mixup"))
    _set_if_absent(out, "cutmix", _maybe_get(aug_kwargs, "cutmix"))
    _set_if_absent(out, "cutmix_minmax", _maybe_get(aug_kwargs, "cutmix_minmax"))
    _set_if_absent(out, "mixup_prob", _maybe_get(aug_kwargs, "mixup_prob"))
    _set_if_absent(out, "mixup_switch_prob", _maybe_get(aug_kwargs, "mixup_switch_prob"))
    _set_if_absent(out, "mixup_mode", _maybe_get(aug_kwargs, "mixup_mode"))
    _set_if_absent(out, "smoothing", _maybe_get(aug_kwargs, "smoothing"))
    _set_if_absent(out, "repeated_aug", _maybe_get(aug_kwargs, "repeated_aug"))

    # Distillation section.
    dist_kwargs = _maybe_get(cfg, "distillation_kwargs") or {}
    _set_if_absent(out, "distillation_type", _maybe_get(dist_kwargs, "distillation_type"))
    _set_if_absent(out, "teacher_model", _maybe_get(dist_kwargs, "teacher_model"))
    _set_if_absent(out, "teacher_path", _maybe_get(dist_kwargs, "teacher_path"))
    _set_if_absent(out, "distillation_alpha", _maybe_get(dist_kwargs, "distillation_alpha"))
    _set_if_absent(out, "distillation_tau", _maybe_get(dist_kwargs, "distillation_tau"))

    # Runtime aliases.
    _set_if_absent(out, "pin_mem", _maybe_get(cfg, "pin_memory"))
    _set_if_absent(out, "fmoe_grouped_ddp", _maybe_get(cfg, "fmoe_grouped_ddp"))
    _set_if_absent(out, "find_unused_parameters", _maybe_get(cfg, "find_unused_parameters"))
    _set_if_absent(out, "model_ema", _maybe_get(cfg, "model_ema"))
    _set_if_absent(out, "model_ema_decay", _maybe_get(cfg, "model_ema_decay"))
    _set_if_absent(out, "model_ema_force_cpu", _maybe_get(cfg, "model_ema_force_cpu"))

    # Wandb section.
    wandb_kwargs = _maybe_get(cfg, "wandb_kwargs") or {}
    _set_if_absent(out, "use_wandb", _maybe_get(cfg, "use_wandb"))
    _set_if_absent(out, "use_wandb", _maybe_get(wandb_kwargs, "use_wandb"))
    _set_if_absent(out, "wandb_project", _maybe_get(cfg, "wandb_project"))
    _set_if_absent(out, "wandb_project", _maybe_get(wandb_kwargs, "project"))
    _set_if_absent(out, "wandb_entity", _maybe_get(cfg, "wandb_entity"))
    _set_if_absent(out, "wandb_entity", _maybe_get(wandb_kwargs, "entity"))
    _set_if_absent(out, "wandb_name", _maybe_get(cfg, "wandb_name"))
    _set_if_absent(out, "wandb_name", _maybe_get(wandb_kwargs, "name"))
    _set_if_absent(out, "wandb_id", _maybe_get(cfg, "wandb_id"))
    _set_if_absent(out, "wandb_id", _maybe_get(wandb_kwargs, "id"))
    _set_if_absent(out, "wandb_resume", _maybe_get(cfg, "wandb_resume"))
    _set_if_absent(out, "wandb_resume", _maybe_get(wandb_kwargs, "resume"))
    _set_if_absent(out, "wandb_mode", _maybe_get(cfg, "wandb_mode"))
    _set_if_absent(out, "wandb_mode", _maybe_get(wandb_kwargs, "mode"))
    return out


def _sync_grouped_ddp_weights(model, except_key_words):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    state_dict = model.state_dict()
    for key, item in state_dict.items():
        if any(key_word in key for key_word in except_key_words):
            continue
        if torch.is_tensor(item):
            torch.distributed.broadcast(item, 0)
    model.load_state_dict(state_dict)


def _resolve_fmoe_grouped_ddp_cls():
    if fmoe is None:
        return None
    cls = getattr(fmoe, "DistributedGroupedDataParallel", None)
    if cls is not None:
        return cls
    try:
        from fmoe.distributed import DistributedGroupedDataParallel as cls  # type: ignore
        return cls
    except Exception:
        return None


def _resolve_ddp_find_unused_parameters(args):
    explicit = getattr(args, "find_unused_parameters", None)
    if explicit is not None:
        return bool(explicit)

    # Auto mode:
    # - distilled=True + distillation_type=none can leave distillation head unused.
    # - otherwise default to False for better performance.
    distill_type = str(getattr(args, "distillation_type", "none") or "none").strip().lower()
    distilled = bool(getattr(args, "distilled", False))
    if distilled and distill_type == "none":
        return True
    return False


def _resolve_env_config_path(config_path):
    if not config_path:
        return ""
    if config_path == "configs/path_env.yml" and not os.path.exists(config_path) and os.path.exists("configs/env.yml"):
        return "configs/env.yml"
    return config_path


def _wandb_run_meta_path(output_dir):
    return os.path.join(output_dir, "wandb_run_meta.json")


def _load_wandb_run_meta(output_dir):
    meta_path = _wandb_run_meta_path(output_dir)
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_wandb_run_meta(output_dir, run):
    if run is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "id": getattr(run, "id", ""),
        "name": getattr(run, "name", ""),
        "project": getattr(run, "project", ""),
        "entity": getattr(run, "entity", ""),
        "url": getattr(run, "url", ""),
    }
    with open(_wandb_run_meta_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _resolve_data_path_from_env(args):
    env_cfg_path = _resolve_env_config_path(getattr(args, "config_path", ""))
    if not env_cfg_path or not os.path.exists(env_cfg_path):
        return

    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f) or {}

    hf_token = str(env_cfg.get("huggingface_access_token", "") or "").strip()
    if hf_token:
        hf_token = os.path.expandvars(os.path.expanduser(hf_token)).strip()
        # Keep placeholders like ${HF_TOKEN} untouched if unresolved.
        if hf_token and "$" not in hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    if getattr(args, "data_path", ""):
        return

    dataset_roots = env_cfg.get("dataset_roots", {}) or {}
    dataset_name = (
        getattr(args, "dataset_name", None)
        or getattr(args, "train_db_name", None)
        or "ImageNet1K"
    )
    data_path = dataset_roots.get(dataset_name, "")
    if data_path:
        args.data_path = os.path.expandvars(os.path.expanduser(str(data_path)))


def _resolve_deit_init_mode(args):
    raw_mode = str(getattr(args, "deit_init_mode", "auto") or "auto").strip().lower()
    valid_modes = {"auto", "scratch", "deit_warm_start", "deit_upcycling"}
    if raw_mode not in valid_modes:
        raise ValueError(f"Unsupported deit_init_mode '{raw_mode}'. Expected one of: {sorted(valid_modes)}")

    if raw_mode == "auto":
        resolved_mode = "scratch" if bool(getattr(args, "random_init", True)) else "deit_upcycling"
    else:
        resolved_mode = raw_mode

    args.deit_init_mode = resolved_mode
    args.random_init = resolved_mode == "scratch"
    if resolved_mode != "deit_upcycling":
        # These knobs only affect upcycling expert/gate initialization.
        args.use_weight_scaling = False
        args.use_virtual_group_initialization = False
    return resolved_mode


def _configure_upcycling_runtime_options(args):
    moe_data_distributed = bool(getattr(args, "moe_data_distributed", False))
    world_size = 1 if moe_data_distributed else int(getattr(args, "world_size", 1) or 1)
    if world_size < 1:
        world_size = 1
    tot_experts = int(getattr(args, "moe_experts", 0) or 0)
    local_experts = None
    if tot_experts > 0 and tot_experts % world_size == 0:
        local_experts = tot_experts // world_size
    elif tot_experts > 0 and world_size == 1:
        local_experts = tot_experts

    args.moe_world_size = int(world_size)
    if local_experts is not None:
        args.moe_experts_local = int(local_experts)

    set_upcycling_runtime_options(
        {
            "deit_init_mode": str(getattr(args, "deit_init_mode", "deit_upcycling")),
            "use_virtual_group_initialization": bool(
                getattr(args, "use_virtual_group_initialization", False)
            ),
            "use_weight_scaling": bool(getattr(args, "use_weight_scaling", False)),
            "moe_top_k": int(getattr(args, "moe_top_k", 0) or 0),
            "moe_experts_local": int(local_experts) if local_experts is not None else None,
            "moe_world_size": int(world_size),
            "tot_experts": int(tot_experts) if tot_experts > 0 else None,
            "moe_experts": int(tot_experts) if tot_experts > 0 else None,
        }
    )


def get_args_parser():
    parser = argparse.ArgumentParser("MoE ViT ImageNet pretrain", add_help=False)

    # config / io
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--config-path", "--config-env", dest="config_path", default="configs/path_env.yml", type=str)
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--output-dir", default="./output/pretrain", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--save-freq", default=10, type=int)
    parser.add_argument("--eval-freq", default=10, type=int)
    parser.add_argument("--dev-test", default=False, type=str2bool)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset-name", default="ImageNet1K", type=str)

    # model
    parser.add_argument("--model", default="moe_vit_small", type=str)
    parser.add_argument("--model-name", default="", type=str)
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

    # MoE
    parser.add_argument("--moe-experts", default=16, type=int)
    parser.add_argument("--moe-top-k", default=4, type=int)
    parser.add_argument("--moe-mlp-ratio", default=4.0, type=float)
    parser.add_argument("--moe-gate-type", default="noisy_vmoe", type=str)
    parser.add_argument("--vmoe-noisy-std", default=1.0, type=float)
    parser.add_argument("--gate-dim", default=-1, type=int)
    parser.add_argument("--gate-task-specific-dim", default=-1, type=int)
    parser.add_argument("--multi-gate", default=False, type=str2bool)
    parser.add_argument("--moe-data-distributed", default=False, type=str2bool)
    parser.add_argument("--moe-cv-weight", default=0.01, type=float)
    parser.add_argument("--use-checkpointing", default=True, type=str2bool)

    # init behavior
    parser.add_argument("--random-init", default=True, type=str2bool)
    parser.add_argument(
        "--deit-init-mode",
        "--init-mode",
        dest="deit_init_mode",
        default="auto",
        type=str,
        choices=["auto", "scratch", "deit_warm_start", "deit_upcycling"],
    )
    parser.add_argument("--use-weight-scaling", default=False, type=str2bool)
    parser.add_argument("--use-virtual-group-initialization", default=False, type=str2bool)
    parser.add_argument("--pos-embed-interp", default=False, type=str2bool)
    parser.add_argument("--align-corners", default=False, type=str2bool)

    # optimization
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

    # augment
    parser.add_argument("--color-jitter", default=0.3, type=float)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--train-interpolation", default="bicubic", type=str)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)

    parser.add_argument("--mixup", default=0.8, type=float)
    parser.add_argument("--cutmix", default=1.0, type=float)
    parser.add_argument("--cutmix-minmax", default=None, type=float, nargs="+")
    parser.add_argument("--mixup-prob", default=1.0, type=float)
    parser.add_argument("--mixup-switch-prob", default=0.5, type=float)
    parser.add_argument("--mixup-mode", default="batch", type=str)
    parser.add_argument("--smoothing", default=0.1, type=float)

    # distillation (scaffold)
    parser.add_argument("--distillation-type", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--teacher-model", default="regnety_160", type=str)
    parser.add_argument(
        "--teacher-path",
        default="https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth",
        type=str,
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=1.0, type=float)

    # runtime
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--pin-mem", action="store_true", dest="pin_mem")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.add_argument("--amp", default=True, type=str2bool)
    parser.add_argument("--fmoe-grouped-ddp", action="store_true", dest="fmoe_grouped_ddp")
    parser.add_argument("--no-fmoe-grouped-ddp", action="store_false", dest="fmoe_grouped_ddp")
    parser.add_argument(
        "--find-unused-parameters",
        default=None,
        type=str2bool_or_auto,
        help="DDP find_unused_parameters (true/false/auto). auto enables only for distilled=true + distillation_type=none.",
    )
    parser.add_argument("--model-ema", action="store_true", dest="model_ema")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.add_argument("--model-ema-decay", default=0.99996, type=float)
    parser.add_argument("--model-ema-force-cpu", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--log-initializations", default=True, type=str2bool)
    parser.add_argument("--use-wandb", action="store_true", dest="use_wandb")
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb-project", default="m3vit-pretrain", type=str)
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-name", default="", type=str)
    parser.add_argument("--wandb-id", default="", type=str)
    parser.add_argument(
        "--wandb-resume",
        default="auto",
        choices=["auto", "allow", "must", "never"],
        help="W&B resume policy. auto: reuse run id from output_dir when --resume is set.",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    # distributed
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--dist-eval", action="store_true", dest="dist_eval")
    parser.add_argument("--no-dist-eval", action="store_false", dest="dist_eval")

    parser.add_argument("--repeated-aug", action="store_true", dest="repeated_aug")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")

    parser.set_defaults(
        pin_mem=True,
        dist_eval=False,
        repeated_aug=True,
        fmoe_grouped_ddp=True,
        model_ema=True,
        model_ema_force_cpu=False,
        use_wandb=False,
    )
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", type=str)
    config_parser.add_argument("--config-path", "--config-env", dest="config_path", default="configs/path_env.yml", type=str)
    cfg_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        "MoE ViT ImageNet pretrain",
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

    if "model" in checkpoint:
        teacher_state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        teacher_state = checkpoint["state_dict"]
    else:
        teacher_state = checkpoint

    teacher.load_state_dict(teacher_state, strict=False)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _dump_run_metadata(args):
    if not is_main_process():
        return

    args_dict = {k: v for k, v in vars(args).items()}
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "args": args_dict,
    }
    out_path = os.path.join(args.output_dir, "run_args.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved run args to {out_path}")
    print("Run arguments:")
    print(json.dumps(args_dict, indent=2, sort_keys=True))


def _resolve_output_dir(args):
    base_output_dir = os.path.expandvars(os.path.expanduser(str(args.output_dir)))
    if getattr(args, "resume", ""):
        args.output_dir = base_output_dir
        return

    leaf = os.path.basename(os.path.normpath(base_output_dir))
    already_timestamped = (
        len(leaf) == 9
        and leaf[4] == "_"
        and leaf[:4].isdigit()
        and leaf[5:].isdigit()
    )
    if already_timestamped:
        args.output_dir = base_output_dir
        return

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if (
        getattr(args, "distributed", False)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        shared = [timestamp if is_main_process() else None]
        torch.distributed.broadcast_object_list(shared, src=0)
        timestamp = str(shared[0])

    args.output_dir = os.path.join(base_output_dir, timestamp)


def _patch_model_initialization_logging(args):
    if not bool(getattr(args, "log_initializations", True)):
        return [], {}

    module_names = [
        # Required by pretrain MoE path.
        "models.ckpt_vision_transformer_moe",
        "models.ckpt_custom_moe_layer",
        "models.gate_funs.ckpt_noisy_gate_vmoe",
        "models.gate_funs.noisy_gate",
        "pretrain.models.moe_vit_cls",
        # Keep train_fastmoe parity where possible (optional by environment).
        "models.gate_funs.token_noisy_gate_vmoe",
        "models.token_custom_moe_layer",
        "models.token_vision_transformer_moe",
        "models.token_vit_up_head",
        "models.vit_up_head",
        "models.models",
    ]

    modules_to_patch = []
    for module_name in module_names:
        try:
            modules_to_patch.append(importlib.import_module(module_name))
        except Exception as e:
            if is_main_process():
                print(f"[InitLog] Skip module '{module_name}': {e}")

    if not modules_to_patch:
        if is_main_process():
            print("[InitLog] No modules available for patching.")
        return [], {}

    try:
        original_inits = patch_and_log_initializations(modules_to_patch, args)
        if is_main_process():
            print(f"[InitLog] Patched {len(modules_to_patch)} modules for initialization logging.")
        return modules_to_patch, original_inits
    except Exception as e:
        if is_main_process():
            print(f"[InitLog] Failed to patch initialization logging: {e}")
        return [], {}


def _build_wandb_logger(args):
    if not bool(getattr(args, "use_wandb", False)):
        return None
    if not is_main_process():
        return None

    wandb_mode = str(getattr(args, "wandb_mode", "online") or "online").strip().lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported --wandb-mode '{wandb_mode}'. Use one of: online/offline/disabled")
    if wandb_mode == "disabled":
        return None

    try:
        import wandb  # type: ignore
        from utils.wandb_logger import WandbLogger, set_wandb_logger
    except Exception as exc:
        raise ImportError(
            "W&B logging requires wandb. Install with `pip install wandb` "
            "or disable with `--no-wandb`."
        ) from exc

    os.environ["WANDB_MODE"] = wandb_mode
    run_name = str(getattr(args, "wandb_name", "") or "").strip()
    if not run_name:
        run_name = os.path.basename(os.path.normpath(args.output_dir))

    run_id = str(getattr(args, "wandb_id", "") or "").strip()
    if not run_id:
        run_id = str(os.environ.get("WANDB_RUN_ID", "") or "").strip()
    resume_policy = str(getattr(args, "wandb_resume", "auto") or "auto").strip().lower()
    if resume_policy not in {"auto", "allow", "must", "never"}:
        raise ValueError(
            f"Unsupported --wandb-resume '{resume_policy}'. "
            "Use one of: auto/allow/must/never"
        )

    if (not run_id) and bool(getattr(args, "resume", "")) and resume_policy != "never":
        meta = _load_wandb_run_meta(args.output_dir)
        run_id = str(meta.get("id", "") or "").strip()
        if run_id:
            print(f"W&B resume: found existing run id '{run_id}' in {args.output_dir}")

    if resume_policy == "auto":
        if run_id:
            resume_mode = "must" if bool(getattr(args, "resume", "")) else "allow"
        else:
            resume_mode = None
    else:
        resume_mode = resume_policy

    if resume_mode == "must" and not run_id:
        raise ValueError("--wandb-resume must requires --wandb-id or existing wandb_run_meta.json")

    wandb_logger = WandbLogger(enabled=True)
    wandb_logger.init(
        project=str(getattr(args, "wandb_project", "m3vit-pretrain") or "m3vit-pretrain"),
        entity=getattr(args, "wandb_entity", None),
        name=run_name,
        config=vars(args),
        resume=resume_mode,
        id=run_id or None,
    )
    set_wandb_logger(wandb_logger)
    if wandb_logger.run is not None:
        wandb_logger.run.config.update({"output_dir": args.output_dir}, allow_val_change=True)
        _save_wandb_run_meta(args.output_dir, wandb_logger.run)
        args.wandb_id = str(getattr(wandb_logger.run, "id", "") or "")

    for path in (
        args.config_path,
        args.config,
        os.path.join(args.output_dir, "run_args.json"),
        _wandb_run_meta_path(args.output_dir),
    ):
        if path and os.path.exists(path):
            wandb.save(path)
    print(
        f"W&B enabled: project={args.wandb_project}, run={run_name}, mode={wandb_mode}, "
        f"resume={resume_mode}, id={run_id or args.wandb_id or 'new'}"
    )
    return wandb_logger


def main():
    args = parse_args()
    if int(args.eval_freq) < 1:
        raise ValueError(f"--eval-freq must be >= 1, got {args.eval_freq}")
    init_distributed_mode(args)
    _resolve_data_path_from_env(args)
    _resolve_deit_init_mode(args)
    if bool(getattr(args, "moe_data_distributed", False)) and bool(getattr(args, "fmoe_grouped_ddp", False)):
        if is_main_process():
            print("moe_data_distributed=True: forcing --no-fmoe-grouped-ddp for replicated experts.")
        args.fmoe_grouped_ddp = False
    _configure_upcycling_runtime_options(args)

    if is_main_process():
        expert_upcycling = args.deit_init_mode in {"deit_warm_start", "deit_upcycling"}
        print(
            "DeiT init mode:",
            args.deit_init_mode,
            f"(random_init={args.random_init}, expert_upcycling={expert_upcycling}, "
            f"gate_random_init={args.deit_init_mode == 'deit_warm_start'})",
        )

    if args.data_path == "":
        raise ValueError(
            "--data-path is required (or set dataset_roots.ImageNet1K in "
            f"{args.config_path} and use --dataset-name ImageNet1K)"
        )
    if args.distillation_type != "none" and not args.distilled:
        raise ValueError("--distilled must be true when --distillation-type is soft/hard")

    if args.unscale_lr:
        effective_lr = args.lr
    else:
        global_batch = args.batch_size * args.world_size
        effective_lr = args.lr * global_batch / 512.0

    args.lr = effective_lr

    device = torch.device(args.device)
    set_seed(args.seed, rank=args.rank)
    cudnn.benchmark = True

    _resolve_output_dir(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process():
        print(f"Resolved output dir: {args.output_dir}")
    _dump_run_metadata(args)
    wandb_logger = _build_wandb_logger(args)

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

    modules_to_patch, original_inits = _patch_model_initialization_logging(args)
    try:
        model = build_model(args)
    finally:
        if modules_to_patch and original_inits:
            restore_original_initializations(modules_to_patch, original_inits)
    model.to(device)
    model_ema = None
    if args.model_ema:
        if TimmModelEma is None:
            raise ImportError("timm.utils.ModelEma or ModelEmaV2 is required when --model-ema is enabled")
        model_ema = TimmModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )

    if args.distributed:
        ddp_find_unused = _resolve_ddp_find_unused_parameters(args)
        if is_main_process():
            print(
                "DDP find_unused_parameters:",
                ddp_find_unused,
                f"(distilled={bool(getattr(args, 'distilled', False))}, "
                f"distillation_type={getattr(args, 'distillation_type', 'none')})",
            )
        if args.fmoe_grouped_ddp and (not bool(getattr(args, "moe_data_distributed", False))):
            grouped_ddp_cls = _resolve_fmoe_grouped_ddp_cls()
            if grouped_ddp_cls is None:
                raise ImportError(
                    "fmoe grouped DDP is unavailable but --fmoe-grouped-ddp is enabled. "
                    "Install a compatible fmoe version or pass --no-fmoe-grouped-ddp."
                )
            print("Using fmoe.DistributedGroupedDataParallel")
            model = grouped_ddp_cls(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=ddp_find_unused,
            )
            _sync_grouped_ddp_weights(
                model,
                except_key_words=("mlp.experts.h4toh", "mlp.experts.htoh4"),
            )
        else:
            print("Using torch.nn.parallel.DistributedDataParallel")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=ddp_find_unused,
            )
        model_without_ddp = model.module if hasattr(model, "module") else model
    else:
        model_without_ddp = model

    optimizer = build_optimizer(args, model_without_ddp)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_model = build_teacher_model(args, device)
    criterion = build_criterion(args, teacher_model=teacher_model)

    if args.distillation_type != "none":
        print(
            "Distillation is enabled. Ensure model returns distillation logits "
            "(tuple(logits, logits_dist)) for full DeiT behavior."
        )

    start_epoch, best_acc1 = auto_resume(
        args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        model_ema=model_ema,
    )

    if args.eval:
        test_stats = evaluate(loader_val, model, device, args)
        if wandb_logger is not None:
            wandb_logger.log({f"eval/{k}": v for k, v in test_stats.items()})
            wandb_logger.finish()
        print(f"Eval only: Acc@1={test_stats['acc1']:.3f} Acc@5={test_stats['acc5']:.3f}")
        return

    if args.dev_test:
        if is_main_process():
            print("Run one pre-flight validation (--dev-test=true)")
        dev_stats = evaluate(loader_val, model, device, args)
        if wandb_logger is not None:
            wandb_logger.log({f"preflight/{k}": v for k, v in dev_stats.items()})

    print("Start training")
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
            model_ema=model_ema,
            wandb_logger=wandb_logger,
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

        save_checkpoint(
            args=args,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_acc1=best_acc1,
            is_best=is_best,
            model_ema=model_ema,
        )

        if is_main_process():
            log_stats = {
                "epoch": epoch,
                "n_parameters": sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad),
                "best_acc1": best_acc1,
                "eval_performed": bool(do_eval),
            }
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            if do_eval:
                log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            train_brief_keys = [
                "loss",
                "loss_no_cv",
                "loss_base",
                "loss_distill",
                "cv_loss",
                "cv_loss_weighted",
                "lr",
            ]
            train_brief = " ".join(
                f"{k}={float(train_stats[k]):.4f}" for k in train_brief_keys if k in train_stats
            )
            if do_eval:
                print(
                    f"[Epoch {epoch + 1}] {train_brief} "
                    f"acc1={float(test_stats.get('acc1', 0.0)):.3f} "
                    f"acc5={float(test_stats.get('acc5', 0.0)):.3f} "
                    f"val_loss={float(test_stats.get('loss', 0.0)):.4f} "
                    f"best_acc1={best_acc1:.3f}"
                )
            else:
                print(f"[Epoch {epoch + 1}] {train_brief} best_acc1={best_acc1:.3f} (eval skipped)")
            if wandb_logger is not None:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "best_acc1": float(best_acc1),
                    "n_parameters": log_stats["n_parameters"],
                    "eval_performed": bool(do_eval),
                }
                wandb_metrics.update({f"train/{k}": v for k, v in train_stats.items()})
                if do_eval:
                    wandb_metrics.update({f"eval/{k}": v for k, v in test_stats.items()})
                wandb_logger.log(wandb_metrics)

    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"Training time {total_time_str}")
    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
