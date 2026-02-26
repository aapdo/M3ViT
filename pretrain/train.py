"""ImageNet pretrain entrypoint for MoE ViT (DeiT-inspired skeleton)."""

import argparse
import json
import os
import time
from copy import deepcopy

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
    _set_if_absent(out, "model_ema", _maybe_get(cfg, "model_ema"))
    _set_if_absent(out, "model_ema_decay", _maybe_get(cfg, "model_ema_decay"))
    _set_if_absent(out, "model_ema_force_cpu", _maybe_get(cfg, "model_ema_force_cpu"))
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


def _resolve_env_config_path(config_path):
    if not config_path:
        return ""
    if config_path == "configs/path_env.yml" and not os.path.exists(config_path) and os.path.exists("configs/env.yml"):
        return "configs/env.yml"
    return config_path


def _resolve_data_path_from_env(args):
    if getattr(args, "data_path", ""):
        return

    env_cfg_path = _resolve_env_config_path(getattr(args, "config_path", ""))
    if not env_cfg_path or not os.path.exists(env_cfg_path):
        return

    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f) or {}

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
    world_size = int(getattr(args, "world_size", 1) or 1)
    if world_size < 1:
        world_size = 1
    tot_experts = int(getattr(args, "moe_experts", 0) or 0)
    local_experts = None
    if tot_experts > 0 and tot_experts % world_size == 0:
        local_experts = tot_experts // world_size

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
    parser.add_argument("--moe-cv-weight", default=0.01, type=float)

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
    parser.add_argument("--model-ema", action="store_true", dest="model_ema")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.add_argument("--model-ema-decay", default=0.99996, type=float)
    parser.add_argument("--model-ema-force-cpu", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)

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


def main():
    args = parse_args()
    init_distributed_mode(args)
    _resolve_data_path_from_env(args)
    _resolve_deit_init_mode(args)
    _configure_upcycling_runtime_options(args)

    if is_main_process():
        print(
            "DeiT init mode:",
            args.deit_init_mode,
            f"(random_init={args.random_init}, upcycling={args.deit_init_mode == 'deit_upcycling'})",
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

    os.makedirs(args.output_dir, exist_ok=True)

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

    model = build_model(args)
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
        if args.fmoe_grouped_ddp:
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
                find_unused_parameters=True,
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
                find_unused_parameters=True,
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
        print(f"Eval only: Acc@1={test_stats['acc1']:.3f} Acc@5={test_stats['acc5']:.3f}")
        return

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
        )

        scheduler.step(epoch + 1)
        test_stats = evaluate(loader_val, model, device, args)
        acc1 = float(test_stats.get("acc1", 0.0))

        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1

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
            }
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    main()
