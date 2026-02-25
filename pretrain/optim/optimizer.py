"""Optimizer builder inspired by DeiT/timm defaults."""

import torch


def _param_groups_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def build_optimizer(args, model):
    skip = set()
    if hasattr(model, "no_weight_decay"):
        no_wd = model.no_weight_decay
        if callable(no_wd):
            no_wd = no_wd()
        if no_wd is not None:
            skip.update(no_wd)

    param_groups = _param_groups_weight_decay(model, args.weight_decay, skip)
    opt_name = args.opt.lower()

    if opt_name == "adamw":
        betas = tuple(args.opt_betas) if args.opt_betas is not None else (0.9, 0.999)
        return torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=betas,
            eps=args.opt_eps,
        )

    if opt_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
        )

    raise ValueError(f"Unsupported optimizer: {args.opt}")
