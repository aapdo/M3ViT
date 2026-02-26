"""Cosine scheduler with warmup (DeiT-style behavior)."""

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(args, optimizer):
    base_lr = float(args.lr)
    min_lr = float(args.min_lr)
    warmup_epochs = int(args.warmup_epochs)
    total_epochs = int(args.epochs)

    if base_lr <= 0:
        raise ValueError(f"lr must be > 0, got {base_lr}")

    min_lr_ratio = min_lr / base_lr

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        if total_epochs <= warmup_epochs:
            return 1.0

        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Start epoch-0 with warmup LR (instead of base LR) for DeiT-like behavior.
    if warmup_epochs > 0:
        warmup_scale = lr_lambda(0)
        for group, base_lr_group in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base_lr_group * warmup_scale
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]

    return scheduler
