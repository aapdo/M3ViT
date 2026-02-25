"""Checkpoint utilities for pretrain."""

import os

import torch

from .dist import get_world_size, is_main_process, save_on_master
from .moe_checkpoint import (
    build_mtl_meta,
    gather_global_expert_state_dict,
    to_mtl_backbone_state_dict,
)


def save_checkpoint(args, epoch, model, optimizer, scheduler, scaler, best_acc1, is_best=False):
    model_to_save = model.module if hasattr(model, "module") else model
    local_model_state = model_to_save.state_dict()

    # Build MTL-compatible global-expert checkpoint payload.
    mtl_state_local, _ = to_mtl_backbone_state_dict(local_model_state)
    mtl_state_global = gather_global_expert_state_dict(mtl_state_local)

    world_size = get_world_size()
    moe_experts_global = getattr(args, "moe_experts", None)
    moe_experts_local = None
    if moe_experts_global is not None and world_size > 0:
        moe_experts_local = int(moe_experts_global) // int(world_size)

    mtl_payload = {
        "state_dict": mtl_state_global,
        "meta": build_mtl_meta(
            state_dict=mtl_state_global,
            source="pretrain",
            world_size=world_size,
            moe_experts_global=moe_experts_global,
            moe_experts_local=moe_experts_local,
        ),
    }

    if not is_main_process():
        return

    os.makedirs(args.output_dir, exist_ok=True)

    state = {
        "model": local_model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_acc1": best_acc1,
        "args": vars(args),
    }

    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    save_on_master(state, latest_path)
    save_on_master(mtl_payload, os.path.join(args.output_dir, "mtl_latest_global.pth"))

    if (epoch + 1) % args.save_freq == 0:
        save_on_master(state, os.path.join(args.output_dir, f"checkpoint_{epoch + 1:04d}.pth"))

    if is_best:
        save_on_master(state, os.path.join(args.output_dir, "checkpoint_best.pth"))
        save_on_master(mtl_payload, os.path.join(args.output_dir, "mtl_best_global.pth"))


def auto_resume(args, model, optimizer=None, scheduler=None, scaler=None):
    if not args.resume:
        return 0, 0.0

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    checkpoint = torch.load(args.resume, map_location="cpu")
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model"], strict=False)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc1 = float(checkpoint.get("best_acc1", 0.0))

    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return start_epoch, best_acc1
