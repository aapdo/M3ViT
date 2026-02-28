"""Checkpoint utilities for pretrain."""

import os
import re

import torch

from .dist import get_rank, get_world_size, is_main_process, save_on_master
from .moe_checkpoint import (
    build_mtl_meta,
    gather_global_expert_state_dict,
    to_mtl_backbone_state_dict,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_model_ema_state_dict(model_ema):
    if model_ema is None:
        return None
    if hasattr(model_ema, "module"):
        return model_ema.module.state_dict()
    if hasattr(model_ema, "ema"):
        return model_ema.ema.state_dict()
    return model_ema.state_dict()


def _load_model_ema_state_dict(model_ema, state_dict):
    if model_ema is None or state_dict is None:
        return
    if hasattr(model_ema, "module"):
        model_ema.module.load_state_dict(state_dict, strict=False)
        return
    if hasattr(model_ema, "ema"):
        model_ema.ema.load_state_dict(state_dict, strict=False)
        return
    model_ema.load_state_dict(state_dict)


_RANK_SUFFIX_RE = re.compile(r"_rank\d+$")


def _ranked_path(path, rank):
    base, ext = os.path.splitext(path)
    return f"{base}_rank{int(rank)}{ext}"


def _has_rank_suffix(path):
    base, _ = os.path.splitext(path)
    return _RANK_SUFFIX_RE.search(base) is not None


def _replace_rank_suffix(path, rank):
    base, ext = os.path.splitext(path)
    base = _RANK_SUFFIX_RE.sub("", base)
    return f"{base}_rank{int(rank)}{ext}"


def _resolve_resume_path(resume, rank, world_size):
    if os.path.isdir(resume):
        ranked_latest = os.path.join(resume, f"checkpoint_latest_rank{int(rank)}.pth")
        global_latest = os.path.join(resume, "checkpoint_latest.pth")
        if world_size > 1 and os.path.isfile(ranked_latest):
            return ranked_latest
        if os.path.isfile(global_latest):
            return global_latest
        raise FileNotFoundError(
            f"Resume directory found but no checkpoint file exists: {resume}"
        )

    if os.path.isfile(resume):
        if world_size > 1:
            ranked_candidate = (
                _replace_rank_suffix(resume, rank)
                if _has_rank_suffix(resume)
                else _ranked_path(resume, rank)
            )
            if os.path.isfile(ranked_candidate):
                return ranked_candidate
        return resume

    if world_size > 1 and not _has_rank_suffix(resume):
        ranked_candidate = _ranked_path(resume, rank)
        if os.path.isfile(ranked_candidate):
            return ranked_candidate

    raise FileNotFoundError(f"Resume checkpoint not found: {resume}")


def save_checkpoint(
    args,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_acc1,
    is_best=False,
    model_ema=None,
):
    should_save_periodic = ((epoch + 1) % int(args.save_freq) == 0)
    should_save_best = bool(is_best)
    should_save_any = should_save_periodic or should_save_best
    if not should_save_any:
        return

    model_to_save = _unwrap_model(model)
    local_model_state = model_to_save.state_dict()
    rank = get_rank()
    world_size = get_world_size()
    moe_world_size = int(
        getattr(
            args,
            "moe_world_size",
            1 if bool(getattr(args, "moe_data_distributed", False)) else world_size,
        )
        or 1
    )
    if moe_world_size < 1:
        moe_world_size = 1

    # Build MTL-compatible global-expert checkpoint payload.
    mtl_state_local, _ = to_mtl_backbone_state_dict(local_model_state)
    if moe_world_size > 1:
        mtl_state_global = gather_global_expert_state_dict(mtl_state_local)
    else:
        # Replicated-expert mode: expert tensors are already full on every rank.
        mtl_state_global = mtl_state_local

    moe_experts_global = getattr(args, "moe_experts", None)
    moe_experts_local = None
    if moe_experts_global is not None:
        if moe_world_size > 1:
            moe_experts_local = int(moe_experts_global) // int(moe_world_size)
        else:
            moe_experts_local = int(moe_experts_global)

    mtl_payload = {
        "state_dict": mtl_state_global,
        "meta": build_mtl_meta(
            state_dict=mtl_state_global,
            source="pretrain",
            world_size=moe_world_size,
            moe_experts_global=moe_experts_global,
            moe_experts_local=moe_experts_local,
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    state = {
        "model": local_model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_ema": _get_model_ema_state_dict(model_ema),
        "epoch": epoch,
        "best_acc1": best_acc1,
        "args": vars(args),
    }

    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    if world_size > 1:
        torch.save(state, _ranked_path(latest_path, rank))
    save_on_master(state, latest_path)
    save_on_master(mtl_payload, os.path.join(args.output_dir, "mtl_latest_global.pth"))

    if should_save_periodic:
        epoch_path = os.path.join(args.output_dir, f"checkpoint_{epoch + 1:04d}.pth")
        if world_size > 1:
            torch.save(state, _ranked_path(epoch_path, rank))
        save_on_master(state, epoch_path)

    if is_best:
        best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
        if world_size > 1:
            torch.save(state, _ranked_path(best_path, rank))
        save_on_master(state, best_path)
        save_on_master(mtl_payload, os.path.join(args.output_dir, "mtl_best_global.pth"))


def auto_resume(args, model, optimizer=None, scheduler=None, scaler=None, model_ema=None):
    if not args.resume:
        return 0, 0.0

    resume_path = _resolve_resume_path(
        args.resume,
        rank=getattr(args, "rank", 0),
        world_size=getattr(args, "world_size", 1),
    )
    if is_main_process():
        print(f"Resuming from: {resume_path}")

    checkpoint = torch.load(resume_path, map_location="cpu")
    model_to_load = _unwrap_model(model)
    model_to_load.load_state_dict(checkpoint["model"], strict=False)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc1 = float(checkpoint.get("best_acc1", 0.0))

    if model_ema is not None:
        if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
            _load_model_ema_state_dict(model_ema, checkpoint["model_ema"])
        else:
            _load_model_ema_state_dict(model_ema, checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return start_epoch, best_acc1
