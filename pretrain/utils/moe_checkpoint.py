"""MoE checkpoint helpers for global-expert MTL compatibility."""

from collections import OrderedDict
import os

import torch
import torch.distributed as dist


EXPERT_KEYWORDS = ("mlp.experts.htoh4", "mlp.experts.h4toh")


def is_expert_key(key):
    return any(pattern in key for pattern in EXPERT_KEYWORDS)


def strip_wrapper_prefix(key):
    if key.startswith("module."):
        key = key[len("module.") :]
    return key


def to_mtl_backbone_state_dict(state_dict):
    """
    Convert model/wrapper state_dict to backbone-only key space for MTL loader.

    - strips `module.` prefix
    - if key starts with `encoder.`, strips only the `encoder.` prefix and keeps the tensor
    - drops wrapper-only params at top level (`head.*`, `norm.*`)
    """
    out_state = OrderedDict()
    dropped = []
    for key, value in state_dict.items():
        key = strip_wrapper_prefix(key)

        if key.startswith("encoder."):
            new_key = key[len("encoder.") :]
            out_state[new_key] = value
            continue

        if key.startswith("head.") or key.startswith("norm."):
            dropped.append(key)
            continue

        # Already-backbone state_dict case (e.g., MTL/global ckpt input).
        out_state[key] = value
    return out_state, dropped


def get_first_expert_dim0(state_dict):
    for key, value in state_dict.items():
        if is_expert_key(key) and torch.is_tensor(value):
            return int(value.shape[0])
    return None


def gather_global_expert_state_dict(state_dict):
    """
    Gather rank-local expert tensors to global expert tensors across all ranks.

    Non-expert parameters are copied as-is.
    """
    if not dist.is_available() or not dist.is_initialized():
        return OrderedDict(state_dict)

    world_size = dist.get_world_size()
    if world_size <= 1:
        return OrderedDict(state_dict)

    global_state = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value) and is_expert_key(key):
            gather_list = [torch.empty_like(value) for _ in range(world_size)]
            dist.all_gather(gather_list, value.contiguous())
            global_state[key] = torch.cat(gather_list, dim=0)
        else:
            global_state[key] = value
    return global_state


def build_mtl_meta(
    state_dict,
    source,
    world_size=None,
    moe_experts_global=None,
    moe_experts_local=None,
):
    if world_size is None:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

    first_dim0 = get_first_expert_dim0(state_dict)
    if moe_experts_global is None and first_dim0 is not None:
        moe_experts_global = int(first_dim0)

    if moe_experts_local is None:
        if first_dim0 is None:
            moe_experts_local = 0
        elif world_size > 0 and first_dim0 % world_size == 0:
            moe_experts_local = int(first_dim0 // world_size)
        else:
            moe_experts_local = int(first_dim0)

    return {
        "expert_format": "global",
        "moe_experts_global": int(moe_experts_global) if moe_experts_global is not None else 0,
        "moe_experts_local": int(moe_experts_local),
        "world_size": int(world_size),
        "source": str(source),
    }


def load_checkpoint_state(path, map_location="cpu", model_key=None):
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict, got {type(checkpoint)}")

    if model_key is not None and model_key in checkpoint:
        state = checkpoint[model_key]
        return checkpoint, state

    if "state_dict" in checkpoint:
        return checkpoint, checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint, checkpoint["model"]
    return checkpoint, checkpoint


def infer_expert_format(
    checkpoint,
    state_dict,
    expected_global_experts=None,
    expected_world_size=None,
):
    """
    Infer expert format for a single checkpoint state_dict.

    Returns one of: 'global', 'local', 'dense', 'unknown'
    """
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("meta", {})
        if isinstance(meta, dict):
            meta_format = meta.get("expert_format", None)
            if meta_format in {"global", "local"}:
                return meta_format
    else:
        meta = {}

    dim0 = get_first_expert_dim0(state_dict)
    if dim0 is None:
        return "dense"

    if expected_global_experts is None and isinstance(checkpoint, dict):
        args = checkpoint.get("args", {})
        if isinstance(args, dict):
            expected_global_experts = args.get("moe_experts", None)
            if expected_world_size is None:
                expected_world_size = args.get("world_size", None)

    if expected_global_experts is not None:
        expected_global_experts = int(expected_global_experts)
        if dim0 == expected_global_experts:
            return "global"
        if expected_world_size is not None:
            expected_world_size = int(expected_world_size)
            if expected_world_size > 1 and dim0 * expected_world_size == expected_global_experts:
                return "local"

    return "unknown"


def _sorted_rank_shard_files(shard_dir):
    files = []
    for name in os.listdir(shard_dir):
        if not name.endswith(".pth"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            files.append((int(stem), os.path.join(shard_dir, name)))
    files.sort(key=lambda x: x[0])
    return files


def merge_moe_sharded_directory(shard_dir):
    """
    Merge train_fastmoe-style rank shard directory into one checkpoint state_dict.
    """
    shard_files = _sorted_rank_shard_files(shard_dir)
    if not shard_files:
        raise ValueError(f"No rank shard '*.pth' files found in: {shard_dir}")
    if shard_files[0][0] != 0:
        raise ValueError("Shard directory must contain rank-0 checkpoint file '0.pth'")

    base_ckpt, base_state = load_checkpoint_state(shard_files[0][1], map_location="cpu")
    merged_state = OrderedDict(base_state)

    for rank, shard_path in shard_files[1:]:
        _ = rank
        shard_ckpt, shard_state = load_checkpoint_state(shard_path, map_location="cpu")
        _ = shard_ckpt
        for key, value in shard_state.items():
            if is_expert_key(key):
                if key in merged_state:
                    merged_state[key] = torch.cat([merged_state[key], value], dim=0)
                else:
                    merged_state[key] = value
            elif key not in merged_state:
                merged_state[key] = value

    return base_ckpt, merged_state, len(shard_files)
