"""Export or migrate checkpoints into global-expert MTL checkpoint format."""

import argparse
import os

import torch

from pretrain.utils.moe_checkpoint import (
    build_mtl_meta,
    infer_expert_format,
    load_checkpoint_state,
    merge_moe_sharded_directory,
    to_mtl_backbone_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser("Export pretrain checkpoint for MTL")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Input checkpoint file or shard directory",
    )
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--model-key",
        default=None,
        type=str,
        help="Optional checkpoint key to read model state from",
    )
    parser.add_argument(
        "--expected-moe-experts",
        default=None,
        type=int,
        help="Optional expected global expert count for validation",
    )
    parser.add_argument(
        "--expected-world-size",
        default=None,
        type=int,
        help="Optional expected world size for validation/meta",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _extract_checkpoint_hints(checkpoint):
    world_size = None
    moe_experts_global = None
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("meta", {})
        if isinstance(meta, dict):
            world_size = meta.get("world_size", world_size)
            moe_experts_global = meta.get("moe_experts_global", moe_experts_global)

        args = checkpoint.get("args", {})
        if isinstance(args, dict):
            world_size = args.get("world_size", world_size)
            moe_experts_global = args.get("moe_experts", moe_experts_global)
    return world_size, moe_experts_global


def _raise_local_single_file_error(path):
    raise ValueError(
        "Input checkpoint appears to contain rank-local experts only. "
        "A single local-only file from multi-GPU pretraining cannot be fully recovered.\n"
        f"checkpoint: {path}\n"
        "Please provide the full shard directory (e.g., containing 0.pth, 1.pth, ...) "
        "and rerun export:\n"
        "python pretrain/export_to_mtl.py --checkpoint <shard_dir> --output <mtl_global.pth>"
    )


def main():
    args = parse_args()
    input_path = args.checkpoint

    if not (os.path.isfile(input_path) or os.path.isdir(input_path)):
        raise FileNotFoundError(f"Checkpoint path not found: {input_path}")

    dropped = []
    source = "pretrain"

    if os.path.isdir(input_path):
        checkpoint, merged_state, num_shards = merge_moe_sharded_directory(input_path)
        shard_world_size = None
        if isinstance(checkpoint, dict):
            ckpt_args = checkpoint.get("args", {})
            if isinstance(ckpt_args, dict):
                shard_world_size = ckpt_args.get("world_size", None)

        if shard_world_size is not None and int(shard_world_size) != int(num_shards):
            raise ValueError(
                "Shard directory appears incomplete: "
                f"expected world_size={int(shard_world_size)} but found shards={int(num_shards)}"
            )

        state, dropped = to_mtl_backbone_state_dict(merged_state)
        source = "pretrain_sharded_dir"

        hint_world_size, hint_global_experts = _extract_checkpoint_hints(checkpoint)
        world_size = args.expected_world_size or hint_world_size or num_shards
        moe_experts_global = args.expected_moe_experts or hint_global_experts

    else:
        checkpoint, state_raw = load_checkpoint_state(
            input_path,
            map_location="cpu",
            model_key=args.model_key,
        )
        state, dropped = to_mtl_backbone_state_dict(state_raw)
        source = "pretrain_single_file"

        hint_world_size, hint_global_experts = _extract_checkpoint_hints(checkpoint)
        world_size = args.expected_world_size or hint_world_size
        moe_experts_global = args.expected_moe_experts or hint_global_experts

        inferred = infer_expert_format(
            checkpoint=checkpoint,
            state_dict=state,
            expected_global_experts=moe_experts_global,
            expected_world_size=world_size,
        )
        if inferred == "local":
            _raise_local_single_file_error(input_path)
        if inferred == "unknown":
            raise ValueError(
                "Could not verify whether this checkpoint uses global experts. "
                "Provide --expected-moe-experts/--expected-world-size, or use "
                "a checkpoint with meta.expert_format='global'."
            )

    meta = build_mtl_meta(
        state_dict=state,
        source=source,
        world_size=world_size,
        moe_experts_global=moe_experts_global,
    )
    out = {"state_dict": state, "meta": meta}
    torch.save(out, args.output)

    print(f"Saved: {args.output}")
    print(f"Params kept: {len(state)}")
    print(f"Params dropped: {len(dropped)}")
    print(f"Meta: {meta}")
    if args.verbose and dropped:
        for key in dropped:
            print(f"  drop: {key}")


if __name__ == "__main__":
    main()
