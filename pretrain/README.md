# ImageNet Pretrain Skeleton (DeiT-style)

This directory provides a DeiT-inspired pretraining pipeline for MoE ViT encoders in this repo.

## Goals

- Keep the training loop close to DeiT structure (`train.py`, `engine`, `datasets`, `samplers`, `losses`).
- Reuse this repository's encoder implementation (`models/ckpt_vision_transformer_moe.py`).
- Export checkpoints in a format directly loadable by the existing MTL flow (`--pretrained` + `cvt_state_dict`).

## Main entrypoints

- `pretrain/train.py`: ImageNet pretrain for MoE ViT classification.
- `pretrain/export_to_mtl.py`: Convert pretrain checkpoint to MTL-compatible checkpoint.

## Checkpoint outputs

Pretraining now writes two checkpoint families:

- Resume checkpoints (rank-local expert layout):
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth`
- MTL-ready checkpoints (global expert layout):
  - `mtl_latest_global.pth`
  - `mtl_best_global.pth`

Use `mtl_*_global.pth` for downstream `--pretrained` in MTL training.

## Example

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --data-path /path/to/imagenet \
  --output-dir /path/to/output

python pretrain/export_to_mtl.py \
  --checkpoint /path/to/output/mtl_best_global.pth \
  --output /path/to/output/mtl_pretrained.pth
```

If `configs/path_env.yml` contains `dataset_roots.ImageNet1K`, you can omit `--data-path`:

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --config-path configs/path_env.yml \
  --dataset-name ImageNet1K \
  --output-dir /path/to/output
```

## DeiT Init Modes

The pretrain entrypoint supports three explicit initialization modes via
`--deit-init-mode` (alias `--init-mode`):

- `scratch`: random init pretraining (`random_init=True`)
- `deit_warm_start`: load DeiT weights, keep MoE experts random
- `deit_upcycling`: load DeiT weights and initialize MoE experts from DeiT MLP
- `auto`: compatibility mode (maps to `scratch` when `random_init=true`, else `deit_upcycling`)

Examples:

```bash
# 1) Scratch pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode scratch \
  --output-dir /path/to/output

# 2) DeiT warm-start pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode deit_warm_start \
  --output-dir /path/to/output

# 3) DeiT upcycling pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode deit_upcycling \
  --output-dir /path/to/output
```

If you have train_fastmoe-style shard directory (`0.pth`, `1.pth`, ...):

```bash
python pretrain/export_to_mtl.py \
  --checkpoint /path/to/shard_dir \
  --output /path/to/output/mtl_global.pth
```

## Notes

- Distillation flags are scaffolded but disabled by default (`--distillation-type none`).
- The default pipeline uses AdamW + warmup + cosine decay (DeiT-style pretrain setup).
- Single-file rank-local MoE checkpoints are intentionally rejected by the MTL loader.
  If a checkpoint was produced in multi-GPU mode and only one local file is available,
  it cannot be fully reconstructed; use the full shard directory export path.
