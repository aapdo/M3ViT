#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DATA_PATH=${DATA_PATH:-/path/to/imagenet}
OUT_DIR=${OUT_DIR:-/tmp/moe_pretrain}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" pretrain/train.py \
    --config pretrain/configs/deit_moe_small.yaml \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUT_DIR}"
