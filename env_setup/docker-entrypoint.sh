#!/bin/bash
set -e

# Create dataset directories if they don't exist (as jy user)
mkdir -p /home/jy/mtl/extended_m3vit/datasets

# Create symbolic links for datasets
if [ -d "/data/multi_task_datasets/PASCAL_MT" ]; then
    ln -sf /data/multi_task_datasets/PASCAL_MT /home/jy/mtl/extended_m3vit/datasets/pascal_context
    chown -h jy:jy /home/jy/mtl/extended_m3vit/datasets/pascal_context
fi

if [ -d "/data/multi_task_datasets/hf/nyudv2" ]; then
    ln -sf /data/multi_task_datasets/hf/nyudv2 /home/jy/mtl/extended_m3vit/datasets/nyud_v2
    chown -h jy:jy /home/jy/mtl/extended_m3vit/datasets/nyud_v2
fi

tail -F /dev/null