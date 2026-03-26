#!/bin/bash
# 현재 학습이 끝날 때까지 대기 후 실행
echo "Waiting for current training to finish..."
while pgrep -f train_fastmoe > /dev/null; do
    sleep 60
done
echo "GPU free. Starting std=0 experiment..."

cd /home/jy/m3vit

COMMON="--moe_gate_type noisy_vmoe \
    --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
    --moe_data_distributed \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --task_one_hot False \
    --multi_gate True \
    --save_dir output_dir \
    --use_checkpointing True \
    --trBatch 2 \
    --gate_task_specific_dim -1 \
    --use_wandb \
    --wandb_project m3vit_pretrained_nyud"

R2A3="output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth"

torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    $COMMON --pretrained $R2A3 \
    --vmoe_noisy_std 0 \
    --wandb_name nyud_r2a3_std0_ckpt
