# Checkpoint MoE (Gradient Checkpointing)

Origin MoE에 gradient checkpointing을 적용하여 GPU 메모리 효율을 높인 버전.
현재 `VisionTransformer_moe` backbone의 기본 구현.

## 파일

| 파일 | 설명 |
|------|------|
| `vision_transformer_moe.py` | Checkpointed MoE ViT backbone (`VisionTransformerMoE`) |
| `custom_moe_layer.py` | Checkpointed FMoE MLP layer (`FMoETransformerMLP`) |
| `noisy_gate_vmoe.py` | V-MoE 스타일 noisy gate (`NoisyGate_VMoE`) |

## 실행 방법

### NYUD (semseg + depth)
```bash
torchrun --nproc_per_node=2 train_fastmoe.py \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True
```

### PASCAL Context (5 tasks)
```bash
torchrun --nproc_per_node=2 train_fastmoe.py \
    --config_exp configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True
```

## 관련 Config

- `configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml`
- `configs/nyud/vit_moe/pup_moe_vit_base_multi_task_baseline.yml`
- `configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml`
- `configs/pascal/vit_moe/pup_moe_vit_base_multi_task_baseline.yml`
