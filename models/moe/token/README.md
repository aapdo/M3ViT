# Token MoE (현재 개발 중)

Token-level MoE with ShareabilityPredictor.
토큰별로 shared/task-specific 경로를 Gumbel-softmax로 분기하는 방식.

## 파일

| 파일 | 설명 |
|------|------|
| `vision_transformer_moe.py` | Token MoE ViT backbone (`TokenVisionTransformerMoE`) |
| `custom_moe_layer.py` | Token-level FMoE MLP layer (`TokenFMoETransformerMLP`) |
| `noisy_gate_vmoe.py` | Token 전용 noisy gate (`TokenNoisyGate_VMoE`) |
| `vit_up_head.py` | Token-aware upsampling head (`TokenVisionTransformerUpHead`) |

## 핵심 컴포넌트

- **ShareabilityPredictor**: Gumbel-softmax 기반 라우터. 각 토큰이 task-specific expert vs shared expert 중 선택
- **Aggregation**: Shared position의 여러 태스크 출력을 합산
- **Temperature Schedule**: cosine annealing (1.5 → 0.5, warmup 5 epochs)

## 실행 방법

### NYUD (semseg + depth)
```bash
torchrun --nproc_per_node=2 train_fastmoe.py \
    --config_exp configs/nyud/token_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True
```

### PASCAL Context (5 tasks)
```bash
torchrun --nproc_per_node=2 train_fastmoe.py \
    --config_exp configs/pascal/token_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True \
    --share_gamma 0.5
```

## 관련 Config

- `configs/nyud/token_moe/pup_moe_vit_small_multi_task_baseline.yml`
- `configs/nyud/token_moe/pup_moe_vit_base_multi_task_baseline.yml`
- `configs/pascal/token_moe/pup_moe_vit_small_multi_task_baseline.yml`
