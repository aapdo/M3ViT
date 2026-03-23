# Mixture-of-Experts (MoE)

ViT backbone에 Sparse MoE를 적용한 모듈.
FFN을 N개 expert로 분할하고, NoisyGate로 top-k routing을 수행.

## 변형 (Variants)

| 디렉토리 | 설명 | 상태 |
|----------|------|------|
| `origin/` | M3ViT 원본 FMoE 구현 | 참고용 (실사용 X) |
| `ckpt/` | Gradient checkpointing 적용 버전 | 활성 사용 |
| `token/` | Token-level MoE + ShareabilityPredictor | **현재 개발 중** |

## 공통 모듈

| 파일 | 설명 |
|------|------|
| `noisy_gate.py` | FMoE base gate (NoisyGate) |
| `gates.py` | NoisyGate, NoisyGate_VMoE 커스텀 구현 |
| `moe.py` | Sparse MoE layer, TaskMoE (ParallelExperts 기반) |
| `parallel_experts.py` | 커스텀 autograd 병렬 expert 연산 |
| `aggregation_stages.py` | 태스크 출력 aggregation |

## 아키텍처 흐름

```
Input Tokens
    |
    v
[Attention] -> [NoisyGate] -> top-k experts 선택
                    |
              [Expert FFN x N] (ParallelExperts)
                    |
              weighted sum
                    |
                    v
            Output Tokens + CV loss (load balancing)
```

## Config 값

```yaml
backbone: VisionTransformer_moe    # ckpt 또는 origin 버전
# 또는
backbone: Token_VisionTransformer_moe  # token 버전

backbone_kwargs:
  moe_mlp_ratio: 2
  moe_top_k: 2
  gate_dim: 386
  vmoe_noisy_std: 1.0
```
