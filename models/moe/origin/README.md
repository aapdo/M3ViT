# Origin MoE (M3ViT 원본)

M3ViT 논문의 원본 FMoE 기반 구현. **참고용으로만 보관** (실사용 X).

## 파일

| 파일 | 설명 |
|------|------|
| `vision_transformer_moe.py` | FMoE 기반 MoE ViT backbone (`VisionTransformerMoE`) |
| `custom_moe_layer.py` | FMoE Transformer MLP layer (`FMoETransformerMLP`) |
| `noisy_gate_vmoe.py` | V-MoE 스타일 noisy gate (`NoisyGate_VMoE`) |

## 참고

- GPU 메모리 이슈로 gradient checkpointing이 필요하여 `ckpt/` 버전으로 대체됨
- 사용하려면 `train_fastmoe.py`에서 `--use_checkpointing False` 옵션 필요
