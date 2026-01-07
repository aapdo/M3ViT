# M3ViT Configuration Guide

이 문서는 M3ViT 모델의 각 컴포넌트별 설정 파라미터와 동작 방식을 설명합니다.

---

## 1. Transformations

이 섹션에서는 모델 학습 및 평가에 사용되는 데이터 전처리(Data Preprocessing) 파이프라인인 `Transformations`에 대해 설명합니다. 이미지 데이터는 모델에 입력되기 전에 다양한 변환 과정을 거쳐 정규화되고 증강(augmentation)됩니다.

### Train transformations (훈련 시 변환)

```
Compose(
    RandomHorizontalFlip
    ScaleNRotate:(rot=(-20, 20),scale=(0.75, 1.25))
    FixedResize:{'image': (512, 512), 'human_parts': (512, 512), 'sal': (512, 512)}
    AddIgnoreRegions
    ToTensor
    Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
)
```

*   **`RandomHorizontalFlip`**: 이미지를 50% 확률로 수평 뒤집기합니다. 이는 모델이 객체의 좌우 방향 변화에 둔감해지도록 돕습니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L190)
*   **`ScaleNRotate`**: 이미지를 무작위로 회전(`rot=(-20, 20)`)시키고 크기를 조절(`scale=(0.75, 1.25)`)합니다. 모델이 다양한 크기와 각도의 객체를 인식하도록 합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L16)
*   **`FixedResize`**: 이미지와 각 태스크(human_parts, sal)에 해당하는 레이블의 크기를 (512, 512)로 고정합니다. 모델의 입력 크기를 통일합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L82)
*   **`AddIgnoreRegions`**: 특정 영역을 모델이 학습하지 않도록 무시 영역으로 추가합니다. (예: 주석이 없는 영역)
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L254)
*   **`ToTensor`**: NumPy 배열로 된 이미지 데이터를 PyTorch 텐서 형식으로 변환합니다. 이미지 픽셀 값의 스케일을 [0, 1] 범위로 조절합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L281)
*   **`Normalize`**: PyTorch 텐서로 변환된 이미지의 각 채널(RGB)에 대해 평균과 표준편차를 이용하여 정규화합니다. 이는 사전학습된 모델(ImageNet)의 입력 분포와 일치시켜 학습 안정성을 높입니다.
    *   평균: `[0.485, 0.456, 0.406]`
    *   표준편차: `[0.229, 0.224, 0.225]`
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L309)

### Val transformations (평가 시 변환)

평가 데이터에 적용되는 변환은 다음과 같습니다. 훈련 시와 달리 데이터 증강은 적용되지 않고, 주로 정규화 및 크기 조정에 초점을 맞춥니다.

```
Compose(
    FixedResize:{'image': (512, 512), 'human_parts': (512, 512), 'sal': (512, 512)}
    AddIgnoreRegions
    ToTensor
    Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
)
```

*   **`FixedResize`**: 훈련 시와 동일하게 이미지 및 레이블의 크기를 (512, 512)로 고정합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L82)
*   **`AddIgnoreRegions`**: 훈련 시와 동일하게 무시 영역을 추가합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L254)
*   **`ToTensor`**: NumPy 배열을 PyTorch 텐서로 변환합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L281)
*   **`Normalize`**: PyTorch 텐서를 정규화합니다.
    *   코드 위치: [data/custom_transforms.py](data/custom_transforms.py#L309)

---

## 2. MoE ViT Model

이 섹션에서는 Mixture-of-Experts(MoE)를 적용한 Vision Transformer 모델의 각 컴포넌트와 파라미터를 설명합니다.

---

### 2.1 VisionTransformerMoE

VisionTransformerMoE는 Mixture-of-Experts(MoE)를 적용한 Vision Transformer의 메인 클래스입니다.

#### 파라미터 설명

##### 기본 아키텍처 파라미터

**`model_name: vit_small_patch16_224`**
- 사용할 사전학습된 모델의 이름
- 코드 위치: [vision_transformer_moe.py:361](models/vision_transformer_moe.py#L361)
- 설정 파일: [vision_transformer_moe.py:32-69](models/vision_transformer_moe.py#L32-L69) `default_cfgs` 딕셔너리
- 동작:
  - `random_init=False`일 때 이 이름으로 사전학습 가중치 URL 찾기
  - 예: `vit_small_patch16_224` → DeiT small 모델 가중치 로드
  - 지원 모델: `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`, etc.

**`img_size: [512, 512]`**
- 입력 이미지 크기 (Height, Width)
- 코드 위치: [vision_transformer_moe.py:362](models/vision_transformer_moe.py#L362)
- 동작:
  - Patch 개수 계산: `h = 512/16 = 32`, `w = 512/16 = 32` → 총 1024개 패치
  - PatchEmbed 초기화 시 사용: [vision_transformer_moe.py:404-405](models/vision_transformer_moe.py#L404-L405)
  - Position embedding interpolation 시 타겟 크기로 사용

**`patch_size: 16`**
- 이미지를 패치로 나눌 때의 패치 크기
- 코드 위치: [vision_transformer_moe.py:363](models/vision_transformer_moe.py#L363)
- 동작:
  - 16×16 픽셀 단위로 이미지를 패치로 분할
  - 패치 개수 = (img_size[0]/patch_size) × (img_size[1]/patch_size) = 32×32 = 1024
  - PatchEmbed의 Conv2d stride로 사용

**`in_chans: 3`**
- 입력 이미지 채널 수 (RGB = 3)
- 코드 위치: [vision_transformer_moe.py:364](models/vision_transformer_moe.py#L364)
- 동작: PatchEmbed Conv2d의 입력 채널 수

**`embed_dim: 384`**
- Transformer 임베딩 차원 (hidden dimension)
- 코드 위치: [vision_transformer_moe.py:365](models/vision_transformer_moe.py#L365)
- 동작:
  - 각 패치가 384차원 벡터로 임베딩됨
  - Attention, MLP 등 모든 레이어의 feature 차원
  - `vit_small` 모델의 표준 값

**`depth: 12`**
- Transformer Block의 개수
- 코드 위치: [vision_transformer_moe.py:366](models/vision_transformer_moe.py#L366)
- 동작:
  - 총 12개의 Block 생성 (0-11 인덱스)
  - **중요**: 짝수 블록(0,2,4,6,8,10)은 일반 Transformer Block
  - **중요**: 홀수 블록(1,3,5,7,9,11)은 MoE Transformer Block
  - 코드: [vision_transformer_moe.py:425-437](models/vision_transformer_moe.py#L425-L437)

**`num_heads: 12`**
- Multi-head Attention의 head 개수
- 코드 위치: [vision_transformer_moe.py:367](models/vision_transformer_moe.py#L367)
- 동작:
  - **`vit_small` 모델의 표준 설정(6개)과 다름**: 이 설정은 표준 DeiT-Small 모델의 `num_heads=6`과 다릅니다. 하지만 이는 의도된 설계 변경일 가능성이 높습니다.
  - **가중치 호환성**: `Attention` 모듈의 `qkv` 레이어 가중치 행렬의 모양(shape)은 `embed_dim`에만 의존하고 `num_heads`와는 무관합니다. 따라서 `num_heads=6`으로 학습된 사전학습 가중치를 `num_heads=12`인 모델에 로드할 때 모양 불일치 오류가 발생하지 않습니다.
  - **동작 방식의 차이**: 가중치는 성공적으로 로드되지만, `forward` 연산 시 `qkv`의 출력 텐서를 12개의 헤드로 재해석(reshape)하여 사용합니다 (헤드 당 차원: 384 / 12 = 32). 이는 6개의 헤드(헤드 당 차원: 64)로 학습된 방식과 다르지만, 랜덤 초기화보다 나은 시작점을 제공할 수 있습니다.

**`num_classes: 40`**
- 분류 클래스 개수 (PASCAL Context 데이터셋의 경우 40개 클래스)
- 코드 위치: [vision_transformer_moe.py:368](models/vision_transformer_moe.py#L368)
- 동작:
  - `VisionTransformerMoE` 자체는 최종 분류 헤드를 직접 포함하지 않습니다. (모델의 `head` 레이어는 주석 처리되어 있습니다.)
  - 이 파라미터는 주로 다음과 같은 목적으로 사용됩니다:
    1.  **사전학습 가중치 로드 시**: `utils/helpers.py`의 `load_pretrained` 함수에서, 이 `num_classes` 값이 사전학습 모델의 클래스 수(`cfg['num_classes']`)와 다를 경우, 사전학습된 분류 헤드(classifier head)의 가중치가 모델에 로드되지 않도록 삭제됩니다. 이는 클래스 수 불일치로 인한 오류를 방지하고, 새로운 태스크에 맞는 분류 헤드가 랜덤 초기화되도록 합니다. 관련 코드: [helpers.py:280-301](utils/helpers.py#L280-L301)
    2.  **태스크 헤드 정의 시**: `VisionTransformerUpHead`와 같은 외부 태스크별 헤드 모듈에서 최종 출력 레이어의 채널 수를 정의하는 데 사용됩니다. 이 경우 `num_classes`는 해당 태스크의 실제 클래스 수를 나타냅니다. 관련 코드: [models/vit_up_head.py](models/vit_up_head.py)
  - Semantic segmentation task head의 출력 채널 수에 영향을 줍니다.

**`mlp_ratio: 4.0`**
- MLP hidden dimension의 확장 비율
- 코드 위치: [vision_transformer_moe.py:369](models/vision_transformer_moe.py#L369)
- 동작:
  - MLP hidden dim = embed_dim × mlp_ratio = 384 × 4 = 1536
  - 일반 Block(짝수 인덱스)의 MLP에 적용
  - 표준 ViT의 기본값

**`qkv_bias: True`**
- Query, Key, Value projection에 bias 사용 여부
- 코드 위치: [vision_transformer_moe.py:370](models/vision_transformer_moe.py#L370)
- 동작:
  - True: QKV Linear layer에 bias 추가
  - Attention 계산: [vision_transformer_moe.py:205](models/vision_transformer_moe.py#L205)

**`qk_scale: None`**
- Query-Key dot product의 스케일 값 (None이면 자동 계산)
- 코드 위치: [vision_transformer_moe.py:371](models/vision_transformer_moe.py#L371)
- 동작:
  - None일 때: scale = (head_dim)^(-0.5) = 32^(-0.5) ≈ 0.177
  - Attention 계산 시 Q@K^T를 scale로 나눔

**`representation_size: None`**
- Pre-logits representation layer의 차원 (None이면 사용 안 함)
- 코드 위치: [vision_transformer_moe.py:442-449](models/vision_transformer_moe.py#L442-L449)
- 동작:
  - None: Identity layer 사용 (bypass)
  - 값 지정 시: Linear + Tanh activation 추가

**`distilled: False`**
- DeiT distillation token 사용 여부
- 코드 위치: [vision_transformer_moe.py:387](models/vision_transformer_moe.py#L387)
- 동작:
  - False: CLS token만 사용 (num_token = 1)
  - True: CLS + distillation token 사용 (num_token = 2)

##### Regularization 파라미터

**`drop_rate: 0.0`**
- MLP와 projection의 dropout 비율
- 코드 위치: [vision_transformer_moe.py:372](models/vision_transformer_moe.py#L372)
- 동작: Attention projection, MLP의 Dropout layer에 적용

**`attn_drop_rate: 0.0`**
- Attention weights의 dropout 비율
- 코드 위치: [vision_transformer_moe.py:373](models/vision_transformer_moe.py#L373)
- 동작: Softmax 후 attention map에 dropout 적용

**`drop_path_rate: 0.0`**
- Stochastic depth (DropPath) 비율
- 코드 위치: [vision_transformer_moe.py:374](models/vision_transformer_moe.py#L374)
- 동작:
  - 각 블록에 선형 증가하는 drop_path 비율 적용
  - dpr[i] = i × drop_path_rate / (depth-1)
  - 0.0이면 DropPath 사용 안 함

##### 초기화 파라미터

**`random_init: False`** ⭐ 중요
- 가중치 랜덤 초기화 여부
- 코드 위치: [vision_transformer_moe.py:379](models/vision_transformer_moe.py#L379) 및 [vision_transformer_moe.py:483-490](models/vision_transformer_moe.py#L483-L490)
- 동작:
  - **`False` (현재 설정)**:
    1.  `model_name`에 해당하는 사전학습 가중치(예: DeiT-Small)를 불러옵니다.
    2.  `torch.load_state_dict`를 `strict=False` 모드로 호출하여, 이름과 모양이 일치하는 가중치만 선택적으로 로드합니다.
    3.  **선택적 가중치 로드 방식**:
        - **로드 성공**: Patch embedding ✓, Position embedding, Attention 레이어, 짝수 블록의 표준 MLP 등 사전학습 모델과 구조가 동일한 부분의 가중치가 성공적으로 로드됩니다.
        - **로드 실패 (랜덤 초기화)**: 홀수 블록의 MoE 레이어(gate, experts)는 사전학습 모델에 존재하지 않으므로, 로드되지 않고 모델 생성 시 적용된 초기값을 유지합니다. 이 초기화 방식은 MoE의 구성 요소에 따라 다릅니다.
          - **Gate Network (`NoisyGate_VMoE`)**: 게이트의 가중치(`w_gate`)는 **Kaiming Uniform 초기화**를 사용합니다. 이 로직은 `models/gate_funs/noisy_gate_vmoe.py`의 `reset_parameters` 메소드에 정의되어 있습니다.
          - **Expert Layers (`_Expert`)**: Expert 내부의 Linear 레이어들은 `VisionTransformerMoE`의 `init_weights` 메소드에 의해 기본적으로 **Truncated Normal 분포** (`trunc_normal_` 함수, `std=0.02`)로 초기화됩니다.
    4.  관련 코드: 가중치 로딩 로직은 `utils/helpers.py`의 `load_pretrained` 함수에 구현되어 있습니다. [helpers.py:192-301](utils/helpers.py#L192-L301)
  - **`True`**:
    - 모든 가중치를 랜덤 초기화하여 모델을 처음부터(from scratch) 학습시킵니다. 사전학습 가중치를 전혀 사용하지 않습니다.

**`pos_embed_interp: True`**
- Position embedding interpolation(보간) 사용 여부
- 코드 위치: [vision_transformer_moe.py:378](models/vision_transformer_moe.py#L378)
- 동작:
  - **`True` (현재 설정)**: 사전학습 모델의 위치 임베딩을 현재 모델의 입력 이미지 해상도에 맞게 동적으로 크기를 조절합니다. 이는 해상도가 다른 데이터셋에 파인튜닝할 때 필수적입니다.
  - **보간 과정**:
    1.  사전학습된 `pos_embed` 텐서에서 `[CLS]` 토큰에 해당하는 임베딩과 나머지 공간적(spatial) 패치 임베딩을 분리합니다.
    2.  공간적 패치 임베딩을 2D 그리드 모양으로 복원합니다. (예: 196개 패치 → 14x14 그리드)
    3.  `torch.nn.functional.interpolate` 함수(bilinear 방식)를 사용하여 이 2D 그리드를 목표 해상도에 맞는 크기로 리사이즈합니다. (예: 14x14 → 32x32)
    4.  리사이즈된 그리드를 다시 1D 시퀀스로 펼친 후, 처음에 분리해 둔 `[CLS]` 토큰 임베딩과 다시 결합하여 최종적인 `pos_embed`를 만듭니다.
    5.  이 과정을 통해, 낮은 해상도(예: 224x224)에서 학습된 위치 정보를 높은 해상도(예: 512x512)에서도 사용할 수 있게 됩니다.
  - **관련 코드**: 이 로직은 `utils/helpers.py`의 `load_pretrained_pos_emb` 및 `load_pretrained` 함수 내에 구현되어 있습니다. [helpers.py:145-163](utils/helpers.py#L145-L163)

**`align_corners: False`**
- Interpolation 시 corner alignment 여부
- 코드 위치: [vision_transformer_moe.py:380](models/vision_transformer_moe.py#L380)
- 동작: `F.interpolate(..., align_corners=False)`

**`weight_init: ''`**
- 가중치 초기화 방법 ('jax', 'jax_nlhb', 'nlhb', '')
- 코드 위치: [vision_transformer_moe.py:458](models/vision_transformer_moe.py#L458)
- 동작:
  - '': 기본 truncated normal 초기화
  - 'jax': JAX 구현과 호환되는 초기화

##### Backbone 파라미터

**`hybrid_backbone: None`**
- CNN backbone 사용 여부 (None이면 순수 ViT)
- 코드 위치: [vision_transformer_moe.py:375](models/vision_transformer_moe.py#L375)
- 동작:
  - None: PatchEmbed (Conv2d) 사용
  - CNN 지정 시: HybridEmbed 사용 (CNN feature를 patch embedding으로)

**`norm_cfg: {'type': 'SyncBN', 'requires_grad': True}`**
- Normalization layer 설정 (주로 decoder head에서 사용)
- 코드 위치: [vision_transformer_moe.py:377](models/vision_transformer_moe.py#L377)
- 동작: Task-specific head의 normalization 설정

**`act_layer: None`**
- Activation function (None이면 GELU 사용)
- 코드 위치: [vision_transformer_moe.py:389](models/vision_transformer_moe.py#L389)
- 동작:
  - None → nn.GELU
  - MLP의 activation으로 사용

##### MoE 관련 파라미터

**`moe_mlp_ratio: 1`**
- MoE MLP의 hidden dimension 확장 비율
- 코드 위치: [vision_transformer_moe.py:432](models/vision_transformer_moe.py#L432)
- 동작:
  - MoE hidden dim = embed_dim × moe_mlp_ratio = 384 × 1 = 384
  - 일반 MLP(mlp_ratio=4)보다 작게 설정하여 계산량 감소
  - 여러 expert로 분산되므로 ratio를 낮춤

**`moe_experts: 8`**
- MoE의 expert 개수
- 코드 위치: [vision_transformer_moe.py:391](models/vision_transformer_moe.py#L391)
- 동작:
  - **분산 학습 환경**: 이 파라미터는 **GPU(워커)당 생성되는 expert의 수**를 지정합니다. FastMoE 프레임워크는 여러 GPU에 걸쳐 expert를 분산시켜 모델의 총 용량을 늘립니다.
  - **총 Expert 수**: 전체 모델의 총 expert 수는 `moe_experts` × `world_size`로 계산됩니다. 현재 실험 로그에 `world_size=2` (GPU 2개 사용)로 기록되어 있으므로, 총 expert 수는 8 * 2 = 16개가 됩니다.
  - **글로벌 Gating**: Gate network는 16개 전체 expert를 인지하고, 토큰을 특정 GPU에 있는 expert에게도 라우팅(All-to-All 통신)할 수 있습니다.
  - 홀수 블록마다 총 16개의 expert MLP가 생성되어 분산 배치됩니다.

**`moe_top_k: 4`**
- 각 토큰당 활성화할 expert 개수
- 코드 위치: [vision_transformer_moe.py:392](models/vision_transformer_moe.py#L392)
- 동작:
  - Gate network가 상위 4개 expert 선택
  - Sparse activation: 8개 중 4개만 사용 → 계산량 50%
  - 각 expert의 출력을 가중합산

**`world_size: 2`**
- 분산 학습의 GPU/프로세스 개수
- 동작:
  - Expert를 여러 GPU에 분산 배치
  - AllToAll communication으로 expert parallelism 구현

**`gate_dim: 389`**
- Gate network의 입력 차원
- 코드 위치: [vision_transformer_moe.py:417](models/vision_transformer_moe.py#L417)
- 동작:
  - gate_dim = embed_dim + task_onehot_dim
  - 389 = 384 (embed_dim) + 5 (task 개수, PASCAL Context의 경우)
  - `num_tasks = gate_dim - embed_dim = 5`
  - Gate network가 token feature + task identity를 입력받아 expert 선택

##### 고급 MoE 파라미터

**`moe_gate_type: "noisy_vmoe"`**
- Gate network의 종류 선택
- 코드 위치: [vision_transformer_moe.py:356](models/vision_transformer_moe.py#L356)
- 옵션:
  - `"noisy"`: NoisyGate (GShard/Switch Transformer 스타일)
    - 학습 가능한 noise stddev 사용
    - `w_noise` 파라미터로 토큰별로 noise 크기 조절
    - 코드: [noisy_gate.py:14-229](models/gate_funs/noisy_gate.py#L14-L229)
  - `"noisy_vmoe"`: NoisyGate_VMoE (Vision MoE 스타일) ← 현재 설정
    - 고정된 noise stddev 사용
    - 더 단순하고 안정적
    - 코드: [noisy_gate_vmoe.py:15-311](models/gate_funs/noisy_gate_vmoe.py#L15-L311)

**`vmoe_noisy_std: 0.0`**
- VMoE gate의 noise 표준편차 (moe_gate_type="noisy_vmoe"일 때만)
- 코드 위치: [vision_transformer_moe.py:356](models/vision_transformer_moe.py#L356)
- 동작:
  - **0.0 (현재 설정)**: noise 없음 → deterministic routing
    - 안정적인 학습
    - 추론과 동일한 동작
  - **> 0 (예: 1.0)**: noise 추가 → stochastic routing
    - 학습 시 exploration 증가
    - Expert 사용 다양성 향상
    - Load balancing 개선 효과
- 공식: `noise = randn() * vmoe_noisy_std * training_flag`

**`gate_task_specific_dim: -1`**
- Task embedding의 차원 (task-specific gate representation)
- 코드 위치: [vision_transformer_moe.py:356](models/vision_transformer_moe.py#L356)
- 동작:
  - **-1 (현재 설정)**: Task-specific representation 미사용
    - Task one-hot을 직접 gate input에 concat
    - 단순하고 효율적
  - **> 0 (예: 64)**: Task one-hot → MLP → task embedding
    - `gate_task_represent` MLP 생성: [vision_transformer_moe.py:423](models/vision_transformer_moe.py#L423)
    - Task 정보를 더 풍부하게 표현
    - 파라미터 추가 필요

**`multi_gate: True`**
- Task별 독립 gate network 사용 여부
- 코드 위치: [vision_transformer_moe.py:356](models/vision_transformer_moe.py#L356)
- 동작:
  - **False (현재 설정)**: 모든 task가 하나의 gate network 공유
    - 파라미터 효율적
    - Task 간 expert 공유 가능
  - **True**: 각 task마다 별도의 gate network
    - `num_tasks`개의 gate 생성
    - Task별로 완전히 독립적인 expert routing
    - 파라미터 크게 증가 (gate 개수 × num_tasks)
    - 코드: [custom_moe_layer.py:133-150](models/custom_moe_layer.py#L133-L150)

**`gate_return_decoupled_activation: False`** ⭐ 고급 파라미터
- Auxiliary gate activation 사용 여부 (load balancing loss 계산용)
- 코드 위치: [vision_transformer_moe.py:393](models/vision_transformer_moe.py#L393)
- 전달 경로: VisionTransformerMoE → Block → FMoETransformerMLP → NoisyGate/NoisyGate_VMoE
- 동작:
  - **False (기본값, 현재 설정)**:
    1. Gate network는 단일 가중치 세트만 사용: `w_gate` (+ `w_noise` for NoisyGate)
    2. Forward 시 단일 로짓 계산:
       ```python
       clean_logits = inp @ w_gate
       noisy_logits = clean_logits + noise
       ```
    3. Top-K expert 선택 및 routing에만 로짓 사용
    4. `self.activation`에 최종 로짓 저장 (외부 접근 가능)
    5. 코드: [noisy_gate.py:142-149](models/gate_funs/noisy_gate.py#L142-L149), [noisy_gate_vmoe.py:217-229](models/gate_funs/noisy_gate_vmoe.py#L217-L229)

  - **True (decoupled activation 사용)**:
    1. Gate network에 추가 가중치 생성:
       - NoisyGate: `w_gate_aux`, `w_noise_aux`
       - NoisyGate_VMoE: `w_gate_aux`
       - 코드: [noisy_gate.py:26-32](models/gate_funs/noisy_gate.py#L26-L32), [noisy_gate_vmoe.py:26-28](models/gate_funs/noisy_gate_vmoe.py#L26-L28)
    2. Forward 시 두 가지 로짓 계산:
       ```python
       # 주 로짓 (routing용)
       clean_logits = inp @ w_gate
       noisy_logits = clean_logits + noise

       # 보조 로짓 (activation 저장용)
       clean_logits_aux = inp @ w_gate_aux
       noisy_logits_aux = clean_logits_aux + noise_aux
       ```
    3. Top-K expert 선택: `noisy_logits` 사용
    4. **중요**: `self.activation`에 `noisy_logits_aux` 저장 (주 로짓과 분리)
    5. 코드: [noisy_gate.py:157-210](models/gate_funs/noisy_gate.py#L157-L210), [noisy_gate_vmoe.py:237-292](models/gate_funs/noisy_gate_vmoe.py#L237-L292)

  - **용도**:
    - Load balancing loss 계산 시 routing과 loss 계산을 분리
    - Routing 결정과 gradient flow를 독립적으로 제어
    - 고급 MoE 학습 테크닉 (논문: Switch Transformer, GShard 등)

  - **get_activation() 메소드**:
    - Gate의 activation(로짓) 값을 외부에서 접근
    - Load balancing loss, 분석, 디버깅 등에 활용
    - 코드: [noisy_gate.py:220-224](models/gate_funs/noisy_gate.py#L220-L224)

  - **현재 설정에서는**:
    - False이므로 보조 가중치 없음 → 파라미터 효율적
    - 단순한 구조로 안정적인 학습

#### 주요 동작 흐름

##### 1. 초기화 과정 (`random_init=False` 기준)

```python
__init__()
  ├─ PatchEmbed 생성 (Conv2d 16x16, in=3, out=384)
  ├─ Position embedding 파라미터 생성 (1025 tokens × 384 dim)
  ├─ CLS token 파라미터 생성
  ├─ 12개 Block 생성:
  │    ├─ Block 0, 2, 4, 6, 8, 10: 일반 Transformer (Attention + MLP)
  │    └─ Block 1, 3, 5, 7, 9, 11: MoE Transformer (Attention + MoE)
  └─ init_weights()
       ├─ 모든 Linear, LayerNorm 랜덤 초기화
       ├─ load_pretrained_pos_emb(): Position embedding 로드 & 보간
       └─ load_pretrained(): DeiT 가중치 로드
            ├─ Patch embedding ✓
            ├─ Attention layers ✓
            ├─ 짝수 블록 MLP ✓
            └─ MoE expert weights ✗ (랜덤 초기화, 사전학습 모델에 없음)
```

##### 2. Forward 과정

```
Input Image (B, 3, 512, 512)
  ↓
PatchEmbed: Conv2d
  ↓
Patches (B, 1024, 384) + CLS token → (B, 1025, 384)
  ↓
+ Position Embedding (interpolated)
  ↓
Block 0 (Attention + MLP)
  ↓
Block 1 (Attention + MoE) ← gate_inp(token + task_onehot) → Gate → Top-4 experts
  ↓
Block 2 (Attention + MLP)
  ↓
Block 3 (Attention + MoE)
  ↓
...
  ↓
Block 11 (Attention + MoE)
  ↓
Output Features (B, 1025, 384)
```

---

### 2.2 PatchEmbed

이미지를 패치로 나누고 임베딩하는 모듈입니다.

#### 파라미터

**`img_size: [512, 512]`**
- 입력 이미지 크기

**`patch_size: 16`**
- 패치 크기 (16×16 픽셀)

**`in_chans: 3`**
- 입력 채널 (RGB)

**`embed_dim: 384`**
- 출력 임베딩 차원

#### 구조

코드 위치: [vision_transformer_moe.py:285-310](models/vision_transformer_moe.py#L285-L310)

```python
self.proj = nn.Conv2d(
    in_channels=3,
    out_channels=384,
    kernel_size=16,
    stride=16
)
```

#### 동작

```
Input: (B, 3, 512, 512)
  ↓
Conv2d (kernel=16, stride=16)
  ↓
Output: (B, 384, 32, 32)
  ↓
Flatten & Transpose
  ↓
(B, 1024, 384)  # 1024 = 32×32 patches
```

- **패치 개수**: (512/16) × (512/16) = 32 × 32 = 1024
- **역할**: 16×16 픽셀 영역을 384차원 벡터로 projection

---

### 2.3 Block

Transformer의 기본 블록 (Attention + MLP 또는 Attention + MoE)

#### 파라미터

**`dim: 384`**
- Feature 차원

**`num_heads: 12`**
- Attention head 개수

**`mlp_ratio: 4.0`**
- 일반 MLP의 hidden 확장 비율

**`qkv_bias: True`**
- QKV projection bias 사용

**`drop: 0.0`**
- Dropout 비율

**`attn_drop: 0.0`**
- Attention dropout 비율

**`drop_path: 0.0`**
- DropPath 비율

**`moe: True/False`**
- MoE 사용 여부 (홀수 블록 True, 짝수 블록 False)

**MoE 관련 파라미터 (moe=True일 때만):**

**`moe_mlp_ratio: 1`**
- MoE MLP hidden 확장 비율

**`moe_experts: 8`**
- Expert 개수

**`moe_top_k: 4`**
- 활성화할 expert 개수

**`moe_gate_dim: 389`**
- Gate 입력 차원 (token feature + task info)

**`moe_gate_type: "noisy_vmoe"`**
- Gate 타입 ("noisy" 또는 "noisy_vmoe")

**`vmoe_noisy_std: 0.0`**
- Gate noise 표준편차 (학습 시 exploration용)

#### 구조

코드 위치: [vision_transformer_moe.py:226-283](models/vision_transformer_moe.py#L226-L283)

##### 일반 Block (짝수 인덱스)
```python
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(MLP(LayerNorm(x)))
```

##### MoE Block (홀수 인덱스)
```python
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(FMoETransformerMLP(LayerNorm(x), gate_inp, task_id))
```

#### 동작

**일반 Block:**
1. LayerNorm → Multi-head Attention → Residual
2. LayerNorm → MLP (Linear → GELU → Linear) → Residual

**MoE Block:**
1. LayerNorm → Multi-head Attention → Residual
2. LayerNorm → MoE (Gate → Top-K Experts → Weighted Sum) → Residual

---

### 2.4 Attention

Multi-head Self-Attention 모듈

#### 파라미터

**`dim: 384`**
- 입력/출력 차원

**`num_heads: 12`**
- Attention head 개수

**`qkv_bias: True`**
- QKV Linear에 bias 추가

**`qk_scale: None`**
- Attention scale (None이면 자동: 1/√head_dim)

**`attn_drop: 0.0`**
- Attention map dropout

**`proj_drop: 0.0`**
- Output projection dropout

#### 구조

코드 위치: [vision_transformer_moe.py:194-224](models/vision_transformer_moe.py#L194-L224)

```python
head_dim = 384 / 12 = 32
scale = 32^(-0.5) ≈ 0.177

QKV = Linear(384, 384×3, bias=True)
Proj = Linear(384, 384)
```

#### 동작

```
Input: (B, 1025, 384)
  ↓
QKV Linear → (B, 1025, 1152)
  ↓
Reshape → (B, 1025, 3, 12, 32)
  ↓
Permute → Q, K, V: (B, 12, 1025, 32)
  ↓
Attention = softmax(Q @ K^T / √32)
  ↓
Output = (Attention @ V)
  ↓
Reshape & Project → (B, 1025, 384)
```

---

### 2.5 FMoETransformerMLP

Mixture-of-Experts MLP 모듈 (FastMoE 기반)

#### 파라미터

**`num_expert: 8`**
- Expert MLP 개수

**`d_model: 384`**
- 입력/출력 차원

**`d_gate: 389`**
- Gate network 입력 차원 (384 token dim + 5 task dim)

**`d_hidden: 384`**
- Expert MLP hidden 차원 (moe_mlp_ratio × d_model)

**`world_size: 2`**
- 분산 학습 GPU 개수

**`top_k: 4`**
- 각 토큰당 활성화할 expert 개수

**`gate: NoisyGate_VMoE`**
- Gate function 클래스

**`vmoe_noisy_std: 0.0`**
- Gate noise 표준편차

**`multi_gate: False`**
- Task별 독립 gate 사용 여부

#### 구조

코드 위치: [custom_moe_layer.py:66-182](models/custom_moe_layer.py#L66-L182)

```python
# 8개 Expert (각각 2-layer MLP)
experts = _Expert(
    num_expert=8,
    d_model=384,
    d_hidden=384
)

# Expert 구조
Expert[i]:
  Linear1: 384 → 384
  GELU
  Dropout
  Linear2: 384 → 384
```

```python
# Gate Network
gate = NoisyGate_VMoE(
    d_model=389,  # 384 + 5
    num_expert=8,
    top_k=4
)
```

#### 동작

```
Input tokens: (B×1025, 384)
Gate input: (B×1025, 389)  # token feature + task one-hot
  ↓
Gate Network: Linear(389, 8)
  ↓
Gate scores: (B×1025, 8)
  ↓
Top-K selection: Top-4 experts per token
  ↓
Dispatch tokens to selected experts
  ↓
Each expert processes its assigned tokens
  ↓
Combine expert outputs (weighted by gate scores)
  ↓
Output: (B×1025, 384)
```

**예시**:
- Token 1: Expert [0, 2, 5, 7] 활성화, weights [0.3, 0.25, 0.25, 0.2]
- Token 2: Expert [1, 3, 4, 6] 활성화, weights [0.4, 0.3, 0.2, 0.1]
- Output = Σ(expert_output × weight)

---

### 2.6 NoisyGate_VMoE

VMoE 스타일의 Top-K Gating network with load balancing

#### 파라미터

**`d_model: 389`**
- 입력 차원 (token feature 384 + task one-hot 5)

**`num_expert: 8`**
- Expert 개수

**`top_k: 4`**
- 선택할 expert 개수

**`noise_std: 0.0`**
- Gating 시 추가할 noise 표준편차
- 학습 시 > 0: exploration 증가
- 추론 시 = 0: deterministic selection

**`return_decoupled_activation: False`** ⭐
- Auxiliary gate 사용 여부 (load balancing loss용)
- VisionTransformerMoE의 `gate_return_decoupled_activation` 파라미터로부터 전달됨
- 자세한 설명은 [VisionTransformerMoE의 gate_return_decoupled_activation](#visiontransformermoe) 섹션 참조

**`regu_experts_fromtask: False`**
- Task별 expert 그룹 제약 사용 여부
- True: 각 task가 특정 expert 서브셋만 사용하도록 제약

**`multi_gate: False`**
- Task별 독립 gate network 사용 여부
- True: 각 task마다 별도의 gate network 생성 (파라미터 증가)

#### 구조

코드 위치: [noisy_gate_vmoe.py:15-200](models/gate_funs/noisy_gate_vmoe.py#L15-L200)

```python
# 주 Gate 가중치 (routing용)
w_gate = nn.Parameter(torch.zeros(389, 8))

# Optional: 보조 Gate 가중치 (activation 저장용)
# return_decoupled_activation=True일 때만 생성
if return_decoupled_activation:
    w_gate_aux = nn.Parameter(torch.zeros(389, 8))
```

#### 동작

##### 기본 동작 (return_decoupled_activation=False)

```
Input: (N, 389)  # N = B×1025 tokens
  ↓
Logits = input @ w_gate → (N, 8)
  ↓
Add noise (training only):
  noise = randn(N, 8) × noise_std
  noisy_logits = logits + noise
  ↓
Top-K selection:
  values, indices = topk(noisy_logits, k=4)
  ↓
Softmax (only on top-4):
  gates = softmax(values)  # (N, 4)
  ↓
Store activation:
  self.activation = noisy_logits  # 저장 (외부 접근 가능)
  ↓
Output:
  - gate_idx: (N, 4) - selected expert indices
  - gate_score: (N, 4) - expert weights
```

##### Decoupled Activation 동작 (return_decoupled_activation=True)

```
Input: (N, 389)
  ↓
┌─────────────────────┬─────────────────────┐
│  주 경로 (routing)    │  보조 경로 (activation) │
├─────────────────────┼─────────────────────┤
│ logits = inp @ w_gate│ logits_aux = inp @ w_gate_aux
│ noisy_logits = ...   │ noisy_logits_aux = ...
│                     │                     │
│ Top-K selection     │ (routing에만 사용)   │
│ ↓                   │                     │
│ Expert routing      │                     │
└─────────────────────┴─────────────────────┘
                       ↓
              self.activation = noisy_logits_aux
                       ↓
                  (외부에서 접근)
                       ↓
          Load balancing loss 계산 등
```

**주요 차이점**:
- **False**: routing과 activation 저장에 동일한 로짓 사용
- **True**: routing과 activation 저장에 다른 로짓 사용 → gradient flow 분리

**Load Balancing**:
- Expert 사용 불균형 방지
- Auxiliary loss로 expert 간 부하 균등화
- Load = 각 expert가 처리하는 token 개수
- Decoupled activation을 사용하면 load balancing loss가 routing 결정에 미치는 영향 제어 가능

---

### 2.7 Mlp

일반 2-layer MLP (짝수 블록에서 사용)

#### 파라미터

**`in_features: 384`**
- 입력 차원

**`hidden_features: 1536`**
- Hidden 차원 (in_features × mlp_ratio = 384 × 4)

**`out_features: 384`**
- 출력 차원 (일반적으로 in_features와 동일)

**`act_layer: nn.GELU`**
- Activation function

**`drop: 0.0`**
- Dropout 비율

#### 구조

코드 위치: [vision_transformer_moe.py:156-172](models/vision_transformer_moe.py#L156-L172)

```python
fc1 = Linear(384, 1536)
act = GELU()
fc2 = Linear(1536, 384)
dropout = Dropout(0.0)
```

#### 동작

```
Input: (B, 1025, 384)
  ↓
Linear1: 384 → 1536
  ↓
GELU
  ↓
Dropout
  ↓
Linear2: 1536 → 384
  ↓
Dropout
  ↓
Output: (B, 1025, 384)
```

---

### 2.8 설정 예시 비교

#### 예시 1: 현재 설정 (PASCAL Context, Small model)

```yaml
embed_dim: 384
depth: 12
num_heads: 12
mlp_ratio: 4.0
moe_mlp_ratio: 1
moe_experts: 8
moe_top_k: 4
gate_dim: 389  # 384 + 5 tasks
```

**특징**:
- MoE는 홀수 블록만 (6개)
- 각 MoE 블록: 8개 expert 중 4개 활성화 (50% sparsity)
- MoE MLP는 일반 MLP보다 작음 (ratio 1 vs 4)
- Task information을 gate에 전달 (5-way multi-task)

#### 예시 2: Base model 설정

```yaml
embed_dim: 768
depth: 12
num_heads: 12
mlp_ratio: 4.0
moe_mlp_ratio: 1
moe_experts: 16
moe_top_k: 4
gate_dim: 773  # 768 + 5 tasks
```

**차이점**:
- Feature 차원 2배 (384 → 768)
- Expert 개수 2배 (8 → 16)
- Sparsity 더 높음 (16개 중 4개 = 25%)

#### 예시 3: 더 많은 expert 사용

```yaml
moe_experts: 16
moe_top_k: 4
```

**효과**:
- Expert specialization 증가 (각 expert가 더 특화됨)
- 계산량은 유사 (top_k 동일)
- 메모리 사용량 증가 (expert 개수만큼)

---

### 2.9 주요 설계 선택 이유

#### 1. 왜 홀수 블록만 MoE?
- **계산 효율성**: 모든 블록을 MoE로 하면 overhead 큼
- **안정성**: Attention은 공유, MLP만 task-specific으로 분리
- **성능**: 실험적으로 이 구조가 잘 작동함

#### 2. 왜 moe_mlp_ratio = 1 (일반은 4)?
- MoE는 여러 expert로 capacity 확보
- 각 expert를 작게 유지하여 파라미터 효율성 확보
- Top-K로 sparsity 있어서 작은 expert로도 충분

#### 3. 왜 gate_dim = embed_dim + num_tasks?
- Token feature만으로는 task 구분 어려움
- Task one-hot을 concat하여 task-aware expert selection
- Multi-task learning의 핵심 메커니즘

#### 4. 왜 random_init=False가 중요?
- ImageNet 사전학습으로 좋은 feature representation 확보
- MoE expert만 task-specific하게 학습
- Transfer learning으로 학습 속도 및 성능 향상

#### 5. gate_return_decoupled_activation은 언제 사용?
- **일반적으로 False (현재 설정)**:
  - 단순하고 안정적
  - 대부분의 경우 충분한 성능
  - 파라미터 효율적
- **True로 설정하는 경우**:
  - Load balancing loss의 영향을 세밀하게 제어하고 싶을 때
  - Expert 사용 패턴을 routing 결정과 독립적으로 분석하고 싶을 때
  - 고급 MoE 학습 기법 실험 시 (Switch Transformer, GShard 스타일)
  - 추가 파라미터 비용 감수 가능

---

## 3. Loss Functions (손실 함수)

이 섹션에서는 다중 작업 학습에 사용되는 손실 함수(Loss Functions)의 구조와 각 태스크별 손실 함수를 설명합니다.

### 3.1 MultiTaskLoss

PASCAL Context 데이터셋의 다중 작업 학습을 위한 손실 함수입니다. 여러 작업(task)에 대한 개별 손실 함수들을 관리하고 결합합니다.

#### 로그 출력 예시
```
Get loss
MultiTaskLoss(
  (loss_ft): ModuleDict(
    (semseg): SoftMaxwithLoss(...)
    (human_parts): SoftMaxwithLoss(...)
    (sal): BalancedCrossEntropyLoss()
    (edge): BalancedCrossEntropyLoss()
    (normals): NormalsLoss()
  )
)
```

#### 구조 설명

**`loss_ft: ModuleDict`**
- Task별 손실 함수를 저장하는 딕셔너리
- 각 task name을 키로, 해당 task의 손실 함수를 값으로 가짐
- 동작: 각 task의 예측값과 정답 레이블을 받아 task별 손실을 계산

---

### 3.2 Semantic Segmentation (semseg)

**태스크 설명**: 이미지의 각 픽셀을 미리 정의된 의미적 클래스(semantic class)로 분류하는 작업입니다. 예를 들어, 사람, 자동차, 나무, 건물 등의 클래스로 픽셀을 분류합니다.

**데이터셋**: PASCAL Context - 20개 클래스 + 배경(background)

#### 손실 함수: SoftMaxwithLoss

코드 위치: [losses/loss_functions.py:16-33](losses/loss_functions.py#L16-L33)

**구성 요소:**

**`LogSoftmax(dim=1)`**
- 클래스별 예측 점수(logits)를 로그 확률로 변환
- `dim=1`: 채널(클래스) 차원에 대해 softmax 적용
- 수식: `log(exp(x_i) / Σexp(x_j))`
- **장점**:
  - 수치 안정성(numerical stability) 향상
  - NLLLoss와 함께 사용 시 계산 효율적
  - Softmax + log 대신 직접 계산하여 underflow 방지

**`NLLLoss(ignore_index=255)`**
- Negative Log Likelihood Loss
- 다중 클래스 분류 문제의 표준 손실 함수
- `ignore_index=255`: 값이 255인 픽셀은 손실 계산에서 제외 (레이블이 없는 영역)
- 수식: `Loss = -log(p_target_class)`

**동작 흐름:**
```
Model output (logits): (B, C, H, W)  # C = 21 (20 클래스 + 배경)
  ↓
LogSoftmax → (B, 21, H, W)  # 로그 확률
  ↓
NLLLoss + Ground truth → Scalar loss
```

**특징:**
- `LogSoftmax + NLLLoss = CrossEntropyLoss`의 수치적으로 안정적인 구현
- 각 픽셀은 하나의 클래스에만 속함 (mutually exclusive)
- 클래스 불균형을 고려하지 않는 표준 손실

---

### 3.3 Human Part Segmentation (human_parts)

**태스크 설명**: 이미지에서 사람의 신체 부위를 분류하는 semantic segmentation 작업입니다. 머리, 몸통, 팔, 다리 등을 구별합니다.

**데이터셋**: PASCAL Context - 6개 부위 클래스 + 배경
- 클래스: background, head, torso, upper arm, lower arm, upper leg, lower leg

#### 손실 함수: SoftMaxwithLoss

코드 위치: [losses/loss_functions.py:16-33](losses/loss_functions.py#L16-L33)

**구성 요소:**

Semantic Segmentation과 동일한 구조:
- **`LogSoftmax(dim=1)`**: 로그 확률 계산
- **`NLLLoss(ignore_index=255)`**: 음의 로그 가능도 손실

**동작 흐름:**
```
Model output (logits): (B, 7, H, W)  # 7 = 6 부위 + 배경
  ↓
LogSoftmax → (B, 7, H, W)  # 로그 확률
  ↓
NLLLoss + Ground truth → Scalar loss
```

**특징:**
- Human parts는 상호 배타적인 클래스 (각 픽셀은 하나의 부위에만 속함)
- 사람이 있는 이미지만 레이블이 존재 (사람이 없으면 모두 배경)
- `ignore_index=255`로 레이블이 없는 영역 제외

---

### 3.4 Saliency Detection (sal)

**태스크 설명**: 이미지에서 시각적으로 두드러진(salient) 영역, 즉 사람의 시선을 끄는 중요한 객체나 영역을 검출하는 binary segmentation 작업입니다.

**데이터셋**: PASCAL Context - Binary labels (전경/배경)

#### 손실 함수: BalancedCrossEntropyLoss

코드 위치: [losses/loss_functions.py:36-84](losses/loss_functions.py#L36-L84)

**클래스 불균형 문제:**
- Saliency detection은 전형적인 클래스 불균형 문제
- 대부분 픽셀이 배경(negative), 소수만 salient object(positive)
- 예: 배경 95%, 전경 5%

**해결 방법 - Balanced Weight:**
```python
# 각 클래스의 픽셀 개수에 반비례하는 가중치 계산
num_positive = (label == 1).sum()
num_negative = (label == 0).sum()
total = num_positive + num_negative

weight_positive = num_negative / total  # 희소한 전경에 큰 가중치
weight_negative = num_positive / total  # 많은 배경에 작은 가중치

# 손실 계산 시 가중치 적용
loss = weight_positive * loss_positive + weight_negative * loss_negative
```

**동작 흐름:**
```
Model output (logits): (B, 1, H, W)  # Binary prediction
Ground truth: (B, 1, H, W)  # 0 or 1
  ↓
Calculate positive/negative pixel counts
  ↓
Compute balanced weights: w = num_neg / total
  ↓
Binary Cross Entropy with balanced weights
  ↓
Final loss = w * loss_pos + (1-w) * loss_neg
```

**효과:**
- 희소한 전경 픽셀의 손실에 더 큰 가중치
- 많은 배경 픽셀의 손실에 작은 가중치
- 전경/배경 모두에 대해 균형 잡힌 학습 가능
- 모델이 단순히 "모두 배경"으로 예측하는 것을 방지

**vs. 표준 CrossEntropyLoss:**
- 표준 CE: 모든 픽셀에 동일한 가중치 → 다수 클래스(배경)에 편향
- Balanced CE: 클래스 빈도에 따라 가중치 조정 → 균형 학습
- Saliency처럼 심한 불균형이 있을 때 필수적

**추가 옵션:**
- `pos_weight`: 수동으로 지정된 positive 가중치 (None이면 자동 계산)
- `void_pixels`: 무시할 영역 마스크 (옵션)

---

### 3.5 Edge Detection (edge)

**태스크 설명**: 이미지에서 객체의 경계선(edge)을 검출하는 binary segmentation 작업입니다. 물체의 윤곽선을 정확히 찾아냅니다.

**데이터셋**: PASCAL Context - Binary edge labels

#### 손실 함수: BalancedCrossEntropyLoss

코드 위치: [losses/loss_functions.py:36-84](losses/loss_functions.py#L36-L84)

**클래스 불균형 문제:**
- Edge detection도 극심한 클래스 불균형
- Edge 픽셀(positive)은 매우 희소: 보통 1~5%
- Non-edge 픽셀(negative)이 대부분: 95~99%

**Balanced Weight 적용:**
- Saliency detection과 동일한 BalancedCrossEntropyLoss 사용
- Edge 픽셀에 매우 큰 가중치 부여
- 배경 픽셀에 작은 가중치 부여

**동작 흐름:**
```
Model output (logits): (B, 1, H, W)  # Binary edge prediction
Ground truth: (B, 1, H, W)  # 0 (non-edge) or 1 (edge)
  ↓
Calculate edge/non-edge pixel counts
  ↓
Compute balanced weights: w = num_non_edge / total
  (w is typically very high, e.g., 0.95-0.99)
  ↓
Binary Cross Entropy with balanced weights
  ↓
Final loss = w * loss_edge + (1-w) * loss_non_edge
```

**특징:**
- Edge는 Saliency보다 더 희소함 (더 극심한 불균형)
- Balanced weight가 없으면 모델이 모든 픽셀을 non-edge로 예측
- HED(Holistically-nested Edge Detection) 스타일의 가중치 사용

**추가 파라미터:**
- `pos_weight`: Edge 가중치를 수동으로 조정 가능
- Config 파일에서 `edge_w` 파라미터로 설정

---

### 3.6 Surface Normal Estimation (normals)

**태스크 설명**: 이미지의 각 픽셀에서 표면의 법선 벡터(surface normal)를 예측하는 회귀(regression) 작업입니다. 3D 방향을 나타내는 (x, y, z) 벡터를 출력합니다.

**데이터셋**: PASCAL Context - 3채널 normal maps (x, y, z)

#### 손실 함수: NormalsLoss

코드 위치: [losses/loss_functions.py:154-197](losses/loss_functions.py#L154-L197)

**L1 Loss 기반:**
- 기본적으로 L1 loss (Mean Absolute Error) 사용
- 옵션으로 L2 loss (Mean Squared Error) 사용 가능

**Normalization:**
```python
class Normalize(nn.Module):
    def forward(self, x):
        # L2 norm으로 정규화 → 단위 벡터로 만듦
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12
        return x / norm
```

**동작 흐름:**
```
Model output: (B, 3, H, W)  # (x, y, z) normal vectors
Ground truth: (B, 3, H, W)  # (x, y, z) normal vectors
  ↓
[Optional] Normalize to unit vectors
  ↓
Mask out invalid regions (label == 255)
  ↓
L1 Loss on valid pixels: |pred - gt|
  ↓
Average over valid pixels
```

**Invalid Region 처리:**
- `ignore_label=255`: Normal 레이블이 없는 영역
- 유효한 마스크: `mask = (label != 255)`
- 손실 계산 시 유효한 픽셀만 사용

**Normalization 옵션:**
- `normalize=True`: 예측과 GT를 단위 벡터로 정규화
  - 방향만 중요하고 크기는 무시
  - 더 안정적인 학습
- `normalize=False`: Raw 벡터 값 사용 (기본값)

**Loss Function 선택:**
```python
# L1 loss (기본값)
loss = |pred - gt|  # Mean Absolute Error

# L2 loss (옵션)
loss = (pred - gt)^2  # Mean Squared Error
```

**특징:**
- 분류(classification)가 아닌 회귀(regression) 문제
- 3차원 벡터를 예측 → 픽셀당 3개 채널
- 유효한 normal이 있는 표면만 학습
- 하늘, 먼 배경 등은 normal이 정의되지 않음

---

### 3.7 전체 손실 계산

#### Multi-Task Loss 통합

```python
# Forward pass
predictions = model(images)

# Task별 손실 계산
loss_dict = {}
loss_dict['semseg'] = loss_ft['semseg'](pred['semseg'], gt['semseg'])
loss_dict['human_parts'] = loss_ft['human_parts'](pred['human_parts'], gt['human_parts'])
loss_dict['sal'] = loss_ft['sal'](pred['sal'], gt['sal'])
loss_dict['edge'] = loss_ft['edge'](pred['edge'], gt['edge'])
loss_dict['normals'] = loss_ft['normals'](pred['normals'], gt['normals'])

# 전체 손실: Task별 손실의 가중합
total_loss = sum([task_weight[t] * loss_dict[t] for t in tasks])
```

#### Task Weight 설정

각 task의 상대적 중요도와 손실 스케일에 따라 가중치 부여:

```python
# 예시 (config 파일에서 설정)
task_weights = {
    'semseg': 1.0,
    'human_parts': 2.0,  # 희소한 레이블 → 높은 가중치
    'sal': 1.0,
    'edge': 50.0,  # 매우 희소 → 매우 높은 가중치
    'normals': 10.0  # 회귀 손실 스케일 조정
}
```

**가중치 설정 이유:**
- **손실 스케일 균형**: Regression loss(normals)는 classification loss보다 자연스럽게 큼
- **데이터 빈도 보정**: 희소한 레이블(human parts, edge)에 더 큰 가중치
- **수렴 안정성**: 모든 task가 비슷한 속도로 학습되도록 조정
- **Task 중요도**: 특정 task를 더 강조하고 싶을 때

---

### 3.8 MoE Load Balancing Loss (보조 손실)

MoE 모델은 task loss 외에 추가적인 **load balancing loss**를 사용하여 expert 활용을 균형있게 유지합니다.

코드 위치: [models/gate_funs/noisy_gate_vmoe.py:270-286](models/gate_funs/noisy_gate_vmoe.py#L270-L286), [utils/moe_utils.py:105-111](utils/moe_utils.py#L105-L111), [train/train_utils.py:248-249](train/train_utils.py#L248-L249)

#### 문제: Expert Collapse

```
시나리오: 8개의 expert가 있을 때
- Expert 0, 1: 거의 모든 토큰 처리 (과부하)
- Expert 2-7: 거의 사용되지 않음 (유휴 상태)

결과:
→ 모델 용량 낭비, 성능 저하, 다양성 부족
```

#### Loss 수식 및 계산

**CV² (Coefficient of Variation Squared) 기반:**

```python
loss = cv_squared(importance) + cv_squared(load)

where cv_squared(x) = x.var() / (x.mean()² + eps)
```

- **Importance**: 각 expert의 gate weight 총합
- **Load**: 각 expert가 처리하는 토큰 개수 (확률적 계산)
- **CV² = 0**: 완벽히 균일한 분포 (이상적)
- **CV² > 0**: 불균형한 분포

#### Training에서 적용

```python
# Task loss
loss_dict = criterion(output, targets)

# Load balancing loss 추가
aux_loss = collect_noisy_gating_loss(model, weight=0.01)
total_loss = loss_dict['total'] + aux_loss

# Backward
total_loss.backward()
```

---

### 3.9 Loss Backward 및 Gradient 계산

이 섹션에서는 loss 계산 후 backward pass와 MoE 분산 학습에서의 gradient 동기화를 설명합니다.

#### 전체 Training Step

코드 위치: [train/train_utils.py:198-280](train/train_utils.py#L198-L280)

```python
for batch in train_loader:
    # 1. Forward Pass
    output = model(images)

    # 2. Loss 계산
    loss_dict = criterion(output, targets)
    if MoE:
        loss_dict['total'] += collect_noisy_gating_loss(model, 0.01)

    # 3. Backward Pass
    optimizer.zero_grad()  # Gradient 초기화
    loss_dict['total'].backward()  # Backpropagation

    # 4. Gradient 동기화 (분산 학습)
    if MoE:
        model.allreduce_params()  # Expert gradients 동기화

    # 5. Optimizer Step
    optimizer.step()  # 파라미터 업데이트
```

#### Backward Pass 상세

**자동 미분 (Automatic Differentiation):**

```python
# Computation graph (forward)
loss → output → features → embeddings → images

# Gradient flow (backward, reverse order)
∂loss/∂param = chain rule 적용

# PyTorch 자동 계산
loss.backward()  # 모든 param.grad 계산
```

#### MoE Gradient 동기화

**문제: Expert가 여러 GPU에 분산**

```
GPU 0: Experts [0-7]
GPU 1: Experts [8-15]
```

**AllReduce 연산:**

```python
model.allreduce_params()

# 동작:
# 1. 각 GPU의 expert gradient 수집
# 2. All-to-All communication으로 동일 expert의 gradient 합산
# 3. 평균 계산 및 모든 GPU에 broadcast
```

**왜 필요?**
- 각 GPU는 다른 batch 처리
- Expert는 모든 데이터에 대해 학습해야 함
- Gradient 동기화 없으면 model divergence

---

## 4. Optimizer (최적화 알고리즘)

이 섹션에서는 모델 학습에 사용되는 최적화 알고리즘과 하이퍼파라미터를 설명합니다.

### 4.1 SGD with Momentum

#### 로그 출력 예시
```
Retrieve optimizer
Optimizer uses a single parameter group - (Default)
optimizer SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.002
    maximize: False
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0001
)
```

### 4.2 파라미터 설명

#### 기본 설정

**`optimizer: SGD` (Stochastic Gradient Descent)**
- 확률적 경사 하강법
- PyTorch의 `torch.optim.SGD` 사용
- Vision 모델 학습에 널리 사용되는 안정적인 최적화 방법

**Parameter Group 구조:**
- "Optimizer uses a single parameter group": 모든 파라미터가 동일한 학습률 및 설정 공유
- Multi-group 설정 시: backbone과 head를 다른 학습률로 학습 가능

#### 학습률 설정

**`lr: 0.002`** (Learning Rate)
- 각 업데이트 스텝에서 파라미터를 조정하는 크기
- **0.002 = 2e-3**: Fine-tuning에 적합한 작은 학습률
- 사전학습된 모델을 사용하므로 큰 학습률은 불안정할 수 있음

**학습률 선택 이유:**
```python
# 일반적인 학습률 범위
- From scratch 학습: 0.01 ~ 0.1
- Fine-tuning: 0.001 ~ 0.01  ← 현재 설정 (0.002)
- Feature extraction: 0.0001 ~ 0.001
```

**학습률 스케줄링:**
- 로그에는 나타나지 않지만, 일반적으로 cosine annealing이나 step decay 사용
- 학습이 진행됨에 따라 학습률을 점진적으로 감소
- Fine-tuning 안정성과 수렴 품질 향상

#### Momentum 설정

**`momentum: 0.9`**
- 이전 gradient의 관성을 현재 업데이트에 반영
- **수식**:
  ```python
  v_t = momentum * v_{t-1} + gradient_t
  parameter -= lr * v_t
  ```
- **0.9**: 이전 속도의 90%를 유지하며 새 gradient 10% 추가

**효과:**
- **가속화**: 일관된 방향으로는 빠르게 이동
- **진동 완화**: 불안정한 gradient의 영향 감소
- **Local minima 탈출**: 관성으로 작은 언덕을 넘을 수 있음
- **수렴 속도 향상**: 순수 SGD보다 빠른 수렴

**왜 0.9?**
- Vision 모델의 표준 값 (실험적으로 검증됨)
- ImageNet 등 대부분의 벤치마크에서 사용
- 0.9 < momentum < 0.99 범위가 일반적

**`dampening: 0`**
- Momentum 계산 시 감쇠 계수
- 0: 감쇠 없음 (표준 momentum 수식 사용)
- > 0: 현재 gradient에 감쇠 적용
- 일반적으로 0으로 설정

**`nesterov: False`**
- Nesterov Accelerated Gradient (NAG) 사용 여부
- **False (현재 설정)**: 표준 momentum 사용
- **True일 때**: "lookahead" momentum
  ```python
  # Nesterov momentum
  v_t = momentum * v_{t-1} + gradient(parameter - momentum * v_{t-1})
  ```
- **NAG의 장점**:
  - 더 정확한 gradient 방향 추정
  - 진동 더욱 감소
  - 수렴 품질 향상 가능
- **False로 설정한 이유**:
  - 표준 momentum도 충분히 효과적
  - 계산 오버헤드 약간 적음
  - 하이퍼파라미터 튜닝이 더 직관적

#### Regularization

**`weight_decay: 0.0001`** (L2 Regularization)
- 가중치 감쇠: 파라미터 크기에 페널티 부여
- **수식**: `loss_total = loss + (weight_decay / 2) * Σ(w^2)`
- **0.0001 = 1e-4**: Vision Transformer의 일반적인 값

**효과:**
- **과적합 방지**: 가중치가 너무 커지는 것 방지
- **일반화 성능 향상**: 테스트 데이터에서 성능 개선
- **안정적인 학습**: Exploding gradients 완화

**ViT에서의 Weight Decay:**
- CNN보다 작은 값 사용 (1e-4 vs 5e-4)
- Self-attention은 이미 implicit regularization 효과 있음
- 너무 큰 weight decay는 사전학습 가중치 손상 가능

**어떤 파라미터에 적용?**
```python
# 일반적으로 제외되는 파라미터:
- Bias terms: weight_decay=0
- LayerNorm/BatchNorm parameters: weight_decay=0
- Positional embeddings: weight_decay=0

# 적용되는 파라미터:
- Linear layer weights
- Convolution weights
- Attention QKV matrices
```

#### PyTorch 고급 옵션

**`differentiable: False`**
- Optimizer 자체를 미분 가능하게 할지 여부
- Meta-learning이나 higher-order optimization에서만 사용
- 일반 학습에서는 항상 False

**`foreach: None`**
- Multi-tensor 병렬 업데이트 사용 여부
- None: PyTorch가 자동으로 결정
- True: 여러 파라미터를 한 번에 업데이트 (속도 향상)
- False: 순차적 업데이트

**`maximize: False`**
- False: 손실 최소화 (기본값)
- True: 손실 최대화 (GAN의 discriminator 등 특수한 경우)

### 4.3 SGD vs. Adam 비교

#### 왜 SGD를 사용하는가?

**SGD with Momentum의 장점:**
- **일반화 성능**: Test set에서 더 좋은 성능 (Vision 태스크에서)
- **안정성**: 사전학습 모델 fine-tuning 시 안정적
- **메모리 효율**: Adam보다 메모리 사용량 적음
- **검증된 방법**: ImageNet 등 대부분의 Vision 벤치마크에서 사용

**Adam의 장점과 한계:**
```
Adam (Adaptive Moment Estimation)
장점:
  - 적응적 학습률: 파라미터별로 학습률 자동 조정
  - 빠른 초기 수렴: 학습 초기에 빠르게 수렴
  - Learning rate에 덜 민감

한계:
  - 과적합 경향: Vision 태스크에서 generalization gap
  - 메모리 사용: 1st, 2nd moment 저장 필요 (2배 메모리)
  - Fine-tuning 불안정: 사전학습 모델과 궁합 안 좋을 수 있음
```

**실험적 결과:**
- ImageNet 분류: SGD > Adam (일반적으로)
- Object detection: SGD 사용이 표준
- NLP 태스크: Adam/AdamW 선호 (Transformer 학습)
- ViT Fine-tuning: SGD with momentum 권장

### 4.4 전체 최적화 과정

#### 한 번의 Training Step

```python
# 1. Forward pass
outputs = model(images)
loss = criterion(outputs, targets)

# 2. Backward pass
optimizer.zero_grad()  # 이전 gradient 초기화
loss.backward()        # Gradient 계산

# 3. Optimizer step (SGD with momentum)
for param in model.parameters():
    # Gradient에 weight decay 적용
    param.grad += weight_decay * param.data

    # Momentum 업데이트
    velocity = momentum * velocity + param.grad

    # 파라미터 업데이트
    param.data -= lr * velocity
```

#### Learning Rate Scheduling (일반적인 전략)

```python
# Cosine Annealing (추정)
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))

# Warmup (초기 몇 epoch)
if epoch < warmup_epochs:
    lr_t = lr_max * (epoch / warmup_epochs)
```

**Warmup의 중요성:**
- Fine-tuning 초기에 큰 학습률은 불안정
- 처음 몇 epoch은 작은 학습률로 시작
- 사전학습 가중치가 task에 적응할 시간 제공

### 4.5 하이퍼파라미터 튜닝 가이드

#### Learning Rate 조정

```python
# 너무 큰 학습률 증상:
- Loss가 발산 (NaN, Inf)
- Training이 불안정
- Validation loss가 진동

# 너무 작은 학습률 증상:
- 수렴이 매우 느림
- Local minima에 갇힘
- Underfitting

# 적절한 학습률 찾기:
1. Learning rate range test (LR finder)
2. 0.001, 0.002, 0.005, 0.01 시도
3. Warmup + Cosine decay 사용
```

#### Momentum 조정

```python
# 일반적 범위: 0.85 ~ 0.95
momentum = 0.9  # 표준값, 대부분 잘 작동
momentum = 0.95 # 더 부드러운 수렴, 느릴 수 있음
momentum = 0.85 # 더 빠른 반응, 불안정할 수 있음
```

#### Weight Decay 조정

```python
# 과적합이 심할 때:
weight_decay = 1e-3 ~ 1e-2  # 더 강한 regularization

# 학습이 너무 제한될 때:
weight_decay = 1e-5 ~ 1e-6  # 더 약한 regularization

# ViT 표준:
weight_decay = 1e-4  # 현재 설정, 균형잡힌 값
```

---

## 5. Evaluation Metrics (평가 지표)

이 섹션에서는 각 태스크별로 모델 성능을 평가하는 지표(metrics)의 계산 방법과 의미를 설명합니다.

### 5.1 Semantic Segmentation (semseg) Metrics

코드 위치: [evaluation/eval_semseg.py](evaluation/eval_semseg.py)

#### mIoU (mean Intersection over Union)

**정의**: 모든 클래스에 대한 IoU의 평균값

**계산 방법:**
```python
# 각 클래스별로 계산
for class_i in classes:
    TP[i] = (pred == i) & (gt == i) & valid  # True Positive
    FP[i] = (pred == i) & (gt != i) & valid  # False Positive
    FN[i] = (pred != i) & (gt == i) & valid  # False Negative

    # IoU (Jaccard Index)
    IoU[i] = TP[i] / (TP[i] + FP[i] + FN[i])

# 모든 클래스의 평균
mIoU = mean(IoU)
```

**수식:**
```
IoU_i = |pred ∩ gt| / |pred ∪ gt|
mIoU = (1/C) * Σ IoU_i
```

**해석:**
- **범위**: 0 ~ 100% (또는 0 ~ 1)
- **높을수록 좋음**
- **의미**: 예측과 정답의 겹치는 영역 비율
- **50% 이상**: 좋은 성능
- **60% 이상**: 매우 좋은 성능 (PASCAL Context 기준)

**예시 (로그에서):**
```
Semantic Segmentation mIoU: 52.223
```

**클래스별 IoU:**
- 각 클래스(background, aeroplane, bicycle, ...)마다 개별 IoU 계산
- 희소한 클래스도 동일한 가중치
- 클래스 불균형에 robust

---

### 5.2 Human Part Segmentation (human_parts) Metrics

코드 위치: [evaluation/eval_human_parts.py](evaluation/eval_human_parts.py)

#### mIoU (mean Intersection over Union)

**정의**: Human parts의 모든 부위 클래스에 대한 IoU 평균

**클래스**: 7개 (background, head, torso, upper arm, lower arm, upper leg, lower leg)

**계산 방법:**
```python
# Semantic Segmentation과 동일한 방식
for part_i in [0, 1, 2, 3, 4, 5, 6]:
    TP[i] = (pred == i) & (gt == i) & valid
    FP[i] = (pred == i) & (gt != i) & valid
    FN[i] = (pred != i) & (gt == i) & valid

    IoU[i] = TP[i] / (TP[i] + FP[i] + FN[i])

mIoU = mean(IoU)
```

**특징:**
- **사람이 있는 이미지만 평가**: 사람이 없으면 모두 background
- **부위별 IoU**: 각 신체 부위의 정확도 개별 측정
- **불균형**: 일부 부위(head, torso)는 많지만, 팔/다리는 상대적으로 적음

**예시 (로그에서):**
```
Human Parts mIoU: 28.3530
```

**해석:**
- Human parts는 semseg보다 어려움 (fine-grained)
- **30% 이상**: 좋은 성능
- **40% 이상**: 매우 좋은 성능

---

### 5.3 Saliency Detection (sal) Metrics

코드 위치: [evaluation/eval_sal.py](evaluation/eval_sal.py), [evaluation/jaccard.py](evaluation/jaccard.py)

#### 1. mIoU (mean Intersection over Union)

**정의**: 여러 threshold에 대한 IoU의 최대값

**계산 방법:**
```python
# 15개 threshold에 대해 평가
thresholds = linspace(0.2, 0.9, 15)

for threshold in thresholds:
    pred_binary = (pred > threshold)  # Binarize prediction

    # Jaccard Index (IoU for binary)
    intersection = (pred_binary & gt).sum()
    union = (pred_binary | gt).sum()
    IoU[threshold] = intersection / union

# 모든 이미지에 대해 평균 후 최대값
mIoU = max(mean(IoU, axis=images))
```

**특징:**
- Binary segmentation이므로 단일 IoU
- Multiple thresholds: 예측 확률을 어디서 자를지 결정
- Best threshold를 자동으로 찾음

**예시 (로그에서):**
```
mIoU: 52.223
```

#### 2. maxF (maximum F-measure)

**정의**: Precision-Recall 균형을 나타내는 F1-score의 최대값

**계산 방법:**
```python
for threshold in thresholds:
    pred_binary = (pred > threshold)

    # Precision: 예측한 것 중 맞은 비율
    TP = (pred_binary & gt).sum()
    FP = (pred_binary & ~gt).sum()
    Precision = TP / (TP + FP)

    # Recall: 정답 중 찾아낸 비율
    FN = (~pred_binary & gt).sum()
    Recall = TP / (TP + FN)

    # F-measure (F1-score)
    F[threshold] = 2 * Precision * Recall / (Precision + Recall)

# 최대 F-measure
maxF = max(F)
```

**수식:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * P * R / (P + R)
```

**해석:**
- **Precision**: 예측의 정확성 (false alarm 낮음)
- **Recall**: 검출의 완전성 (miss 낮음)
- **F1**: 두 지표의 조화 평균
- **높을수록 좋음**

**예시 (로그에서):**
```
maxF: 68.520
```

**Threshold별 평가 이유:**
- Saliency는 확률값 출력 (0~1 또는 0~255)
- 적절한 threshold를 찾아야 binary mask로 변환 가능
- 15개 threshold (0.2, 0.267, ..., 0.9) 시도하여 최적 성능 측정

---

### 5.4 Edge Detection (edge) Metrics

코드 위치: [evaluation/eval_edge.py](evaluation/eval_edge.py)

#### 훈련 중 Metric: Loss

**정의**: BalancedCrossEntropyLoss의 평균값

```python
# EdgeMeter는 validation 중 loss만 기록
loss_avg = total_loss / num_pixels
```

**특징:**
- 훈련/검증 중에는 단순히 loss 추적
- 실제 edge quality는 seism으로 평가 (추론 후 별도 실행)

#### 추론 후 Metric: SEISM Benchmark

**정의**: Structured Edge Detector 벤치마크

**SEISM Metrics:**
- **ODS (Optimal Dataset Scale)**: 전체 데이터셋에 대한 최적 threshold의 F-score
- **OIS (Optimal Image Scale)**: 이미지별 최적 threshold의 F-score 평균
- **AP (Average Precision)**: Precision-Recall 곡선 아래 면적

**계산 방법:**
```
1. 예측한 edge map 저장
2. SEISM 디렉토리로 복사
3. Matlab 스크립트 실행 (non-maximum suppression 등)
4. Precision-Recall 곡선 계산
5. ODS, OIS, AP 출력
```

**해석:**
- **ODS F-score**: 전체 데이터셋에 대한 edge quality
- **70% 이상**: 좋은 성능
- Edge detection은 pixel-level이 아닌 구조적 평가 필요

---

### 5.5 Surface Normal Estimation (normals) Metrics

코드 위치: [evaluation/eval_normals.py](evaluation/eval_normals.py)

#### 1. Mean Angular Error

**정의**: 예측 normal과 GT normal 사이의 평균 각도 오차

**계산 방법:**
```python
# Normal 정규화
pred_norm = pred / ||pred||
gt_norm = gt / ||gt||

# 내적으로 cosine 계산
cos_angle = sum(pred_norm * gt_norm, dim=channel)
cos_angle = clip(cos_angle, -1, 1)  # Numerical stability

# 각도로 변환 (degree)
angle_error = arccos(cos_angle) * 180 / π

# 유효한 픽셀에 대해 평균
mean_error = mean(angle_error[valid_mask])
```

**수식:**
```
error_i = arccos(pred_i · gt_i) * 180/π
mean = (1/N) * Σ error_i
```

**해석:**
- **범위**: 0° ~ 180°
- **낮을수록 좋음**
- **15° 이하**: 매우 좋은 성능
- **20° 이하**: 좋은 성능
- **30° 이상**: 개선 필요

#### 2. Median Angular Error

**정의**: 각도 오차의 중간값 (median)

**계산 방법:**
```python
median_error = median(angle_error[valid_mask])
```

**해석:**
- **Mean vs Median**:
  - Mean: 극단적 오차에 민감
  - Median: Outlier에 robust
- Median이 Mean보다 낮으면 일부 큰 오차가 평균을 올림

#### 3. RMSE (Root Mean Squared Error)

**정의**: 각도 오차의 제곱근 평균

**계산 방법:**
```python
rmse = sqrt(mean(angle_error^2))
```

**해석:**
- 큰 오차에 더 큰 페널티
- Mean보다 항상 크거나 같음
- 안정성 지표

#### 4. Percentage within Thresholds

**정의**: 특정 각도 이하 오차를 가진 픽셀의 비율

**계산 방법:**
```python
# 11.25도 이하 비율
percentage_11.25 = (angle_error < 11.25).sum() / total * 100

# 22.5도 이하 비율
percentage_22.5 = (angle_error < 22.5).sum() / total * 100

# 30도 이하 비율
percentage_30 = (angle_error < 30).sum() / total * 100
```

**해석:**
- **높을수록 좋음**
- **30° 내 비율 > 80%**: 좋은 성능
- **22.5° 내 비율 > 60%**: 매우 좋은 성능
- **11.25° 내 비율 > 40%**: 탁월한 성능

**예시 출력:**
```
Results for Surface Normal Estimation
mean       12.34
median     10.56
rmse       15.67
11.25      45.23%
22.5       68.91%
30         82.34%
```

---

### 5.6 Multi-Task Performance

코드 위치: [evaluation/evaluate_utils.py:45-70](evaluation/evaluate_utils.py#L45-L70)

#### MTL Performance Metric

**정의**: Multi-Task Learning의 Single-Task 대비 상대적 성능 향상

**계산 방법:**
```python
def calculate_multi_task_performance(mtl_dict, stl_dict):
    delta_performance = 0

    for task in tasks:
        mtl_score = mtl_dict[task]
        stl_score = stl_dict[task]  # Single-task baseline

        if task in ['semseg', 'sal', 'human_parts']:
            # mIoU: 높을수록 좋음
            delta = (mtl_score['mIoU'] - stl_score['mIoU']) / stl_score['mIoU']
            delta_performance += delta

        elif task == 'edge':
            # odsF: 높을수록 좋음
            delta = (mtl_score['odsF'] - stl_score['odsF']) / stl_score['odsF']
            delta_performance += delta

        elif task == 'normals':
            # mean error: 낮을수록 좋음 (부호 반대)
            delta = -(mtl_score['mean'] - stl_score['mean']) / stl_score['mean']
            delta_performance += delta

    return delta_performance / num_tasks
```

**수식:**
```
ΔP_task = (MTL - STL) / STL  (higher is better)
ΔP_task = -(MTL - STL) / STL  (lower is better)

MTL_Perf = (1/T) * Σ ΔP_task
```

**해석:**
- **> 0**: MTL이 STL보다 평균적으로 우수
- **< 0**: MTL이 STL보다 평균적으로 열등
- **> 5%**: 의미 있는 성능 향상
- **> 10%**: 큰 성능 향상

**예시:**
```
Task               STL mIoU    MTL mIoU    Δ%
semseg             60.5        62.9        +3.97%
human_parts        35.2        40.6        +15.34%
sal                70.1        76.1        +8.56%

MTL Performance: +9.29% (average improvement)
```

### 5.7 Metric 요약표

| Task | Primary Metric | Range | Higher/Lower is Better | Good Performance |
|------|---------------|-------|----------------------|------------------|
| semseg | mIoU | 0-100% | Higher | > 50% |
| human_parts | mIoU | 0-100% | Higher | > 30% |
| sal | mIoU | 0-100% | Higher | > 50% |
| sal | maxF | 0-100% | Higher | > 70% |
| edge | ODS F-score | 0-100% | Higher | > 70% |
| normals | Mean Error | 0-180° | Lower | < 20° |
| normals | % within 30° | 0-100% | Higher | > 80% |

---

## 참고 자료

- 코드 베이스: M3ViT (Multi-task, Multi-scale Vision Transformer with Mixture-of-Experts)
- 주요 파일:
  - [models/vision_transformer_moe.py](models/vision_transformer_moe.py): VisionTransformerMoE 구현
  - [models/custom_moe_layer.py](models/custom_moe_layer.py): FMoETransformerMLP 구현
  - [models/gate_funs/noisy_gate_vmoe.py](models/gate_funs/noisy_gate_vmoe.py): Gate network 구현
  - [utils/helpers.py](utils/helpers.py): Weight loading utilities
  - [losses/loss_functions.py](losses/loss_functions.py): Loss function 구현
  - [evaluation/evaluate_utils.py](evaluation/evaluate_utils.py): Evaluation utilities
