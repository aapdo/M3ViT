#!/usr/bin/env bash
set -euo pipefail

# 20-epoch hyperparameter sweep for MoE pretraining
# - init: controlled by INIT_MODE (scratch | deit_warm_start)
# - distillation: hard
# - eval every 5 epochs
# - each run logs to a separate W&B run (project: pretrain_test by default)
# - optional 3-way split:
#   SPLIT_ID=1 -> runs 1,4,7
#   SPLIT_ID=2 -> runs 2,5,8
#   SPLIT_ID=3 -> runs 3,6,9

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

CONFIG="${CONFIG:-pretrain/configs/deit_moe_small.yaml}"
CONFIG_PATH="${CONFIG_PATH:-configs/path_env.yml}"
DATASET_NAME="${DATASET_NAME:-ImageNet1K}"

EPOCHS="${EPOCHS:-20}"
EVAL_FREQ="${EVAL_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"

HARD_ALPHA="${HARD_ALPHA:-0.5}"
INIT_MODE="${INIT_MODE:-scratch}"   # scratch | deit_warm_start
WARM_MOE_MLP_RATIO="${WARM_MOE_MLP_RATIO:-1.0}"

USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-pretrain_test}"

# 0 = run all 9; 1/2/3 = run one split only (1,4,7 / 2,5,8 / 3,6,9)
SPLIT_ID="${SPLIT_ID:-0}"
# Global run index to start from (1..9). Useful for partial reruns.
START_FROM="${START_FROM:-1}"

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M)}"
GROUP_NAME="${GROUP_NAME:-}"
if [[ -z "${GROUP_NAME}" ]]; then
  if [[ "${INIT_MODE}" == "deit_warm_start" ]]; then
    GROUP_NAME="deit_small_warmstart_hard_grid20"
  else
    GROUP_NAME="deit_small_scratch_hard_grid20"
  fi
fi
BASE_OUT="${BASE_OUT:-output_dir/pretrain/${GROUP_NAME}}"
BASE_LOG="${BASE_LOG:-logs/pretrain/${GROUP_NAME}}"

mkdir -p "${BASE_OUT}" "${BASE_LOG}"

# Router grid (user-requested).
declare -a ROUTER_GRID=(
  "0.01,0.5"
  "0.02,0.5"
  "0.02,0.3"
)

declare -a AUG_GRID=(
  "0.8,1.0,0.1"
  "0.4,0.2,0.05"
  "0.2,0.0,0.0"
)

# Priority order by aug setting id:
# a3 -> a2 -> a1
declare -a AUG_PRIORITY=(2 1 0)

echo "== Sweep config =="
echo "RUN_TAG=${RUN_TAG}"
echo "CONFIG=${CONFIG}"
echo "EPOCHS=${EPOCHS}, EVAL_FREQ=${EVAL_FREQ}, SAVE_FREQ=${SAVE_FREQ}"
echo "GROUP_NAME=${GROUP_NAME}"
echo "INIT_MODE=${INIT_MODE}"
if [[ "${INIT_MODE}" == "deit_warm_start" ]]; then
  echo "WARM_MOE_MLP_RATIO=${WARM_MOE_MLP_RATIO}"
fi
echo "BASE_OUT=${BASE_OUT}"
echo "BASE_LOG=${BASE_LOG}"
echo "SPLIT_ID=${SPLIT_ID} (0=all, 1=1/4/7, 2=2/5/8, 3=3/6/9)"
echo "START_FROM=${START_FROM} (global index; 1..9)"
echo

if ! [[ "${SPLIT_ID}" =~ ^[0-3]$ ]]; then
  echo "[ERROR] SPLIT_ID must be one of 0,1,2,3. got='${SPLIT_ID}'" >&2
  exit 2
fi

if ! [[ "${START_FROM}" =~ ^[0-9]+$ ]] || (( START_FROM < 1 || START_FROM > 9 )); then
  echo "[ERROR] START_FROM must be an integer in [1, 9]. got='${START_FROM}'" >&2
  exit 2
fi

case "${INIT_MODE}" in
  scratch|deit_warm_start)
    ;;
  *)
    echo "[ERROR] INIT_MODE must be one of: scratch, deit_warm_start. got='${INIT_MODE}'" >&2
    exit 2
    ;;
esac

run_idx=0
run_sel=0
for a_i in "${AUG_PRIORITY[@]}"; do
  for r_i in "${!ROUTER_GRID[@]}"; do
    IFS=',' read -r moe_cv_weight vmoe_noisy_std <<< "${ROUTER_GRID[$r_i]}"
    IFS=',' read -r mixup cutmix smoothing <<< "${AUG_GRID[$a_i]}"

    run_idx=$((run_idx + 1))
    if (( run_idx < START_FROM )); then
      continue
    fi
    split_bucket=$(( ((run_idx - 1) % 3) + 1 ))
    if [[ "${SPLIT_ID}" != "0" && "${split_bucket}" != "${SPLIT_ID}" ]]; then
      continue
    fi
    run_sel=$((run_sel + 1))

    exp_id="r$((r_i + 1))_a$((a_i + 1))"
    run_name="${GROUP_NAME}_${exp_id}_${RUN_TAG}"
    out_dir="${BASE_OUT}/${run_name}"
    log_file="${BASE_LOG}/${run_name}.log"

    CMD=(
      torchrun --nproc_per_node="${NPROC_PER_NODE}" --module pretrain.train
      --config "${CONFIG}"
      --config-path "${CONFIG_PATH}"
      --dataset-name "${DATASET_NAME}"
      --epochs "${EPOCHS}"
      --eval-freq "${EVAL_FREQ}"
      --save-freq "${SAVE_FREQ}"
      --output-dir "${out_dir}"
      --deit-init-mode "${INIT_MODE}"
      --distillation-type hard
      --distilled true
      --distillation-alpha "${HARD_ALPHA}"
      --moe-top-k 4
      --moe-data-distributed true
      --moe-cv-weight "${moe_cv_weight}"
      --vmoe-noisy-std "${vmoe_noisy_std}"
      --mixup "${mixup}"
      --cutmix "${cutmix}"
      --smoothing "${smoothing}"
      --wandb-resume never
    )

    if [[ "${INIT_MODE}" == "deit_warm_start" ]]; then
      CMD+=(--moe-mlp-ratio "${WARM_MOE_MLP_RATIO}")
    fi

    if [[ "${USE_WANDB,,}" == "true" ]]; then
      CMD+=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-name "${run_name}")
    fi

    echo "[$(date +%H:%M:%S)] (global ${run_idx}/9, selected ${run_sel}/$([[ "${SPLIT_ID}" == "0" ]] && echo 9 || echo 3)) START ${run_name}"
    echo "  router: moe_cv_weight=${moe_cv_weight}, vmoe_noisy_std=${vmoe_noisy_std}"
    echo "  aug: mixup=${mixup}, cutmix=${cutmix}, smoothing=${smoothing}"
    echo "  split_bucket=${split_bucket}"
    echo "  out_dir=${out_dir}"
    echo "  log=${log_file}"
    echo "  cmd: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} PYTHONPATH=$(pwd) ${CMD[*]}"
    echo

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="$(pwd)" "${CMD[@]}" 2>&1 | tee "${log_file}"
  done
done

echo "Completed selected runs: ${run_sel} (SPLIT_ID=${SPLIT_ID})"
