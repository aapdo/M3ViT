#!/usr/bin/env bash
set -euo pipefail

# Auto-pilot:
# 1) Run short screening for each experiment candidate.
# 2) Rank by screening eval acc1.
# 3) Resume top-K candidates to full training.
#
# Defaults are tuned for your current setup (2 GPUs, deit_moe_small config).
# Override via env vars if needed.
#
# Example:
#   CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 bash pretrain/auto_screen_then_full.sh

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

CONFIG="${CONFIG:-pretrain/configs/deit_moe_small.yaml}"
CONFIG_PATH="${CONFIG_PATH:-configs/path_env.yml}"
DATASET_NAME="${DATASET_NAME:-ImageNet1K}"

SCREEN_EPOCHS="${SCREEN_EPOCHS:-10}"
SCREEN_EVAL_FREQ="${SCREEN_EVAL_FREQ:-2}"
FULL_EPOCHS="${FULL_EPOCHS:-300}"
FULL_EVAL_FREQ="${FULL_EVAL_FREQ:-10}"
FULL_SAVE_FREQ="${FULL_SAVE_FREQ:-10}"
TOPK="${TOPK:-2}"

SOFT_ALPHA="${SOFT_ALPHA:-0.5}"
SOFT_TAU="${SOFT_TAU:-1.0}"

USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-pretrain}"
RUN_NO_DISTILL="${RUN_NO_DISTILL:-false}"  # true -> add warm_none/scratch_none

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M)}"   # 9 chars => train.py won't append timestamp again
GROUP_NAME="${GROUP_NAME:-deit_small_auto}"
BASE_OUT="${BASE_OUT:-output_dir/pretrain/${GROUP_NAME}}"
BASE_LOG="${BASE_LOG:-logs/pretrain/${GROUP_NAME}}"
SCORE_CSV="${BASE_LOG}/screen_scores_${RUN_TAG}.csv"

mkdir -p "${BASE_OUT}" "${BASE_LOG}"

declare -a CANDIDATES=(
  "warm_hard"
  "warm_soft"
  "scratch_hard"
  "scratch_soft"
)

if [[ "${RUN_NO_DISTILL,,}" == "true" ]]; then
  CANDIDATES+=("warm_none" "scratch_none")
fi

echo "== Auto Screen+Full =="
echo "RUN_TAG=${RUN_TAG}"
echo "CANDIDATES=${CANDIDATES[*]}"
echo "SCREEN_EPOCHS=${SCREEN_EPOCHS}, SCREEN_EVAL_FREQ=${SCREEN_EVAL_FREQ}, FULL_EPOCHS=${FULL_EPOCHS}, FULL_EVAL_FREQ=${FULL_EVAL_FREQ}, TOPK=${TOPK}"
echo "BASE_OUT=${BASE_OUT}"
echo "BASE_LOG=${BASE_LOG}"
echo

append_mode_args() {
  local exp_id="$1"
  case "${exp_id}" in
    warm_*)
      CMD+=(--deit-init-mode deit_warm_start --moe-mlp-ratio 1.0)
      ;;
    scratch_*)
      CMD+=(--deit-init-mode scratch)
      ;;
    *)
      echo "[ERROR] Unknown init family in experiment id: ${exp_id}" >&2
      exit 2
      ;;
  esac
}

append_distill_args() {
  local exp_id="$1"
  case "${exp_id}" in
    *_none)
      CMD+=(--distillation-type none --distilled false)
      ;;
    *_hard)
      CMD+=(--distillation-type hard --distilled true)
      ;;
    *_soft)
      CMD+=(--distillation-type soft --distilled true --distillation-alpha "${SOFT_ALPHA}" --distillation-tau "${SOFT_TAU}")
      ;;
    *)
      echo "[ERROR] Unknown distillation type in experiment id: ${exp_id}" >&2
      exit 2
      ;;
  esac
}

append_wandb_args() {
  local run_name="$1"
  if [[ "${USE_WANDB,,}" == "true" ]]; then
    CMD+=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-name "${run_name}")
  fi
}

run_train() {
  local exp_id="$1"
  local phase="$2"           # screen | full
  local epochs="$3"
  local eval_freq="$4"
  local save_freq="$5"
  local out_dir="$6"
  local log_file="$7"
  local resume_dir="${8:-}"

  CMD=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --module pretrain.train
    --config "${CONFIG}"
    --config-path "${CONFIG_PATH}"
    --dataset-name "${DATASET_NAME}"
    --epochs "${epochs}"
    --eval-freq "${eval_freq}"
    --save-freq "${save_freq}"
    --output-dir "${out_dir}"
  )

  append_mode_args "${exp_id}"
  append_distill_args "${exp_id}"
  append_wandb_args "${exp_id}_${phase}_${RUN_TAG}"

  if [[ -n "${resume_dir}" ]]; then
    CMD+=(--resume "${resume_dir}")
  fi

  echo "[$(date +%H:%M:%S)] START ${phase}: ${exp_id}"
  echo "  out_dir=${out_dir}"
  echo "  log=${log_file}"
  echo "  cmd: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} PYTHONPATH=$(pwd) ${CMD[*]}"
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="$(pwd)" "${CMD[@]}" 2>&1 | tee "${log_file}"
}

has_latest_checkpoint() {
  local dir="$1"
  [[ -n "${dir}" ]] || return 1
  [[ -d "${dir}" ]] || return 1
  compgen -G "${dir}/checkpoint_latest*.pth" >/dev/null
}

extract_latest_eval_acc1() {
  local log_json="$1"
  python - "$log_json" <<'PY'
import json, sys
path = sys.argv[1]
latest = -1.0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("eval_performed") and ("test_acc1" in row):
            try:
                latest = float(row["test_acc1"])
            except Exception:
                pass
print(f"{latest:.6f}")
PY
}

echo "exp_id,screen_acc1,screen_out_dir" > "${SCORE_CSV}"

for exp_id in "${CANDIDATES[@]}"; do
  screen_out_dir="${BASE_OUT}/${exp_id}/screen/${RUN_TAG}"
  screen_log="${BASE_LOG}/${exp_id}_screen_${RUN_TAG}.log"
  mkdir -p "${screen_out_dir}"

  run_train "${exp_id}" "screen" "${SCREEN_EPOCHS}" "${SCREEN_EVAL_FREQ}" "${SCREEN_EPOCHS}" "${screen_out_dir}" "${screen_log}"

  screen_json="${screen_out_dir}/log.txt"
  if [[ ! -f "${screen_json}" ]]; then
    echo "[ERROR] Screening log not found: ${screen_json}" >&2
    exit 3
  fi
  screen_acc1="$(extract_latest_eval_acc1 "${screen_json}")"
  echo "${exp_id},${screen_acc1},${screen_out_dir}" | tee -a "${SCORE_CSV}"
  echo
done

echo "== Screening scores =="
column -s, -t "${SCORE_CSV}" || cat "${SCORE_CSV}"
echo

TOP_FILE="$(mktemp)"
trap 'rm -f "${TOP_FILE}"' EXIT
tail -n +2 "${SCORE_CSV}" | sort -t, -k2,2gr | head -n "${TOPK}" > "${TOP_FILE}"

echo "== Top-${TOPK} selected for full run =="
cat "${TOP_FILE}"
echo

while IFS=',' read -r exp_id screen_acc1 screen_out_dir; do
  full_out_dir="${BASE_OUT}/${exp_id}/full/${RUN_TAG}"
  full_log="${BASE_LOG}/${exp_id}_full_${RUN_TAG}.log"
  mkdir -p "${full_out_dir}"

  resume_dir=""
  if has_latest_checkpoint "${full_out_dir}"; then
    resume_dir="${full_out_dir}"
    echo "[RESUME] Found full checkpoint. Resume full run from: ${resume_dir}"
  elif has_latest_checkpoint "${screen_out_dir}"; then
    resume_dir="${screen_out_dir}"
    echo "[RESUME] No full checkpoint found. Resume from screen run: ${resume_dir}"
  else
    echo "[RESUME] No checkpoint found in full/screen directories. Start full run from scratch."
  fi

  run_train "${exp_id}" "full" "${FULL_EPOCHS}" "${FULL_EVAL_FREQ}" "${FULL_SAVE_FREQ}" "${full_out_dir}" "${full_log}" "${resume_dir}"
done < "${TOP_FILE}"

echo
echo "Done. Summary:"
echo "  Screening CSV: ${SCORE_CSV}"
echo "  Logs: ${BASE_LOG}"
echo "  Outputs: ${BASE_OUT}"
