#!/usr/bin/env bash
set -euo pipefail

# Compare Dense DeiT eval results between:
# - cache mode  : indexed ImageFolder cache
# - direct mode : torchvision ImageFolder directory scan
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
#   bash pretrain/compare_dense_deit_cache_vs_direct_eval.sh

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

CONFIG="${CONFIG:-pretrain/configs/deit_dense_small.yaml}"
CONFIG_PATH="${CONFIG_PATH:-configs/path_env.yml}"
DATASET_NAME="${DATASET_NAME:-ImageNet1K}"
MODEL="${MODEL:-deit_small}"
DISTILLED="${DISTILLED:-true}"
RESUME="${RESUME:-deit://warm-start}"

RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M)}"
BASE_OUT="${BASE_OUT:-output_dir/pretrain_dense_deit/eval_cache_vs_direct}"
BASE_LOG="${BASE_LOG:-logs/pretrain/dense_deit_eval_compare}"

# Optional extra args for Dense_DeiT (space-separated, shell-split).
# Example: DENSE_DEIT_EXTRA_ARGS="--batch-size 128 --workers 16"
DENSE_DEIT_EXTRA_ARGS="${DENSE_DEIT_EXTRA_ARGS:-}"

mkdir -p "${BASE_OUT}" "${BASE_LOG}"

SUMMARY_FILE="${BASE_LOG}/compare_${RUN_TAG}.txt"
CSV_FILE="${BASE_LOG}/compare_${RUN_TAG}.csv"

RUN_ACC1=""
RUN_ACC5=""
RUN_LOSS=""
RUN_SEC=""
RUN_OUT_DIR=""
RUN_LOG_FILE=""

extract_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys

path = sys.argv[1]
acc1 = None
acc5 = None
loss = None

pat_main = re.compile(r"\* Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)\s+loss\s+([0-9.]+)")
pat_eval_only = re.compile(r"Eval only: Acc@1=([0-9.]+)\s+Acc@5=([0-9.]+)")

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat_main.search(line)
        if m:
            acc1 = float(m.group(1))
            acc5 = float(m.group(2))
            loss = float(m.group(3))
        m2 = pat_eval_only.search(line)
        if m2:
            acc1 = float(m2.group(1))
            acc5 = float(m2.group(2))

if acc1 is None or acc5 is None:
    print("nan,nan,nan")
else:
    print(f"{acc1:.6f},{acc5:.6f},{'nan' if loss is None else f'{loss:.6f}'}")
PY
}

run_one_mode() {
  local mode="$1"
  local out_dir="${BASE_OUT}/${RUN_TAG}/${mode}"
  local log_file="${BASE_LOG}/${RUN_TAG}_${mode}.log"
  mkdir -p "${out_dir}"

  local cmd=(
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --module pretrain.Dense_DeiT
    --config "${CONFIG}"
    --config-path "${CONFIG_PATH}"
    --dataset-name "${DATASET_NAME}"
    --model "${MODEL}"
    --distilled "${DISTILLED}"
    --resume "${RESUME}"
    --eval
    --imagenet-loader-mode "${mode}"
    --output-dir "${out_dir}"
  )

  if [[ -n "${DENSE_DEIT_EXTRA_ARGS}" ]]; then
    # Intentional shell splitting for user-provided extra CLI flags.
    # shellcheck disable=SC2206
    local extra_arr=( ${DENSE_DEIT_EXTRA_ARGS} )
    cmd+=("${extra_arr[@]}")
  fi

  echo "[$(date +%H:%M:%S)] START mode=${mode}"
  echo "  out_dir=${out_dir}"
  echo "  log_file=${log_file}"
  echo "  cmd: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} PYTHONPATH=$(pwd) ${cmd[*]}"

  local start_sec
  local end_sec
  start_sec="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="$(pwd)" "${cmd[@]}" 2>&1 | tee "${log_file}"
  end_sec="$(date +%s)"

  local parsed
  parsed="$(extract_metrics "${log_file}")"
  IFS=',' read -r RUN_ACC1 RUN_ACC5 RUN_LOSS <<< "${parsed}"
  RUN_SEC="$((end_sec - start_sec))"
  RUN_OUT_DIR="${out_dir}"
  RUN_LOG_FILE="${log_file}"
}

echo "== Dense DeiT cache vs direct eval compare =="
echo "RUN_TAG=${RUN_TAG}"
echo "CONFIG=${CONFIG}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "MODEL=${MODEL}"
echo "DISTILLED=${DISTILLED}"
echo "RESUME=${RESUME}"
echo "BASE_OUT=${BASE_OUT}"
echo "BASE_LOG=${BASE_LOG}"
if [[ -n "${DENSE_DEIT_EXTRA_ARGS}" ]]; then
  echo "DENSE_DEIT_EXTRA_ARGS=${DENSE_DEIT_EXTRA_ARGS}"
fi
echo

run_one_mode "cache"
CACHE_ACC1="${RUN_ACC1}"
CACHE_ACC5="${RUN_ACC5}"
CACHE_LOSS="${RUN_LOSS}"
CACHE_SEC="${RUN_SEC}"
CACHE_OUT="${RUN_OUT_DIR}"
CACHE_LOG="${RUN_LOG_FILE}"

run_one_mode "direct"
DIRECT_ACC1="${RUN_ACC1}"
DIRECT_ACC5="${RUN_ACC5}"
DIRECT_LOSS="${RUN_LOSS}"
DIRECT_SEC="${RUN_SEC}"
DIRECT_OUT="${RUN_OUT_DIR}"
DIRECT_LOG="${RUN_LOG_FILE}"

read -r DELTA_ACC1 DELTA_ACC5 DELTA_LOSS <<EOF
$(python - "${CACHE_ACC1}" "${DIRECT_ACC1}" "${CACHE_ACC5}" "${DIRECT_ACC5}" "${CACHE_LOSS}" "${DIRECT_LOSS}" <<'PY'
import math
import sys

def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

def fmt(v):
    return "nan" if math.isnan(v) else f"{v:+.6f}"

cache_acc1 = to_float(sys.argv[1])
direct_acc1 = to_float(sys.argv[2])
cache_acc5 = to_float(sys.argv[3])
direct_acc5 = to_float(sys.argv[4])
cache_loss = to_float(sys.argv[5])
direct_loss = to_float(sys.argv[6])

print(fmt(direct_acc1 - cache_acc1), fmt(direct_acc5 - cache_acc5), fmt(direct_loss - cache_loss))
PY
)
EOF

{
  echo "mode,acc1,acc5,loss,elapsed_sec,out_dir,log_file"
  echo "cache,${CACHE_ACC1},${CACHE_ACC5},${CACHE_LOSS},${CACHE_SEC},${CACHE_OUT},${CACHE_LOG}"
  echo "direct,${DIRECT_ACC1},${DIRECT_ACC5},${DIRECT_LOSS},${DIRECT_SEC},${DIRECT_OUT},${DIRECT_LOG}"
  echo "delta_direct_minus_cache,${DELTA_ACC1},${DELTA_ACC5},${DELTA_LOSS},$((DIRECT_SEC - CACHE_SEC)),,"
} > "${CSV_FILE}"

{
  echo "== Dense DeiT cache vs direct comparison =="
  echo "RUN_TAG=${RUN_TAG}"
  echo
  echo "[cache]"
  echo "acc1=${CACHE_ACC1} acc5=${CACHE_ACC5} loss=${CACHE_LOSS} elapsed_sec=${CACHE_SEC}"
  echo "out_dir=${CACHE_OUT}"
  echo "log=${CACHE_LOG}"
  echo
  echo "[direct]"
  echo "acc1=${DIRECT_ACC1} acc5=${DIRECT_ACC5} loss=${DIRECT_LOSS} elapsed_sec=${DIRECT_SEC}"
  echo "out_dir=${DIRECT_OUT}"
  echo "log=${DIRECT_LOG}"
  echo
  echo "[delta: direct - cache]"
  echo "acc1=${DELTA_ACC1} acc5=${DELTA_ACC5} loss=${DELTA_LOSS} elapsed_sec=$((DIRECT_SEC - CACHE_SEC))"
  echo
  echo "csv=${CSV_FILE}"
} | tee "${SUMMARY_FILE}"

echo
echo "Done. Summary: ${SUMMARY_FILE}"
echo "Done. CSV: ${CSV_FILE}"
