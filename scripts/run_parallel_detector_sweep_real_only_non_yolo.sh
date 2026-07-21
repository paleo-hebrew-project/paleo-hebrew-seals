#!/usr/bin/env bash
# Parallel detector sweep — real-only training for official Ultralytics RT-DETR YAML backbones
# (same idea as scripts/run_parallel_detector_sweep_phase0_non_yolo.sh / phase1_non_yolo.sh, but single-stage
# real subset: no synth phase and no --detector-start-phase).
#
# Base config: sweep_detector_real_only_100e_base.yaml (regime: real_only, 100 epochs).
# Default models match the non-YOLO "single pack": rtdetr-l/x + ResNet50/101 configs.
#
# Usage:
#   DETECTOR_GPUS="0 1 2 3" \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_real_only_non_yolo \
#   bash scripts/run_parallel_detector_sweep_real_only_non_yolo.sh
#   DETECTOR_MODELS="rtdetr-l.yaml" DETECTOR_GPUS="0" bash scripts/run_parallel_detector_sweep_real_only_non_yolo.sh
#
# Group-split manifest + RT-DETR-safe hyperparams (lr0, amp off), from repo root:
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_real_only_non_yolo_group_split \
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_real_only_group_split_rtdetr.yaml \
#   DETECTOR_GPUS="0 1 2 3" \
#     bash scripts/run_parallel_detector_sweep_real_only_non_yolo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_real_only_100e_base.yaml}"

MODELS_DEFAULT=(
  rtdetr-l.yaml
  rtdetr-x.yaml
  rtdetr-resnet50.yaml
  rtdetr-resnet101.yaml
)

MODELS=( "${MODELS_DEFAULT[@]}" )
if [[ -n "${DETECTOR_MODELS:-}" ]]; then
  read -r -a MODELS <<< "${DETECTOR_MODELS}"
fi

if [[ -n "${DETECTOR_GPUS:-}" ]]; then
  read -r -a GPUS <<< "${DETECTOR_GPUS}"
else
  GPUS=(0 1 2 3)
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPU ids provided (DETECTOR_GPUS is empty)." >&2
  exit 1
fi

if [[ -n "${DETECTOR_SWEEP_LOG_DIR:-}" ]]; then
  mkdir -p "${DETECTOR_SWEEP_LOG_DIR}"
fi

echo "[detector-real-only-non-yolo] base config: ${BASE}"
echo "[detector-real-only-non-yolo] models: ${#MODELS[@]} (${MODELS[*]}), gpus: ${GPUS[*]}"

pids=()
job_descs=()
job_idx=0
num_gpus="${#GPUS[@]}"

launch_job() {
  local gpu="$1"
  local kind="$2"
  local target="$3"
  local cmd="$4"
  local desc="${kind}:${target}@gpu${gpu}"

  echo "[detector-real-only-non-yolo] GPU ${gpu} <- ${kind}:${target}"
  if [[ -n "${DETECTOR_SWEEP_LOG_DIR:-}" ]]; then
    local safe_target
    safe_target="$(echo "${target}" | tr '/ ' '__')"
    local log_file="${DETECTOR_SWEEP_LOG_DIR}/$(printf '%03d' "${job_idx}")_${kind}_${safe_target}.log"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}"
    ) >"${log_file}" 2>&1 &
    echo -e "$(printf '%03d' "${job_idx}")\t${desc}\t${log_file}" >> "${DETECTOR_SWEEP_LOG_DIR}/jobs.tsv"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" &
  fi

  pids+=($!)
  job_descs+=("${desc}")
  job_idx=$((job_idx + 1))
}

for m in "${MODELS[@]}"; do
  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "model" \
    "${m}" \
    "python -m paleo_ocr.experiments.run_experiment --config \"${BASE}\" --detector-model \"${m}\""
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No detector jobs were launched." >&2
  exit 1
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[detector-real-only-non-yolo][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} detector job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Real-only non-YOLO detector sweep finished (${#pids[@]} jobs)."
