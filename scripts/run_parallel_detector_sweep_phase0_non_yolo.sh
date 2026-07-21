#!/usr/bin/env bash
# Parallel detector sweep — phase 0 only (synth pretrain), official non-YOLO RT-DETR family.
#
# This is the "single pack" for direct backbone comparison without experimental branches.
# Default models:
#   - rtdetr-l.yaml
#   - rtdetr-x.yaml
#   - rtdetr-resnet50.yaml
#   - rtdetr-resnet101.yaml
#
# These are official Ultralytics RT-DETR configs and run through the same repo pipeline as the
# other detector sweeps. Override with DETECTOR_MODELS if you want a custom subset.
#
# Usage:
#   DETECTOR_GPUS="0 1 2 3" \
#   DETECTOR_SWEEP_LOG_DIR="logs/detector_sweep_non_yolo_phase0" \
#   bash scripts/run_parallel_detector_sweep_phase0_non_yolo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_stageb_base.yaml}"

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

echo "[detector-p0-non-yolo] base config: ${BASE}"
echo "[detector-p0-non-yolo] models: ${#MODELS[@]} (${MODELS[*]}), gpus: ${GPUS[*]}"
echo "[detector-p0-non-yolo] phases: only index 0 (pretrain_synth); --detector-end-phase 1"

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

  echo "[detector-p0-non-yolo] GPU ${gpu} <- ${kind}:${target}"
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
    "python -m paleo_ocr.experiments.run_experiment --config $(printf '%q' "${BASE}") --detector-model $(printf '%q' "${m}") --detector-start-phase 0 --detector-end-phase 1"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No detector jobs were launched." >&2
  exit 1
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[detector-p0-non-yolo][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} detector job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Detector phase-0 (non-YOLO) sweep finished (${#pids[@]} jobs)."
