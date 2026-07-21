#!/usr/bin/env bash
# Parallel detector sweep — real train / real val only (100 epochs by default in base YAML).
# Analogue of scripts/run_parallel_detector_sweep.sh, but uses:
#   configs/experiments/sweep_detector_real_only_100e_base.yaml
# which sets mixing.mode: single and detector.regime: real_only (no synth pretrain, no sequential phases).
#
# There is no separate "phase 1" launcher for this regime: training is a single stage. Compare with
# run_parallel_detector_sweep_phase1.sh (two-stage pipeline: resume at finetune_real only).
#
# Usage:
#   bash scripts/run_parallel_detector_sweep_real_only.sh
#   DETECTOR_SWEEP_PROFILE=full DETECTOR_GPUS="0 1 2 3" \
#     DETECTOR_SWEEP_LOG_DIR=logs/detector_real_only_paper \
#     bash scripts/run_parallel_detector_sweep_real_only.sh
#   DETECTOR_MODELS="yolo11m.pt rtdetr-l.pt" DETECTOR_GPUS="0 1" bash scripts/run_parallel_detector_sweep_real_only.sh
#
# Group-split manifest (same JSONL for real + val_manifest; split filters in YAML), from repo root:
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_real_only_group_split \
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_real_only_group_split.yaml \
#   DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
#     bash scripts/run_parallel_detector_sweep_real_only.sh
#
# Optional extra YAMLs (same mechanism as run_parallel_detector_sweep.sh), default none — the main sweep's
# default extra (stage-A ablation) is synth-based and not mixed into this grid unless you set it explicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_real_only_100e_base.yaml}"
DETECTOR_SWEEP_PROFILE="${DETECTOR_SWEEP_PROFILE:-paper}"

MODELS_PAPER=(
  yolov8m.pt
  yolov8x.pt
  yolo11m.pt
  yolo11x.pt
  rtdetr-l.pt
  yolo26m.pt
  yolo26x.pt
)

MODELS_FULL=(
  yolov8n.pt
  yolov8s.pt
  yolov8m.pt
  yolov8l.pt
  yolov8x.pt
  yolo11n.pt
  yolo11s.pt
  yolo11m.pt
  yolo11l.pt
  yolo11x.pt
  yolo26n.pt
  yolo26s.pt
  yolo26m.pt
  yolo26l.pt
  yolo26x.pt
  rtdetr-l.pt
  rtdetr-x.pt
)

case "${DETECTOR_SWEEP_PROFILE}" in
  paper) MODELS=( "${MODELS_PAPER[@]}" ) ;;
  full) MODELS=( "${MODELS_FULL[@]}" ) ;;
  *)
    echo "Invalid DETECTOR_SWEEP_PROFILE=${DETECTOR_SWEEP_PROFILE} (use paper or full)" >&2
    exit 1
    ;;
esac

if [[ -n "${DETECTOR_MODELS:-}" ]]; then
  read -r -a MODELS <<< "${DETECTOR_MODELS}"
fi

if [[ -n "${DETECTOR_GPUS:-}" ]]; then
  read -r -a GPUS <<< "${DETECTOR_GPUS}"
else
  GPUS=(0 1 2 3 4 5 6 7)
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPU ids provided (DETECTOR_GPUS is empty)." >&2
  exit 1
fi

if [[ -n "${DETECTOR_EXTRA_CONFIGS:-}" ]]; then
  if [[ "${DETECTOR_EXTRA_CONFIGS}" == "none" || "${DETECTOR_EXTRA_CONFIGS}" == "-" ]]; then
    EXTRA_CONFIGS=()
  else
    read -r -a EXTRA_CONFIGS <<< "${DETECTOR_EXTRA_CONFIGS}"
  fi
else
  EXTRA_CONFIGS=()
fi

if [[ -n "${DETECTOR_SWEEP_LOG_DIR:-}" ]]; then
  mkdir -p "${DETECTOR_SWEEP_LOG_DIR}"
fi

echo "[detector-real-only] base config: ${BASE}"
echo "[detector-real-only] profile: ${DETECTOR_SWEEP_PROFILE}, models: ${#MODELS[@]}, gpus: ${GPUS[*]}"

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

  echo "[detector-real-only] GPU ${gpu} <- ${kind}:${target}"
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

for cfg in "${EXTRA_CONFIGS[@]}"; do
  if [[ "${cfg}" =~ ^/ ]]; then
    cfg_path="${cfg}"
  else
    cfg_path="${ROOT}/${cfg}"
  fi
  if [[ ! -f "${cfg_path}" ]]; then
    echo "[detector-real-only][warn] missing extra config: ${cfg_path}" >&2
    continue
  fi

  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "extra" \
    "$(basename "${cfg_path}")" \
    "python -m paleo_ocr.experiments.run_experiment --config \"${cfg_path}\""
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No detector jobs were launched." >&2
  exit 1
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[detector-real-only][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} detector job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Real-only detector sweep finished (${#pids[@]} jobs)."
