#!/usr/bin/env bash
# Parallel detector backbone sweep (Ultralytics YOLO + RT-DETR + YOLO26 family).
# Uses one base YAML + --detector-model per job (see paleo_ocr/experiments/run_experiment.py).
# Sequential two-phase training is defined in the base YAML (default: sweep_detector_stageb_base.yaml).
# Stage-A ablations should use scripts/run_parallel_detector_stagea_sweep.sh so the whole
# backbone grid is evaluated on the same train/val split.
#
# From repo root:
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_paper \
#   DETECTOR_NUM_GPUS=8 \
#     bash scripts/run_parallel_detector_sweep.sh
#
#   DETECTOR_SWEEP_PROFILE=full \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_full \
#   DETECTOR_NUM_GPUS=8 \
#     bash scripts/run_parallel_detector_sweep.sh
#
# Group-split Hebrew manifest (two-phase, same as stageb but data paths + split filters):
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_stageb_group_split.yaml \
#   DETECTOR_EXTRA_CONFIGS=none \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_group_split \
#   DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
#     bash scripts/run_parallel_detector_sweep.sh
#
# Custom subset:
#   DETECTOR_MODELS="yolov8m.pt yolo11m.pt rtdetr-l.pt" DETECTOR_GPUS="0 1" \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_subset \
#     bash scripts/run_parallel_detector_sweep.sh
#
# GPU selection:
#   DETECTOR_GPUS="0 1 2 3"  # explicit list, highest priority
#   DETECTOR_NUM_GPUS=4      # first N GPUs from nvidia-smi
#   unset both               # auto-detect all GPUs via nvidia-smi; fallback: 0..7
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_stageb_base.yaml}"
DETECTOR_SWEEP_PROFILE="${DETECTOR_SWEEP_PROFILE:-paper}"

# Paper subset: compact family comparison for the article.
MODELS_PAPER=(
  yolov8m.pt
  yolov8x.pt
  yolo11m.pt
  yolo11x.pt
  rtdetr-l.pt
  yolo26m.pt
  yolo26x.pt
)

# Full benchmark grid.
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

# Explicit model override wins over profile.
if [[ -n "${DETECTOR_MODELS:-}" ]]; then
  read -r -a MODELS <<< "${DETECTOR_MODELS}"
fi

_detect_gpus() {
  local -a detected=()
  local gpu
  if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r gpu; do
      gpu="${gpu//[[:space:]]/}"
      [[ -n "${gpu}" ]] && detected+=("${gpu}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  fi

  if [[ "${#detected[@]}" -eq 0 ]]; then
    detected=(0 1 2 3 4 5 6 7)
  fi

  printf '%s\n' "${detected[@]}"
}

if [[ -n "${DETECTOR_GPUS:-}" ]]; then
  read -r -a GPUS <<< "${DETECTOR_GPUS}"
else
  GPUS=()
  while IFS= read -r gpu; do
    [[ -n "${gpu}" ]] && GPUS+=("${gpu}")
  done < <(_detect_gpus)

  if [[ -n "${DETECTOR_NUM_GPUS:-}" ]]; then
    if ! [[ "${DETECTOR_NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${DETECTOR_NUM_GPUS}" -lt 1 ]]; then
      echo "Invalid DETECTOR_NUM_GPUS=${DETECTOR_NUM_GPUS} (must be a positive integer)." >&2
      exit 1
    fi
    if [[ "${DETECTOR_NUM_GPUS}" -gt "${#GPUS[@]}" ]]; then
      echo "[detector][warn] DETECTOR_NUM_GPUS=${DETECTOR_NUM_GPUS}, but only detected ${#GPUS[@]} GPU(s); using ${#GPUS[@]}." >&2
    fi
    GPUS=( "${GPUS[@]:0:${DETECTOR_NUM_GPUS}}" )
  fi
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPU ids available (DETECTOR_GPUS is empty / auto-detect failed)." >&2
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

# Optional shared YOLO export cache root override. Defaults to the YAML value.
# On slow FUSE/NFS shares, set DETECTOR_SHARED_CACHE_ROOT=/tmp/paleo_yolo_cache
# (local overlay) for ~16x faster dataset export + Ultralytics cache priming.
_shared_cache_suffix=""
if [[ -n "${DETECTOR_SHARED_CACHE_ROOT:-}" ]]; then
  mkdir -p "${DETECTOR_SHARED_CACHE_ROOT}"
  _shared_cache_suffix=" --detector-shared-cache-root $(printf '%q' "${DETECTOR_SHARED_CACHE_ROOT}")"
  echo "[detector] shared cache root override: ${DETECTOR_SHARED_CACHE_ROOT}"
fi

echo "[detector] base config: ${BASE}"
echo "[detector] profile: ${DETECTOR_SWEEP_PROFILE}, models: ${#MODELS[@]}, gpus: ${GPUS[*]}"

# Concurrency cap (RAM/FUSE control). Default: one job per GPU (queued).
# Set DETECTOR_MAX_PARALLEL=0 for the old "all at once" behavior.
MAX_PARALLEL="${DETECTOR_MAX_PARALLEL:-${#GPUS[@]}}"
echo "[detector] max parallel: ${MAX_PARALLEL} (DETECTOR_MAX_PARALLEL; 0 = all at once)"

pids=()
job_descs=()
job_idx=0
num_gpus="${#GPUS[@]}"

_wait_for_slot() {
  [[ "${MAX_PARALLEL}" -gt 0 ]] || return 0
  while true; do
    local alive=0
    for p in "${pids[@]:-}"; do
      if [[ -n "${p}" ]] && kill -0 "${p}" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    (( alive < MAX_PARALLEL )) && return
    wait -n 2>/dev/null || true
  done
}

launch_job() {
  local gpu="$1"
  local kind="$2"
  local target="$3"
  local cmd="$4"
  local desc="${kind}:${target}@gpu${gpu}"

  echo "[detector] GPU ${gpu} <- ${kind}:${target}"
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
  _wait_for_slot
  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "model" \
    "${m}" \
    "python -m paleo_ocr.experiments.run_experiment --config \"${BASE}\" --detector-model \"${m}\"${_shared_cache_suffix}"
done

for cfg in "${EXTRA_CONFIGS[@]}"; do
  if [[ "${cfg}" =~ ^/ ]]; then
    cfg_path="${cfg}"
  else
    cfg_path="${ROOT}/${cfg}"
  fi
  if [[ ! -f "${cfg_path}" ]]; then
    echo "[detector][warn] missing extra config: ${cfg_path}" >&2
    continue
  fi

  _wait_for_slot
  gpu="${GPUS[$((job_idx % num_gpus))]}"
  launch_job \
    "${gpu}" \
    "extra" \
    "$(basename "${cfg_path}")" \
    "python -m paleo_ocr.experiments.run_experiment --config \"${cfg_path}\"${_shared_cache_suffix}"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No detector jobs were launched." >&2
  exit 1
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[detector][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} detector job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Detector sweep finished (${#pids[@]} jobs). Next: pick best weights, then run scripts/run_parallel_classifier_backbones.sh"
