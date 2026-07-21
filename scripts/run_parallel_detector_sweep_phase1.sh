#!/usr/bin/env bash
# Parallel detector sweep — phase 1 only (real finetune), using completed phase-0 weights.
# Same model grid as run_parallel_detector_sweep.sh, but each job runs:
#   python -m paleo_ocr.experiments.run_experiment ... --detector-start-phase 1
# Phase-0 checkpoints are resolved automatically (runs/detect vs runs/detect/runs/detect, _ph0 vs _ph02).
#
# Prerequisite: phase 0 must already exist for each backbone (same experiment_name / run_name as full sweep).
#
# Common usage:
#   bash scripts/run_parallel_detector_sweep_phase1.sh
#   DETECTOR_SWEEP_PROFILE=paper DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
#     DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_paper_phase1 bash scripts/run_parallel_detector_sweep_phase1.sh
#
# Group-split Hebrew manifest (notebooks/manifest_hebrew_unambiguous_group_split_rowid.jsonl), from repo root:
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_phase1_group_split \
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_stageb_group_split.yaml \
#   DETECTOR_GPUS="0 1 2 3 4 5 6 7" \
#     bash scripts/run_parallel_detector_sweep_phase1.sh
#
# Phase-1 (finetune) early stopping: override Ultralytics patience vs YAML (e.g. group_split default 15):
#   DETECTOR_PHASE1_PATIENCE=90   -> patience 90 epochs
#   DETECTOR_PHASE1_PATIENCE=0    -> off (same as off/none/disable)
#   (omit variable to use YAML)
#
# Phase-1 mosaic: Ultralytics normally turns off mosaic for the last N epochs ("Closing dataloader mosaic"):
#   DETECTOR_PHASE1_CLOSE_MOSAIC=0    -> keep mosaic for all finetune epochs (or off/none/disable)
#   DETECTOR_PHASE1_CLOSE_MOSAIC=50   -> last 50 epochs without mosaic (YAML-style)
#   (omit variable to use YAML)
#
# Explicit weights override (optional, normally not needed):
#   DETECTOR_INIT_WEIGHTS=/abs/path/best.pt bash scripts/run_parallel_detector_sweep_phase1.sh
#   (same file would be used for every job — use only for a single-model debug)
#
# Extra YAMLs are not launched here: the usual control (stage A only) is single-regime, not sequential.
# Run scripts/run_parallel_detector_sweep.sh if you need that job alongside the backbone grid.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_stageb_base.yaml}"
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

# Optional: same init weights for every job (debug only). Usually omit — Python resolves phase-0 best.pt per model.
_init_weights_suffix() {
  if [[ -n "${DETECTOR_INIT_WEIGHTS:-}" ]]; then
    printf ' --detector-init-weights %q' "${DETECTOR_INIT_WEIGHTS}"
  fi
}

_phase1_patience_suffix() {
  if [[ -n "${DETECTOR_PHASE1_PATIENCE:-}" ]]; then
    printf ' --detector-phase1-patience %q' "${DETECTOR_PHASE1_PATIENCE}"
  fi
}

_phase1_close_mosaic_suffix() {
  if [[ -n "${DETECTOR_PHASE1_CLOSE_MOSAIC:-}" ]]; then
    printf ' --detector-phase1-close-mosaic %q' "${DETECTOR_PHASE1_CLOSE_MOSAIC}"
  fi
}

_shared_cache_suffix() {
  if [[ -n "${DETECTOR_SHARED_CACHE_ROOT:-}" ]]; then
    printf ' --detector-shared-cache-root %q' "${DETECTOR_SHARED_CACHE_ROOT}"
  fi
}

_batch_suffix() {
  if [[ -n "${DETECTOR_BATCH:-}" ]]; then
    printf ' --detector-batch %q' "${DETECTOR_BATCH}"
  fi
}

if [[ -n "${DETECTOR_SWEEP_LOG_DIR:-}" ]]; then
  mkdir -p "${DETECTOR_SWEEP_LOG_DIR}"
fi

echo "[detector-phase1] base config: ${BASE}"
echo "[detector-phase1] profile: ${DETECTOR_SWEEP_PROFILE}, models: ${#MODELS[@]}, gpus: ${GPUS[*]}"
echo "[detector-phase1] --detector-start-phase 1 (finetune_real only; expects phase 0 weights on disk)"
if [[ -n "${DETECTOR_PHASE1_PATIENCE:-}" ]]; then
  echo "[detector-phase1] phase-1 patience override: ${DETECTOR_PHASE1_PATIENCE} (0/off/none/disable = early stop off)"
fi
if [[ -n "${DETECTOR_PHASE1_CLOSE_MOSAIC:-}" ]]; then
  echo "[detector-phase1] phase-1 close_mosaic override: ${DETECTOR_PHASE1_CLOSE_MOSAIC} (0 = never close mosaic)"
fi
if [[ -n "${DETECTOR_SHARED_CACHE_ROOT:-}" ]]; then
  echo "[detector-phase1] shared cache root override: ${DETECTOR_SHARED_CACHE_ROOT}"
fi
if [[ -n "${DETECTOR_BATCH:-}" ]]; then
  echo "[detector-phase1] batch override: ${DETECTOR_BATCH}"
fi

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

  echo "[detector-phase1] GPU ${gpu} <- ${kind}:${target}"
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
    "python -m paleo_ocr.experiments.run_experiment --config $(printf '%q' "${BASE}") --detector-model $(printf '%q' "${m}") --detector-start-phase 1$(_init_weights_suffix)$(_phase1_patience_suffix)$(_phase1_close_mosaic_suffix)$(_shared_cache_suffix)$(_batch_suffix)"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No detector jobs were launched." >&2
  exit 1
fi

failures=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "${pid}"; then
    echo "[detector-phase1][fail] ${job_descs[$i]}" >&2
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo "ERROR: ${failures} detector job(s) failed (out of ${#pids[@]})." >&2
  exit 1
fi

echo "Detector phase-1 sweep finished (${#pids[@]} jobs). Next: aggregate_runs / classifier backbones as needed."
