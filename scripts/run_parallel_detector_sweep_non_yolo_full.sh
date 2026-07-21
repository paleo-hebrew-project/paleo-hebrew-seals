#!/usr/bin/env bash
# One-command full non-YOLO detector sweep.
# Runs the official Ultralytics RT-DETR family pack in two steps:
#   1) phase 0 (synth pretrain)
#   2) phase 1 (real finetune)
#
# Default models come from the delegated scripts:
#   - rtdetr-l.yaml
#   - rtdetr-x.yaml
#   - rtdetr-resnet50.yaml
#   - rtdetr-resnet101.yaml
#
# Same pipeline as phase scripts; default BASE is sweep_detector_stageb_base.yaml (sequential).
# Logs: DETECTOR_SWEEP_LOG_DIR is the root; per-phase logs go to .../phase0 and .../phase1
# (override with DETECTOR_PHASE0_LOG_DIR / DETECTOR_PHASE1_LOG_DIR).
#
# From repo root:
#   DETECTOR_GPUS="0 1 2 3" \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_non_yolo_full \
#     bash scripts/run_parallel_detector_sweep_non_yolo_full.sh
#
# Group-split manifest (two-phase YAML with train/validation split filters):
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_stageb_group_split.yaml \
#   DETECTOR_GPUS="0 1 2 3" \
#   DETECTOR_SWEEP_LOG_DIR=logs/detector_sweep_non_yolo_full_group_split \
#     bash scripts/run_parallel_detector_sweep_non_yolo_full.sh
#
# RT-DETR *.yaml training can hit NaN with default lr0; if needed, copy the stageb YAML and set
# train_overrides.lr0 (e.g. 5e-4) and amp: false, then pass it as DETECTOR_BASE_CONFIG.
#
# Optional overrides passed through:
#   - DETECTOR_BASE_CONFIG
#   - DETECTOR_MODELS
#   - DETECTOR_GPUS
#   - DETECTOR_INIT_WEIGHTS (used only by phase 1 if explicitly set)
#   - DETECTOR_PHASE0_LOG_DIR / DETECTOR_PHASE1_LOG_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PHASE0_SCRIPT="${ROOT}/scripts/run_parallel_detector_sweep_phase0_non_yolo.sh"
PHASE1_SCRIPT="${ROOT}/scripts/run_parallel_detector_sweep_phase1_non_yolo.sh"
BASE="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_stageb_base.yaml}"

LOG_ROOT="${DETECTOR_SWEEP_LOG_DIR:-logs/detector_sweep_non_yolo_full}"
PHASE0_LOG_DIR="${DETECTOR_PHASE0_LOG_DIR:-${LOG_ROOT}/phase0}"
PHASE1_LOG_DIR="${DETECTOR_PHASE1_LOG_DIR:-${LOG_ROOT}/phase1}"

mkdir -p "${PHASE0_LOG_DIR}" "${PHASE1_LOG_DIR}"

if [[ ! -f "${PHASE0_SCRIPT}" ]]; then
  echo "Missing script: ${PHASE0_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${PHASE1_SCRIPT}" ]]; then
  echo "Missing script: ${PHASE1_SCRIPT}" >&2
  exit 1
fi

echo "[detector-non-yolo-full] base config: ${BASE}"
echo "[detector-non-yolo-full] models: ${DETECTOR_MODELS:-default RT-DETR family pack}"
echo "[detector-non-yolo-full] gpus: ${DETECTOR_GPUS:-0 1 2 3}"
echo "[detector-non-yolo-full] phase0 logs: ${PHASE0_LOG_DIR}"
echo "[detector-non-yolo-full] phase1 logs: ${PHASE1_LOG_DIR}"

export DETECTOR_BASE_CONFIG="${BASE}"

echo "[detector-non-yolo-full] starting phase 0"
export DETECTOR_SWEEP_LOG_DIR="${PHASE0_LOG_DIR}"
bash "${PHASE0_SCRIPT}"

echo "[detector-non-yolo-full] phase 0 complete; starting phase 1"
export DETECTOR_SWEEP_LOG_DIR="${PHASE1_LOG_DIR}"
bash "${PHASE1_SCRIPT}"

echo "[detector-non-yolo-full] full non-YOLO sweep finished"
