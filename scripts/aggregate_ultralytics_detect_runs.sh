#!/usr/bin/env bash
# Aggregate Ultralytics training folders under scripts/runs/detect/... into one CSV + val-set sanity check.
# Override DETECT_RUNS_ROOT or pass extra args to the Python module.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

DETECT_RUNS_ROOT="${DETECT_RUNS_ROOT:-${ROOT}/scripts/runs/detect/runs/detect}"
OUT_CSV="${OUT_CSV:-${ROOT}/scripts/runs/detect/yolo_runs_summary.csv}"

exec python -m paleo_ocr.experiments.aggregate_yolo_runs \
  --root "$DETECT_RUNS_ROOT" \
  --repo-root "$ROOT" \
  --out-csv "$OUT_CSV" \
  "$@"
