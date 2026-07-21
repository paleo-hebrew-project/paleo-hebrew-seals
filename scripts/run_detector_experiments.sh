#!/usr/bin/env bash
# Run detector YAML configs in parallel on multiple GPUs (edit CONFIGS and CUDA_VISIBLE_DEVICES).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIGS=(
  "${ROOT}/configs/experiments/detector_yolo_synth_b_val_real.yaml"
  "${ROOT}/configs/experiments/detector_weighted_synth_real.yaml"
)

GPUS=(0 1 2 3)
i=0
for cfg in "${CONFIGS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "CUDA $gpu -> $cfg"
  CUDA_VISIBLE_DEVICES="$gpu" python -m paleo_ocr.experiments.run_experiment --config "$cfg" &
  i=$((i + 1))
done
wait
echo "All detector jobs finished."
