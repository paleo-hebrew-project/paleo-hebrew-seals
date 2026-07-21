#!/usr/bin/env bash
# Run classifier YAML configs in parallel (edit CONFIGS / GPUS).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIGS=(
  "${ROOT}/configs/experiments/classifier_convnext_synthb_val_real.yaml"
  "${ROOT}/configs/experiments/classifier_weighted_mix.yaml"
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
echo "All classifier jobs finished."
