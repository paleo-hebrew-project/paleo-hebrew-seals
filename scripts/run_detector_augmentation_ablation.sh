#!/usr/bin/env bash
# Augmentation ablation for detector: noaug vs lightaug vs strongaug.
# Uses the same model list and sequential synth->real schedule.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIGS=(
  "${ROOT}/configs/experiments/sweep_detector_stageb_noaug.yaml"
  "${ROOT}/configs/experiments/sweep_detector_stageb_lightaug.yaml"
  "${ROOT}/configs/experiments/sweep_detector_stageb_base.yaml"
)

# Use one or more backbones for ablation. Keep this shorter than full sweep.
MODELS=(
  yolo11m.pt
  yolo26m.pt
  rtdetr-l.pt
)

GPUS=(0 1 2 3 4 5 6 7)
i=0
for cfg in "${CONFIGS[@]}"; do
  for m in "${MODELS[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    echo "[det-aug-ablation] GPU $gpu <- $(basename "$cfg") + $m"
    CUDA_VISIBLE_DEVICES="$gpu" python -m paleo_ocr.experiments.run_experiment \
      --config "$cfg" --detector-model "$m" &
    i=$((i + 1))
  done
done
wait
echo "Detector augmentation ablation finished."
