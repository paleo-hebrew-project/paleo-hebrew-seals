#!/usr/bin/env bash
# Parallel classifier backbone sweep — 40 epochs on real data only.
# Uses separate experiment names / configs, so it does NOT conflict with the
# sequential 20 synth + 20 real sweep outputs.
#
# Common usage:
#   bash scripts/run_parallel_classifier_backbones_real_only_40e.sh
#   CLASSIFIER_GPUS="0 1 2 3" CLASSIFIER_EXCLUDE_MODELS="model_convnext_small model_convnext_tiny" \
#     bash scripts/run_parallel_classifier_backbones_real_only_40e.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_BASE_CONFIG="${CLASSIFIER_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_real_only_40e_base.yaml}"
export CLASSIFIER_SWINV2_BASE_CONFIG="${CLASSIFIER_SWINV2_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_real_only_40e_swinv2_256.yaml}"
export CLASSIFIER_REGIME="real_only"

# Make sure inherited sequential-resume env does not accidentally affect the real-only run.
unset CLASSIFIER_START_PHASE
unset CLASSIFIER_END_PHASE
unset CLASSIFIER_INIT_WEIGHTS
unset CLASSIFIER_RESUME_WEIGHTS

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
