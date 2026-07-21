#!/usr/bin/env bash
# Parallel classifier sweep — phase 1 only (real finetune), using completed phase-0 synth weights.
# Mirrors scripts/run_parallel_detector_sweep_phase1.sh, but delegates to the classifier backbone sweep.
#
# Prerequisite: phase 0 must already exist for each backbone (same experiment_name as the full sweep).
# Optional override for all jobs:
#   CLASSIFIER_INIT_WEIGHTS=/abs/path/best.pt bash scripts/run_parallel_classifier_backbones_phase1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_START_PHASE=1

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
