#!/usr/bin/env bash
# Parallel classifier backbone sweep — 40 epochs on real data only,
# using the backup row_id-group split manifest.
#
# Uses dedicated configs / experiment names, so it does NOT conflict with the
# earlier real-only runs on the row-overlap manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_real_only_40e_base_group_split.yaml"
export CLASSIFIER_SWINV2_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_real_only_40e_swinv2_256_group_split.yaml"
export CLASSIFIER_REGIME="real_only"

# Make sure inherited sequential-resume env does not accidentally affect the real-only run.
unset CLASSIFIER_START_PHASE
unset CLASSIFIER_END_PHASE
unset CLASSIFIER_INIT_WEIGHTS
unset CLASSIFIER_RESUME_WEIGHTS

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
