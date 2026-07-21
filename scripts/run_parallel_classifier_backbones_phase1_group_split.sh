#!/usr/bin/env bash
# Parallel classifier sweep — phase 1 only (real finetune) on the backup row_id-group split.
# Requires phase 0 from the same group-split configs / experiment names.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_stageb_base_group_split.yaml"
export CLASSIFIER_SWINV2_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_swinv2_256_group_split.yaml"
export CLASSIFIER_START_PHASE=1

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
