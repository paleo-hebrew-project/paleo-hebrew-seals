#!/usr/bin/env bash
# Parallel classifier backbone sweep on the backup row_id-group split:
# full sequential regime = 20 synth + 20 real.
#
# Uses dedicated configs / experiment names, so it does NOT conflict with the
# earlier row-overlap runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_stageb_base_group_split.yaml"
export CLASSIFIER_SWINV2_BASE_CONFIG="${ROOT}/configs/experiments/sweep_classifier_swinv2_256_group_split.yaml"

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
