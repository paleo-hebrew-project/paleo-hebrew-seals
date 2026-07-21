#!/usr/bin/env bash
# Parallel classifier sweep — phase 0 only (synth pretrain).
# Convenience wrapper over run_parallel_classifier_backbones.sh with CLASSIFIER_END_PHASE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export CLASSIFIER_END_PHASE=1

exec bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh"
