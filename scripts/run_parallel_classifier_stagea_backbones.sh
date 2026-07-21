#!/usr/bin/env bash
# Parallel classifier Stage-A ablation sweep.
# Reuses run_parallel_classifier_backbones.sh with Stage-A base configs.
#
# Default: 20 epochs Stage A pretrain + 120 epochs real finetune.
# To reproduce the paper's other classifier regimes, override real finetune length:
#   CLASSIFIER_PHASE1_EPOCHS=20 bash scripts/run_parallel_classifier_stagea_backbones.sh
#   CLASSIFIER_PHASE1_EPOCHS=60 bash scripts/run_parallel_classifier_stagea_backbones.sh
#   CLASSIFIER_PHASE1_EPOCHS=120 bash scripts/run_parallel_classifier_stagea_backbones.sh
#
# Group split:
#   CLASSIFIER_BASE_CONFIG=configs/experiments/sweep_classifier_stagea_base_group_split.yaml \
#   CLASSIFIER_SWINV2_BASE_CONFIG=configs/experiments/sweep_classifier_swinv2_256_stagea_group_split.yaml \
#     bash scripts/run_parallel_classifier_stagea_backbones.sh
#
# Logs: per-job logs always go to a timestamped dir under logs/ (survives terminal
# disconnect). The sweep runs under `setsid` so it survives SIGHUP; this wrapper
# blocks until the sweep finishes (so you can paste several commands sequentially).
#
# Disable the auto RAM monitor: RAM_MONITOR=0 bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CLASSIFIER_BASE_CONFIG="${CLASSIFIER_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_stagea_base.yaml}"
export CLASSIFIER_SWINV2_BASE_CONFIG="${CLASSIFIER_SWINV2_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_classifier_swinv2_256_stagea.yaml}"

# Default per-job log dir (timestamped). The classifier base script already
# auto-defaults when this is unset, but we set it explicitly so the wrapper log
# and per-job logs share one directory.
if [[ -z "${CLASSIFIER_SWEEP_LOG_DIR+x}" ]]; then
  export CLASSIFIER_SWEEP_LOG_DIR="${ROOT}/logs/classifier_stagea_${CLASSIFIER_PHASE1_EPOCHS:-full}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${CLASSIFIER_SWEEP_LOG_DIR}"
WRAPPER_LOG="${CLASSIFIER_SWEEP_LOG_DIR}/_sweep.log"

_ram_avail_mb=$(free -m | awk '/^Mem:/ {print $7}')
_ram_total_mb=$(free -m | awk '/^Mem:/ {print $2}')
{
  echo "[stagea-classifier] $(date -Iseconds) host=$(hostname)"
  echo "[stagea-classifier] log dir: ${CLASSIFIER_SWEEP_LOG_DIR}"
  echo "[stagea-classifier] phase1 epochs: ${CLASSIFIER_PHASE1_EPOCHS:-<yaml default>}"
  echo "[stagea-classifier] RAM: ${_ram_avail_mb} MB avail / ${_ram_total_mb:-?} MB total"
} | tee -a "${WRAPPER_LOG}"

RAM_MONITOR="${RAM_MONITOR:-1}"
MONITOR_PID=""
if [[ "${RAM_MONITOR}" == "1" ]]; then
  bash "${ROOT}/scripts/monitor_ram.sh" "${CLASSIFIER_SWEEP_LOG_DIR}/ram_monitor.log" \
    >"${CLASSIFIER_SWEEP_LOG_DIR}/ram_monitor.out" 2>&1 &
  MONITOR_PID=$!
  echo "[stagea-classifier] ram monitor -> ${CLASSIFIER_SWEEP_LOG_DIR}/ram_monitor.log (pid ${MONITOR_PID})" | tee -a "${WRAPPER_LOG}"
fi

cleanup() {
  [[ -n "${MONITOR_PID}" ]] && kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[stagea-classifier] tail live: tail -f ${WRAPPER_LOG}" | tee -a "${WRAPPER_LOG}"

setsid bash "${ROOT}/scripts/run_parallel_classifier_backbones.sh" >>"${WRAPPER_LOG}" 2>&1 &
SWEEP_PID=$!
echo "[stagea-classifier] sweep pid ${SWEEP_PID}" | tee -a "${WRAPPER_LOG}"

wait "${SWEEP_PID}"
rc=$?
echo "[stagea-classifier] sweep finished with exit code ${rc}" | tee -a "${WRAPPER_LOG}"
exit "${rc}"
