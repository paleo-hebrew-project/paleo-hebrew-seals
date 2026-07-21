#!/usr/bin/env bash
# Parallel detector Stage-A ablation sweep.
# Uses the same model grid machinery as run_parallel_detector_sweep.sh, but defaults to
# configs/experiments/sweep_detector_stagea_base.yaml and disables legacy extra configs.
#
# Logs: per-job logs always go to a timestamped dir under logs/ (survives terminal
# disconnect). The sweep itself runs under `setsid` so it survives SIGHUP when the
# terminal/SSH closes; this wrapper blocks until the sweep finishes (so you can paste
# several commands sequentially without GPU oversubscription).
#
# From repo root:
#   DETECTOR_NUM_GPUS=8 bash scripts/run_parallel_detector_stagea_sweep.sh
#
# Use 4 cards:
#   DETECTOR_NUM_GPUS=4 bash scripts/run_parallel_detector_stagea_sweep.sh
#
# Group split:
#   DETECTOR_BASE_CONFIG=configs/experiments/sweep_detector_stagea_group_split.yaml \
#     bash scripts/run_parallel_detector_stagea_sweep.sh
#
# Disable the auto RAM monitor:
#   RAM_MONITOR=0 bash scripts/run_parallel_detector_stagea_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DETECTOR_BASE_CONFIG="${DETECTOR_BASE_CONFIG:-${ROOT}/configs/experiments/sweep_detector_stagea_base.yaml}"
export DETECTOR_EXTRA_CONFIGS="${DETECTOR_EXTRA_CONFIGS:-none}"

# Stage A pretrain exports the full 200k-image synth manifest into a YOLO layout.
# On the FUSE share this is ~16x slower than local disk; default the shared
# export cache to local /tmp so the build finishes in minutes, not hours.
# Override with DETECTOR_SHARED_CACHE_ROOT=/some/path; set to "" to use the YAML value.
if [[ -z "${DETECTOR_SHARED_CACHE_ROOT+x}" ]]; then
  export DETECTOR_SHARED_CACHE_ROOT="/tmp/paleo_yolo_cache"
fi

# Default per-job log dir (timestamped) so output persists on disk.
if [[ -z "${DETECTOR_SWEEP_LOG_DIR+x}" ]]; then
  export DETECTOR_SWEEP_LOG_DIR="${ROOT}/logs/detector_stagea_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${DETECTOR_SWEEP_LOG_DIR}"
WRAPPER_LOG="${DETECTOR_SWEEP_LOG_DIR}/_sweep.log"

# Pre-flight RAM check.
_ram_avail_mb=$(free -m | awk '/^Mem:/ {print $7}')
_ram_total_mb=$(free -m | awk '/^Mem:/ {print $2}')
{
  echo "[stagea-detector] $(date -Iseconds) host=$(hostname)"
  echo "[stagea-detector] log dir: ${DETECTOR_SWEEP_LOG_DIR}"
  echo "[stagea-detector] shared cache: ${DETECTOR_SHARED_CACHE_ROOT:-<yaml>}"
  echo "[stagea-detector] RAM: ${_ram_avail_mb} MB avail / ${_ram_total_mb} MB total"
} | tee -a "${WRAPPER_LOG}"

RAM_MONITOR="${RAM_MONITOR:-1}"
MONITOR_PID=""
if [[ "${RAM_MONITOR}" == "1" ]]; then
  bash "${ROOT}/scripts/monitor_ram.sh" "${DETECTOR_SWEEP_LOG_DIR}/ram_monitor.log" \
    >"${DETECTOR_SWEEP_LOG_DIR}/ram_monitor.out" 2>&1 &
  MONITOR_PID=$!
  echo "[stagea-detector] ram monitor -> ${DETECTOR_SWEEP_LOG_DIR}/ram_monitor.log (pid ${MONITOR_PID})" | tee -a "${WRAPPER_LOG}"
fi

cleanup() {
  [[ -n "${MONITOR_PID}" ]] && kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[stagea-detector] tail live: tail -f ${WRAPPER_LOG}" | tee -a "${WRAPPER_LOG}"

# Run the base sweep under setsid so it survives terminal disconnect (SIGHUP),
# but wait on it here so this wrapper blocks (keeps sequential GPU usage safe).
setsid bash "${ROOT}/scripts/run_parallel_detector_sweep.sh" >>"${WRAPPER_LOG}" 2>&1 &
SWEEP_PID=$!
echo "[stagea-detector] sweep pid ${SWEEP_PID}" | tee -a "${WRAPPER_LOG}"

wait "${SWEEP_PID}"
rc=$?
echo "[stagea-detector] sweep finished with exit code ${rc}" | tee -a "${WRAPPER_LOG}"
exit "${rc}"
