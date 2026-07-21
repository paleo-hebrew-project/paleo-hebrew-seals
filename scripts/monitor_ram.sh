#!/usr/bin/env bash
# Lightweight system RAM sampler. Appends one line per interval to a log file,
# plus the top RAM-consuming processes. Use it to diagnose whether the Jupyter
# server (or training jobs) are being OOM-killed.
#
# Usage:
#   bash scripts/monitor_ram.sh logs/ram_monitor.log            # 10s default
#   RAM_MONITOR_INTERVAL=5 bash scripts/monitor_ram.sh logs/ram_monitor.log
#   RAM_MONITOR_TOPN=8 bash scripts/monitor_ram.sh logs/ram_monitor.log
#
# Tail it live:
#   tail -f logs/ram_monitor.log
#
# Alert threshold (MB avail below which a warning is emitted):
#   RAM_MONITOR_ALERT_MB=8192 bash scripts/monitor_ram.sh logs/ram_monitor.log
set -euo pipefail

LOG="${1:-logs/ram_monitor.log}"
INTERVAL="${RAM_MONITOR_INTERVAL:-10}"
TOPN="${RAM_MONITOR_TOPN:-6}"
ALERT_MB="${RAM_MONITOR_ALERT_MB:-8192}"

mkdir -p "$(dirname "$LOG")"

{
  echo "# ram_monitor start $(date -Iseconds) interval=${INTERVAL}s alert<${ALERT_MB}MB host=$(hostname)"
  echo "# epoch total_mb used_mb avail_mb swap_used_mb alert top_procs(rss)"
} >> "$LOG"

while true; do
  avail=$(free -m | awk '/^Mem:/ {print $7}')
  total=$(free -m | awk '/^Mem:/ {print $2}')
  used=$(free -m | awk '/^Mem:/ {print $3}')
  swap_used=$(free -m | awk '/^Swap:/ {print $3}')
  avail="${avail:-0}"; total="${total:-0}"; used="${used:-0}"; swap_used="${swap_used:-0}"
  alert="ok"
  if [[ "${avail}" -lt "${ALERT_MB}" ]]; then
    alert="LOW-RAM"
  fi
  top=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk -v n="$TOPN" 'NR>1 && NR<=n+1 {printf "%s:%dMB ", $2, $1}')
  printf '%s %s %s %s %s %s %s\n' "$(date +%s)" "$total" "$used" "$avail" "$swap_used" "$alert" "$top" >> "$LOG"
  sleep "$INTERVAL"
done
