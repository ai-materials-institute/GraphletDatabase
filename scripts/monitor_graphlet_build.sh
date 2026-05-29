#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-$REPO_ROOT/Graphlets/MP_cifs}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
TAIL_LINES="${TAIL_LINES:-15}"
ONCE="${ONCE:-0}"

STATE_PATH="$OUTPUT_DIR/graphlet_build_state.json"
MANIFEST_PATH="$OUTPUT_DIR/graphlet_build_manifest.json"
PROGRESS_LOG="$OUTPUT_DIR/graphlet_build_progress.log"
SUPERVISOR_LOG="$OUTPUT_DIR/full_run_supervisor.log"

print_block() {
  local title="$1"
  echo ""
  echo "===== $title ====="
}

print_state_summary() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(f"state: missing ({p})")
    sys.exit(0)

try:
    d = json.loads(p.read_text())
except Exception as exc:
    print(f"state: unreadable ({exc})")
    sys.exit(0)

status = d.get("status")
counts = d.get("counts", {})
parallel = d.get("parallel", {})
timing = d.get("timing", {})

print(f"state_file: {p}")
print(f"status: {status}")
print(
    "progress: "
    f"completed={counts.get('num_completed_total')}/{counts.get('num_input_cifs')} "
    f"built={counts.get('num_graphlets_built')} "
    f"skipped={counts.get('num_graphlets_skipped')} "
    f"failed={counts.get('num_graphlets_failed')} "
    f"pending={counts.get('num_pending_total')}"
)
print(
    "workers: "
    f"effective={parallel.get('max_workers_effective')} "
    f"running_now={parallel.get('running_now')} "
    f"queued={parallel.get('queued_not_submitted')} "
    f"cpu_cap={parallel.get('cpu_cap')}"
)
print(
    "timing: "
    f"elapsed={timing.get('elapsed_human')} "
    f"eta={timing.get('eta_human')} "
    f"rate_per_min={timing.get('rate_items_per_min')}"
)
PY
}

while true; do
  printf '\n[%s] monitor output_dir=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$OUTPUT_DIR"

  print_block "State"
  print_state_summary "$STATE_PATH"

  if [[ -f "$MANIFEST_PATH" ]]; then
    print_block "Manifest"
    python3 - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())
print(f"manifest_file: {p}")
print(
    "final_counts: "
    f"built={d.get('num_graphlets_built')} "
    f"skipped={d.get('num_graphlets_skipped')} "
    f"failed={d.get('num_graphlets_failed')}"
)
print(f"elapsed: {d.get('elapsed_human')}  throughput_per_min: {d.get('throughput_items_per_min')}")
PY
  fi

  if [[ -f "$PROGRESS_LOG" ]]; then
    print_block "Progress Log (tail)"
    tail -n "$TAIL_LINES" "$PROGRESS_LOG" || true
  fi

  if [[ -f "$SUPERVISOR_LOG" ]]; then
    print_block "Supervisor Log (tail)"
    tail -n "$TAIL_LINES" "$SUPERVISOR_LOG" || true
  fi

  if [[ "$ONCE" == "1" ]]; then
    exit 0
  fi

  sleep "$INTERVAL_SEC"
done
