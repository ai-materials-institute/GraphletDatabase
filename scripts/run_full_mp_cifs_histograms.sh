#!/usr/bin/env bash
#
# Supervisor for a full histogram build over a directory of graphlet JSON files.
#
# Stage 2 of the release pipeline. Run scripts/run_full_mp_cifs_graphlets.sh
# first to produce the graphlet JSONs, then this to produce the histograms.
# Together they reproduce the two-directory layout of the published dataset:
#
#     Graphlet_Database/Graphlets/            <- stage 1
#     Graphlet_Database/Graphlet_Histograms/  <- stage 2 (this script)
#
# Wraps the `build-histograms` CLI with a single-instance lock, bounded
# restarts, and stall detection, mirroring the graphlet supervisor. The build
# is resumable via --resume, so a restart skips histograms already written.
#
# Depends only on this repository: python3, `src/cli.py`, and `config/`.
#
# Usage:
#   scripts/run_full_mp_cifs_histograms.sh
#   GRAPHLET_DIR=/path/to/graphlets HISTOGRAM_OUT_DIR=/path/to/hists \
#     scripts/run_full_mp_cifs_histograms.sh
#   MAX_FILES=200 HISTOGRAM_OUT_DIR=/tmp/dryrun_hist \
#     scripts/run_full_mp_cifs_histograms.sh          # dry run
#
# Bin centers: all materials must share one binning for the histograms to be
# comparable. BIN_CENTERS_PATH is computed once from the corpus on the first
# attempt and reused on every restart. Do NOT reuse a bin-centers file that was
# derived from a different (e.g. subset) corpus -- set RECOMPUTE_BINS=1 to
# rebuild it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"

GRAPHLET_DIR="${GRAPHLET_DIR:-$REPO_ROOT/Graphlets/MP_cifs}"
HISTOGRAM_OUT_DIR="${HISTOGRAM_OUT_DIR:-$REPO_ROOT/Graphlet_Histograms/MP_cifs}"
PATTERN="${PATTERN:-*_graphlet.json}"

NUM_BINS="${NUM_BINS:-20}"
BIN_WIDTH_FACTOR="${BIN_WIDTH_FACTOR:-1.0}"
HIST_DENSITY="${HIST_DENSITY:-0}"
RECOMPUTE_BINS="${RECOMPUTE_BINS:-0}"
RESUME="${RESUME:-1}"

MAX_WORKERS="${MAX_WORKERS:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"   # -1 means unlimited retries
STALL_LIMIT="${STALL_LIMIT:-3}"      # stop after N non-improving failure counts; -1 disables
ALLOW_FAILED="${ALLOW_FAILED:-1}"    # 1: accept residual failures and finish; 0: retry until 0

# Dry-run support: build histograms for only the first MAX_FILES graphlet JSONs
# by staging a directory of symlinks. Unset or 0 means use GRAPHLET_DIR directly.
MAX_FILES="${MAX_FILES:-0}"

BIN_CENTERS_PATH="${BIN_CENTERS_PATH:-$HISTOGRAM_OUT_DIR/bin_centers.json}"
STATE_PATH="${STATE_PATH:-$HISTOGRAM_OUT_DIR/histogram_build_state.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$HISTOGRAM_OUT_DIR/histogram_build_manifest.json}"
PROGRESS_LOG="${PROGRESS_LOG:-$HISTOGRAM_OUT_DIR/histogram_build_progress.log}"
RUNNER_LOG="${RUNNER_LOG:-$HISTOGRAM_OUT_DIR/full_run_supervisor.log}"
LOCK_FILE="${LOCK_FILE:-$HISTOGRAM_OUT_DIR/full_run.lock}"

mkdir -p "$HISTOGRAM_OUT_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  local line="[$(timestamp)] $*"
  echo "$line"
  echo "$line" >> "$RUNNER_LOG"
}

STAGE_DIR=""

acquire_lock() {
  if [[ -f "$LOCK_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Another run is already active (pid=$old_pid, lock=$LOCK_FILE)." >&2
      exit 1
    fi
    rm -f "$LOCK_FILE"
  fi
  echo "$$" > "$LOCK_FILE"
}

cleanup() {
  rm -f "$LOCK_FILE"
  [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]] && rm -rf "$STAGE_DIR"
  return 0
}

trap cleanup EXIT INT TERM

acquire_lock

if [[ ! -d "$GRAPHLET_DIR" ]]; then
  echo "Graphlet directory not found: $GRAPHLET_DIR" >&2
  echo "Run scripts/run_full_mp_cifs_graphlets.sh first." >&2
  exit 1
fi

if [[ "$MAX_FILES" -gt 0 ]]; then
  STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/histogram_subset.XXXXXX")"
  n=0
  for f in "$GRAPHLET_DIR"/$PATTERN; do
    [[ -e "$f" ]] || continue
    ln -s "$f" "$STAGE_DIR/"
    n=$((n + 1))
    [[ "$n" -ge "$MAX_FILES" ]] && break
  done
  log "Dry run: staged $n graphlet symlinks in $STAGE_DIR"
  log "NOTE: bin centers will be derived from this subset, not the full corpus."
  GRAPHLET_DIR="$STAGE_DIR"
fi

log "Starting histogram build supervisor"
log "Repo root:     $REPO_ROOT"
log "Python:        $PYTHON"
log "Graphlet dir:  $GRAPHLET_DIR"
log "Histogram dir: $HISTOGRAM_OUT_DIR"
log "Bin centers:   $BIN_CENTERS_PATH (num_bins=$NUM_BINS recompute=$RECOMPUTE_BINS)"
log "Workers:       max_workers=$MAX_WORKERS"
log "Retry settings: restart_delay_sec=$RESTART_DELAY_SEC max_restarts=$MAX_RESTARTS stall_limit=$STALL_LIMIT"
log "Failure policy: allow_failed=$ALLOW_FAILED"

attempt=0
restart_count=0
prev_failed_count=""
stall_count=0

while true; do
  attempt=$((attempt + 1))
  log "Launch attempt #$attempt"

  cmd=(
    "$PYTHON" -m cli build-histograms
    --graphlet-dir "$GRAPHLET_DIR"
    --pattern "$PATTERN"
    --histogram-out-dir "$HISTOGRAM_OUT_DIR"
    --bin-centers-path "$BIN_CENTERS_PATH"
    --num-bins "$NUM_BINS"
    --bin-width-factor "$BIN_WIDTH_FACTOR"
    --max-workers "$MAX_WORKERS"
    --progress-every "$PROGRESS_EVERY"
    --manifest-path "$MANIFEST_PATH"
    --state-path "$STATE_PATH"
    --progress-log "$PROGRESS_LOG"
  )
  [[ "$HIST_DENSITY" == "1" ]] && cmd+=(--hist-density)
  [[ "$RESUME" == "1" ]] && cmd+=(--resume)
  # Recompute bins only on the first attempt; restarts must reuse the same bins.
  if [[ "$RECOMPUTE_BINS" == "1" ]] && [[ "$attempt" -eq 1 ]]; then
    cmd+=(--recompute-bins)
  fi

  set +e
  ( cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT/src" "${cmd[@]}" )
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    failed_count="$("$PYTHON" - "$MANIFEST_PATH" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(-1)
    sys.exit(0)
try:
    d = json.loads(p.read_text())
except Exception:
    print(-1)
    sys.exit(0)
print(int(d.get("num_histograms_failed", -1)))
PY
)"

    if [[ "$ALLOW_FAILED" == "1" ]] || [[ "$failed_count" == "0" ]]; then
      log "Build completed successfully on attempt #$attempt (failed=$failed_count)"
      exit 0
    fi

    if [[ "$failed_count" =~ ^[0-9]+$ ]] && [[ "$failed_count" -gt 0 ]]; then
      if [[ -n "$prev_failed_count" ]] && [[ "$failed_count" -ge "$prev_failed_count" ]]; then
        stall_count=$((stall_count + 1))
      else
        stall_count=0
      fi
      prev_failed_count="$failed_count"

      if [[ "$STALL_LIMIT" -ge 0 ]] && [[ "$stall_count" -ge "$STALL_LIMIT" ]]; then
        log "Failed count did not improve for ${stall_count} retries (failed=$failed_count). Exiting."
        exit 86
      fi
    fi

    log "Build finished with failed=$failed_count; will retry resume"
    rc=86
  fi

  log "Build process exited with code $rc"

  if [[ "$MAX_RESTARTS" -ge 0 ]] && [[ "$restart_count" -ge "$MAX_RESTARTS" ]]; then
    log "Reached max restarts ($MAX_RESTARTS). Exiting with code $rc"
    exit "$rc"
  fi

  restart_count=$((restart_count + 1))
  log "Sleeping ${RESTART_DELAY_SEC}s before resume retry"
  sleep "$RESTART_DELAY_SEC"
done
