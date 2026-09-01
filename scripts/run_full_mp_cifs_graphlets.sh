#!/usr/bin/env bash
#
# Supervisor for a full graphlet build over a directory of CIF files.
#
# Wraps the `build-folder` CLI with a single-instance lock, bounded restarts,
# and stall detection. The build itself is resumable: existing outputs are
# skipped unless OVERWRITE=1, so a restart picks up where the last attempt
# stopped.
#
# Depends only on this repository: python3, `src/cli.py`, and `config/`.
#
# Usage:
#   scripts/run_full_mp_cifs_graphlets.sh
#   INPUT_DIR=/path/to/cifs OUTPUT_DIR=/path/to/out scripts/run_full_mp_cifs_graphlets.sh
#   MAX_FILES=200 OUTPUT_DIR=/tmp/dryrun scripts/run_full_mp_cifs_graphlets.sh   # dry run
#
# Note on failures: CIFs whose neighbour distances fall below 1 A are rejected
# by design (Create_Graphlets.get_neighbors raises). They are counted in
# num_graphlets_failed and are expected to remain non-zero on real corpora, so
# set ALLOW_FAILED=1 (the default) unless you want the supervisor to keep
# retrying them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"

INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/data/MP_cifs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Graphlets/MP_cifs}"
PATTERN="${PATTERN:-*.cif}"
SUFFIX="${SUFFIX:-_graphlets.json}"

MAX_WORKERS="${MAX_WORKERS:-20}"
CPU_CAP="${CPU_CAP:-20}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
OVERWRITE="${OVERWRITE:-0}"

RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"   # -1 means unlimited retries
STALL_LIMIT="${STALL_LIMIT:-3}"      # stop after N non-improving failure counts; -1 disables
ALLOW_FAILED="${ALLOW_FAILED:-1}"    # 1: accept <1 A rejections and finish; 0: retry until failures=0

# Dry-run support: build only the first MAX_FILES CIFs by staging a directory
# of symlinks. Unset or 0 means use INPUT_DIR directly.
MAX_FILES="${MAX_FILES:-0}"

STATE_PATH="${STATE_PATH:-$OUTPUT_DIR/graphlet_build_state.json}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/graphlet_build_manifest.json}"
PROGRESS_LOG="${PROGRESS_LOG:-$OUTPUT_DIR/graphlet_build_progress.log}"
RUNNER_LOG="${RUNNER_LOG:-$OUTPUT_DIR/full_run_supervisor.log}"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_DIR/full_run.lock}"

mkdir -p "$OUTPUT_DIR"

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

if [[ "$MAX_FILES" -gt 0 ]]; then
  STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/graphlet_subset.XXXXXX")"
  # shellcheck disable=SC2231
  n=0
  for f in "$INPUT_DIR"/$PATTERN; do
    [[ -e "$f" ]] || continue
    ln -s "$f" "$STAGE_DIR/"
    n=$((n + 1))
    [[ "$n" -ge "$MAX_FILES" ]] && break
  done
  log "Dry run: staged $n CIF symlinks in $STAGE_DIR"
  INPUT_DIR="$STAGE_DIR"
fi

log "Starting graphlet build supervisor"
log "Repo root:  $REPO_ROOT"
log "Python:     $PYTHON"
log "Input dir:  $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "CPU settings: max_workers=$MAX_WORKERS cpu_cap=$CPU_CAP max_in_flight=${MAX_IN_FLIGHT:-auto}"
log "Retry settings: restart_delay_sec=$RESTART_DELAY_SEC max_restarts=$MAX_RESTARTS stall_limit=$STALL_LIMIT"
log "Failure policy: allow_failed=$ALLOW_FAILED (CIFs with <1 A contacts are rejected by design)"

attempt=0
restart_count=0
prev_failed_count=""
stall_count=0

while true; do
  attempt=$((attempt + 1))
  log "Launch attempt #$attempt"

  cmd=(
    "$PYTHON" -m cli build-folder
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --pattern "$PATTERN"
    --suffix "$SUFFIX"
    --max-workers "$MAX_WORKERS"
    --cpu-cap "$CPU_CAP"
    --monitor-interval "$MONITOR_INTERVAL"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --progress-every "$PROGRESS_EVERY"
    --manifest-path "$MANIFEST_PATH"
    --state-path "$STATE_PATH"
    --progress-log "$PROGRESS_LOG"
  )
  [[ -n "$MAX_IN_FLIGHT" ]] && cmd+=(--max-in-flight "$MAX_IN_FLIGHT")
  [[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)

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
print(int(d.get("num_graphlets_failed", -1)))
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
