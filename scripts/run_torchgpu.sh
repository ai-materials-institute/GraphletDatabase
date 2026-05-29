#!/usr/bin/env sh
set -eu

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CONDA_ENV_NAME:-torchgpu}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "Conda binary not found or not executable: $CONDA_BIN" >&2
  echo "Set CONDA_BIN to your conda path and retry." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export MPLCONFIGDIR
export XDG_CACHE_HOME

exec "$CONDA_BIN" run -n "$ENV_NAME" "$@"
