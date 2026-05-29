#!/bin/sh
set -eu

PROJECT_ROOT="/data/ICSD_graphlet/GraphletDatabase/FinalFeaturization"
INPUT_DIR="/data/ICSD_graphlet/GraphletDatabase/data/MP_cifs"
OUTPUT_DIR="$PROJECT_ROOT/Graphlets"

mkdir -p "$OUTPUT_DIR" /tmp/mpl-final
export MPLCONFIGDIR=/tmp/mpl-final

exec /home/ap2563/miniconda3/bin/conda run -n mlcomp \
  python "$PROJECT_ROOT/main/Build_Folder_Graphlets.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pattern '*.cif' \
  --overwrite \
  --progress-every 100 \
  --checkpoint-every 25 \
  --monitor-interval 30
