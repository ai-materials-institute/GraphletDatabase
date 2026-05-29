#!/bin/sh
set -u

export MPLCONFIGDIR=/tmp/mpl
export XDG_CACHE_HOME=/tmp/xdg-cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE="/data/ICSD_graphlet/GPTc_simplified"
GP_DIR="$BASE/src/GP_Models"
CSV_PATH="$BASE/Misc/merged_superconductor_data.csv"
REG_FEATURE_DIR="$BASE/ICSD_Features/regression_feature_jsons"
CLAS_FEATURE_DIR="$BASE/ICSD_Features/classification_feature_jsons"

MASTER_LOG="$BASE/train_all67_symm_master.log"
REG_LOG="$BASE/train_regression_all67_symm.log"
CLAS_LOG="$BASE/train_classification_all67_symm.log"

REG_SAVE_DIR="$GP_DIR/Trained Models/Regressor_All67_Symm_JSONCSV_20260317"
CLAS_SAVE_DIR="$GP_DIR/Trained Models/Classifier_All67_Symm_JSONCSV_20260317"

HIST_IDX="$(seq -s ' ' 0 66)"
REG_EMD_BATCHES="${REG_EMD_BATCHES:-10}"
CLAS_EMD_BATCHES="${CLAS_EMD_BATCHES:-10}"
CLAS_BATCH_SIZE="${CLAS_BATCH_SIZE:-1024}"
CLAS_N_INDUCING="${CLAS_N_INDUCING:-1024}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

cd "$GP_DIR"

: > "$REG_LOG"
: > "$CLAS_LOG"

echo "[$(timestamp)] Starting regression training" >> "$MASTER_LOG"
set +e
/home/ap2563/miniconda3/bin/conda run --no-capture-output -n torchgpu python -u train.py regression \
  --path_to_data_file "$REG_FEATURE_DIR" \
  --labels_csv_file "$CSV_PATH" \
  --csv_split_file "$CSV_PATH" \
  --hist-idx $HIST_IDX \
  --emd_batches "$REG_EMD_BATCHES" \
  --use-symm \
  --save_dir "$REG_SAVE_DIR" \
  > "$REG_LOG" 2>&1
REG_STATUS=$?
set -e
echo "[$(timestamp)] Regression finished with status $REG_STATUS" >> "$MASTER_LOG"

if [ "$REG_STATUS" -ne 0 ]; then
  echo "[$(timestamp)] Skipping classification because regression failed" >> "$MASTER_LOG"
  exit "$REG_STATUS"
fi

echo "[$(timestamp)] Starting classification training" >> "$MASTER_LOG"
set +e
/home/ap2563/miniconda3/bin/conda run --no-capture-output -n torchgpu python -u train.py classification \
  --path_to_data_file "$CLAS_FEATURE_DIR" \
  --labels_csv_file "$CSV_PATH" \
  --csv_split_file "$CSV_PATH" \
  --hist-idx $HIST_IDX \
  --emd_batches "$CLAS_EMD_BATCHES" \
  --batch_size "$CLAS_BATCH_SIZE" \
  --n_inducing "$CLAS_N_INDUCING" \
  --use-symm \
  --save_dir "$CLAS_SAVE_DIR" \
  > "$CLAS_LOG" 2>&1
CLAS_STATUS=$?
set -e
echo "[$(timestamp)] Classification finished with status $CLAS_STATUS" >> "$MASTER_LOG"
exit "$CLAS_STATUS"
