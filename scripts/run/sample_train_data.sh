#!/bin/bash
# Step 1: Sample training data for DecompDPO using the pre-trained model.
# This script samples 10 molecules for each training protein.
# Usage: bash scripts/run/sample_train_data.sh [start_id] [end_id_exclusive]

set -euo pipefail

TOTAL_NUM=4165  # total number of training proteins
START_ID=${1:-0}
END_ID_EXCLUSIVE=${2:-$TOTAL_NUM}

OUTDIR="outputs/training_data"
mkdir -p "${OUTDIR}"
CKPT_PATH='./checkpoints/decompdiff_bv_sche.pt'

if (( START_ID < 0 )); then START_ID=0; fi
if (( END_ID_EXCLUSIVE > TOTAL_NUM )); then END_ID_EXCLUSIVE=$TOTAL_NUM; fi

for ((data_id=START_ID; data_id<END_ID_EXCLUSIVE; data_id++)); do
  batch_size=10
  while ((batch_size >= 1)); do
    python -m scripts.sample_for_train \
      configs/sample/sample_train_data.yml \
      --outdir "${OUTDIR}" \
      -i "${data_id}" \
      --ckpt_path "${CKPT_PATH}" \
      --metric linear_scalar \
      --batch_size "${batch_size}" \
      --prior_mode ref_prior

    if [ $? -eq 0 ]; then
      echo "Program executed successfully with batch_size: ${batch_size}"
      break
    else
      echo "Program failed with batch_size: ${batch_size}. Trying with a smaller batch size."
      batch_size=$((batch_size / 2))
    fi
  done
done
