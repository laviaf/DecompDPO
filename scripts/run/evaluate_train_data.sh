#!/bin/bash
# Step 2: Evaluate sampled training data.
# Usage: bash scripts/run/evaluate_train_data.sh [start_id] [end_id_exclusive]

set -euo pipefail

TOTAL_NUM=4165  # total number of training proteins
START_ID=${1:-0}
END_ID_EXCLUSIVE=${2:-$TOTAL_NUM}

OUTDIR="outputs/training_data"

if (( START_ID < 0 )); then START_ID=0; fi
if (( END_ID_EXCLUSIVE > TOTAL_NUM )); then END_ID_EXCLUSIVE=$TOTAL_NUM; fi

for ((data_id=START_ID; data_id<END_ID_EXCLUSIVE; data_id++)); do
  padded_data_id=$(printf "%03d" "${data_id}")
  python -m scripts.evaluate \
    "${OUTDIR}/sample_train_data_${padded_data_id}" \
    --data_id "${data_id}"
done
