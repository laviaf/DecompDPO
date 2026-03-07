#!/bin/bash
# Step 8: Evaluate 100 sampled molecules for each test protein.
# Usage: bash scripts/run/evaluate.sh [start_id] [end_id_exclusive]
# NOTE: Set CKPT_PATH to match the checkpoint used in Step 7.

set -euo pipefail

TOTAL_NUM=100
START_ID=${1:-0}
END_ID_EXCLUSIVE=${2:-$TOTAL_NUM}

CKPT_PATH=${CKPT_PATH:-"checkpoints/best_ckpt.pt"}
echo "ckpt_path=${CKPT_PATH}"

SAMPLE_OUT_DIR="outputs/final_eval_best_ckpt"
mkdir -p "${SAMPLE_OUT_DIR}"
output_dir=${SAMPLE_OUT_DIR}

if (( START_ID < 0 )); then START_ID=0; fi
if (( END_ID_EXCLUSIVE > TOTAL_NUM )); then END_ID_EXCLUSIVE=$TOTAL_NUM; fi

for ((data_id=START_ID; data_id<END_ID_EXCLUSIVE; data_id++)); do
  padded_data_id=$(printf "%03d" "${data_id}")
  if [ -f "${output_dir}/sample-ref_prior-${padded_data_id}/eval/result.json" ] || \
    [ -f "${output_dir}/sample-beta_prior-${padded_data_id}/eval/result.json" ]; then
    echo "finished: ${padded_data_id}"
    continue
  fi

  sample_dir=$(find "${output_dir}" -name "sample-*-${padded_data_id}")
  echo "sample_dir=${sample_dir}"
  python -m scripts.evaluate_ckpt \
    --sample_res_path "${sample_dir}" \
    --item "${data_id}" \
    --save_path "${sample_dir}/eval/result.json" \
    --protein_root data/test_set
done
