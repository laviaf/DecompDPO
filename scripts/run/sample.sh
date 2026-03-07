#!/bin/bash
# Step 7: Sample 100 molecules from the best checkpoint for final evaluation.
# Usage: bash scripts/run/sample.sh [start_id] [end_id_exclusive]
# NOTE: Set CKPT_PATH to the best checkpoint selected in Step 6.

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
  if [ -f "${output_dir}/sample-ref_prior-${padded_data_id}/result.pt" ] || \
    [ -f "${output_dir}/sample-beta_prior-${padded_data_id}/result.pt" ]; then
    echo "finished: ${padded_data_id}"
    continue
  fi

  echo "data_id=${padded_data_id}"
  echo "ckpt_path=${CKPT_PATH}"
  batch_size=100
  while ((batch_size >= 1)); do
    python -m scripts.sample \
      configs/sample/sample.yml \
      --ckpt_path "${CKPT_PATH}" \
      --outdir "${output_dir}" \
      --batch_size "${batch_size}" \
      -i "${data_id}"

    if [ $? -eq 0 ]; then
      echo "Program executed successfully with batch_size: ${batch_size}"
      break
    else
      echo "Program failed with batch_size: ${batch_size}. Trying with a smaller batch size."
      batch_size=$((batch_size / 2))
    fi
  done
done
