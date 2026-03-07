#!/bin/bash
# Step 5: Sample and evaluate 20 molecules for each checkpoint.
# Usage: bash scripts/run/sample_ckpt.sh [eval_num] [outdir] [ckpt_dir] [start_id] [end_id_exclusive]

set -euo pipefail

EVAL_NUM=${1:-100}
OUTDIR=${2:-"outputs/ckpt_eval"}
CKPT_DIR=${3:-"logs_decompdpo"}
START_ID=${4:-0}
END_ID_EXCLUSIVE=${5:-$EVAL_NUM}

# Find the checkpoint directory with all checkpoints
ckpt_list_path=$(ls -d ${CKPT_DIR}/*/checkpoints | head -n1)

if (( START_ID < 0 )); then START_ID=0; fi
if (( END_ID_EXCLUSIVE > EVAL_NUM )); then END_ID_EXCLUSIVE=$EVAL_NUM; fi

for ((data_id=START_ID; data_id<END_ID_EXCLUSIVE; data_id++)); do
  padded_data_id=$(printf "%03d" "${data_id}")
  for ckpt in $(ls "${ckpt_list_path}"); do
    ckpt_path="${ckpt_list_path}/${ckpt}"
    ckpt_id=${ckpt:0:-3}
    output_dir="${OUTDIR}/ckpt_${ckpt_id}"

    if [ -f "${output_dir}/sample_ckpt-ref_prior-${padded_data_id}/eval/result.json" ] || \
      [ -f "${output_dir}/sample_ckpt-beta_prior-${padded_data_id}/eval/result.json" ]; then
      echo "finished: ${padded_data_id}-${ckpt_id}"
      continue
    fi

    echo "data_id=${padded_data_id}, ckpt=${ckpt_path}"

    python -m scripts.sample \
      configs/sample/sample_ckpt.yml \
      --ckpt_path "${ckpt_path}" \
      --outdir "${output_dir}" \
      -i "${data_id}" || { exit 817; }

    sample_dir=$(find "${output_dir}" -name "sample_ckpt-*-${padded_data_id}")
    echo "sample_dir=${sample_dir}"
    python -m scripts.evaluate_ckpt \
      --sample_res_path "${sample_dir}" \
      --item "${data_id}" \
      --save_path "${sample_dir}/eval/result.json" \
      --protein_root data/test_set
  done
done
