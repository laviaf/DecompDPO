#!/bin/bash
# Step 6: Select the best checkpoint based on success rate.
# Usage: bash scripts/run/select_best_ckpt.sh <ckpt_eval_dir>

CKPT_EVAL_DIR=${1:-"outputs/ckpt_eval"}

python -m scripts.select_best_ckpt ${CKPT_EVAL_DIR}
