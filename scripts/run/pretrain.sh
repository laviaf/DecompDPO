#!/bin/bash
# Pre-train the DecompDiff* model.
# Usage: bash scripts/run/pretrain.sh

set -euo pipefail

python -m scripts.pretrain \
    --config configs/pretrain/pretrain.yml \
    --logdir ./logs_pretrain
