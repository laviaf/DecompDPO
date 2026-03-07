#!/bin/bash
# Step 4: Train the DecompDPO model.
# Usage: bash scripts/run/train_decompdpo.sh

set -euo pipefail

python -m scripts.train_decompdpo \
    --config configs/decompdpo/train.yml \
    --logdir ./logs_decompdpo \
    --schedule_type linear
