#!/bin/bash
# Step 3: Process training data into preference pairs.
# Usage: bash scripts/run/process_train_data.sh

python -m scripts.process_data \
    --data_dir outputs/training_data \
    --save_dir data/training_pairs_wo_recon.pt \
    --has_recon_failed_data True
