# DecompDPO: Decomposed Direct Preference Optimization for Structure-Based Drug Design

[![Paper](https://img.shields.io/badge/Paper-TMLR_2025-blue)](https://openreview.net/forum?id=dwSpo5DRk8)

Official implementation of **DecompDPO**, a structure-based optimization method that aligns diffusion models with pharmaceutical needs using multi-granularity preference pairs.

## Installation

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate decompdiff

# Install additional dependencies
pip install meeko==0.1.dev3 vina==1.2.2 pdb2pqr==3.6.1
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git
```

## Data Preparation

We release the processed data and checkpoints on Hugging Face: [Annie37/DecompDPO](https://huggingface.co/datasets/Annie37/DecompDPO/tree/main).

1. Download the released folders (requires `git-lfs`):

```bash
git lfs install
git clone https://huggingface.co/datasets/Annie37/DecompDPO hf_decompdpo
```

2. Copy (or symlink) them into this repo root:

```bash
ln -s "$(pwd)/hf_decompdpo/data" data
ln -s "$(pwd)/hf_decompdpo/checkpoints" checkpoints
```

If you want to **pre-train DecompDiff\*** from scratch, our data preparation follows the [DecompDiff](https://github.com/bytedance/DecompDiff) pipeline for CrossDocked2020 (i.e., you can reproduce `data/` by following DecompDiff’s preprocessing steps and matching the layout above).

## Pipeline

We provide the processed dataset and checkpoints on Hugging Face, so **Part 1** and **Part 2** below are optional:

- **Processed dataset**: [Hugging Face `data/`](https://huggingface.co/datasets/Annie37/DecompDPO/tree/main/data)
- **Released checkpoints**: [Hugging Face `checkpoints/`](https://huggingface.co/datasets/Annie37/DecompDPO/tree/main/checkpoints)
  - **Pre-trained DecompDiff\*** base: [`decompdiff_bv_sche.pt`](https://huggingface.co/datasets/Annie37/DecompDPO/blob/main/checkpoints/decompdiff_bv_sche.pt)
  - **Trained DecompDPO (best)**: [`best_ckpt.pt`](https://huggingface.co/datasets/Annie37/DecompDPO/blob/main/checkpoints/best_ckpt.pt)

### Part 1 (Optional): Pretraining & Data Preparation

- **Use released artifacts (recommended)**: follow `## Data Preparation` above to place/symlink Hugging Face `data/` and `checkpoints/` into this repo.
- **Reproduce preprocessing / pre-train from scratch**: the data preparation pipeline follows [DecompDiff](https://github.com/bytedance/DecompDiff) for CrossDocked2020. To pre-train the DecompDiff\* base model:

```bash
bash scripts/run/pretrain.sh
```

### Part 2 (Optional): DecompDPO training

If you want to reproduce DecompDPO training (instead of using the released `best_ckpt.pt`), run:

#### Step 1: Sample training data

Sample 10 molecules for each training protein using the pre-trained model:

```bash
bash scripts/run/sample_train_data.sh
```

#### Step 2: Evaluate training data

Evaluate the chemical properties and binding affinity of sampled molecules:

```bash
bash scripts/run/evaluate_train_data.sh
```

#### Step 3: Process training data

Construct preference pairs from the evaluation results:

```bash
bash scripts/run/process_train_data.sh
```

#### Step 4: Train DecompDPO

Fine-tune the model with DecompDPO:

```bash
bash scripts/run/train_decompdpo.sh
```

Key training arguments:
- `--schedule_type linear`: Use linear beta schedule (recommended)

### Part 3: Sampling and Evaluation

#### (Optional) Evaluate checkpoints and select the best

Sample and evaluate 20 molecules per target protein for each checkpoint:

```bash
bash scripts/run/sample_ckpt.sh
```

Select the best checkpoint based on normalized success rate:

```bash
bash scripts/run/select_best_ckpt.sh outputs/ckpt_eval
```

#### Final evaluation

Sample 100 molecules per target protein from the best checkpoint and evaluate:

```bash
# path to best checkpoint
export CKPT_PATH="checkpoints/best_ckpt.pt"

# Sample
bash scripts/run/sample.sh

# Evaluate
bash scripts/run/evaluate.sh
```

## Citation

```bibtex
@article{cheng2025decompdpo,
  title={Decomposed Direct Preference Optimization for Structure-Based Drug Design},
  author={Cheng, Xiwei and Zhou, Xiangxin and Bao, Yu and Yang, Yuwei and Gu, Quanquan},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## Acknowledgements

This codebase builds upon [DecompDiff](https://github.com/bytedance/DecompDiff). We thank the authors for their excellent work.
