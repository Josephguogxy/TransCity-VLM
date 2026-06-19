<p align="center">
  <img src="assets/TransCityLogo.svg" alt="TransCity-VLM Logo" width="180">
</p>

# An Interpretable Vision–Language Foundation Model for Sustainable Smart Cities with Physical World Data-to-Reasoning Traceability

[![License](https://img.shields.io/badge/license-See%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](#environment)
[![PyTorch](https://img.shields.io/badge/pytorch-required-orange.svg)](#environment)

A research-oriented codebase for **multimodal forecasting** (e.g., traffic flow / electricity load) with a **multi-stage training pipeline**:

- **Stage-A**: multimodal warm-up / alignment
- **Stage-B**: supervised multi-task training
- **Stage-C**: further tuning
- **Stage-D**: RL-style fine-tuning for reasoning quality

> ⚠️ Data and checkpoints are not shipped. You must prepare your own JSONL datasets and model weights.

---

## 🔥 News

- [2025.12.30] Initial open-source release.

---

## ✨ Demos / Figures

---

## Brief Introduction

This repository implements a multimodal LLM-based forecasting workflow. Each sample can include:

- **Text chunks** (e.g., POI / News / Accident / HopSensor / HopBA)
- **Optional images** (paths/URIs)
- A **prompt** formatted for chat-style decoders (example: Qwen3 chat template)

The core pipeline uses:

- Hugging Face Transformers (`Trainer`, `TrainingArguments`)
- DeepSpeed ZeRO for distributed training (Stage-A/B scripts)
- A unified JSONL data schema normalized by `dataprocessing.py`
- Optional RL fine-tuning via GRPO

---

## Getting Started

### Table of Contents

- [Code Structure](#code-structure)
- [Environment](#environment)
- [Data Format](#data-format)
- [Training](#training)
  - [Stage A](#stage-a)
  - [Stage B](#stage-b)
  - [Stage C](#stage-c)
  - [Stage-D](#Stage-d)
- [Inference & Evaluation](#inference--evaluation)
- [Serving / Integration](#serving--integration)
- [Tokenizer & Vocab Policy (No New Tokens)](#tokenizer--vocab-policy-no-new-tokens)
- [License Notices](#license-notices)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Code Structure

A simplified layout (some folders omitted):

```text
TransCity-VLM/
|-- README.md
|-- LICENSE
|-- LICENSE-APACHE-2.0.txt
|-- LICENSE_NOTICES.md
|-- requirements.txt
|-- dataprocessing.py
|-- train_mm_stageA.py
|-- train_mm_stageB.py
|-- train_mm_stageC.py
|-- train_stageD_sft_warmup.py
|-- train_grpo.py
|-- ds_config_zero2.json
|-- ds_config_zero3.json
|-- ds_config_zero3_hfopt.json
|-- configs/
|   |-- deepspeed_zero3.json
|   `-- deepspeed_zero2_stageD.json
|-- model/
|   |-- builder.py
|   |-- decoder_mixlora.py
|   |-- mixlora_masked.py
|   |-- language_model/
|   |-- multimodal_encoder/
|   |-- multimodal_projector/
|   `-- reinforcement_learning/
|-- agentic_system/
|   |-- cli.py
|   |-- config.py
|   |-- schemas.py
|   |-- agents/
|   |-- clients/
|   |-- core/
|   |-- db/
|   |-- llm/
|   `-- utils/
|-- train/
|   |-- run_stageA_ds.sh
|   |-- run_stageB_ds.sh
|   |-- run_stageC_ds.sh
|   |-- run_stageD_sft_warmup_ds.sh
|   `-- run_grpo.sh
|-- eval/
|   |-- inference_abc.py
|   |-- inference_grpo.py
|   `-- inference_agentic.py
`-- utils/
    `-- mixed_pr_sampler.py
```

---
## Environment

### Option 1: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: conda (example)

```bash
conda create -n trafficPrediction python=3.10 -y
conda activate trafficPrediction
pip install -r requirements.txt
```

---

## Training

> Training launchers are kept under `train/`. Run them from the repository root and edit environment variables or server path defaults as needed.

### Stage A

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export TRAIN_FILES="dataset_stageA/train_understand_dataset.jsonl"
export VAL_FILES="dataset_stageA/test_understand_dataset.jsonl"
export OUTPUT_DIR="outputs/stageA"

bash train/run_stageA_ds.sh
```

### Stage B

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export FULL_MODEL_LOAD_DIR="outputs/stageA"
export TRAIN_FILES="dataset_stageB/train_*.jsonl"
export VAL_FILES="dataset_stageB/val_*.jsonl"
export OUTPUT_DIR="outputs/stageB"

bash train/run_stageB_ds.sh
```

### Stage C

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export FULL_MODEL_LOAD_DIR="outputs/stageB"
export TRAIN_FILES="dataset_stageC/train_*.jsonl"
export VAL_FILES="dataset_stageC/val_*.jsonl"
export OUTPUT_DIR="outputs/stageC"

bash train/run_stageC_ds.sh
```

### Stage-D

```bash
export SFT_DIR="outputs/stageC"
export TRAIN_FILES="dataset_rl/train_reason.jsonl"
export OUTPUT_DIR="outputs/grpo"

bash train/run_grpo.sh
```

---

## Inference & Evaluation

This repo keeps three inference entry points:

```bash
python eval/inference_abc.py      # Stage-B/Stage-C style model inference and metrics
python eval/inference_grpo.py     # GRPO / Stage-D checkpoint inference
python eval/inference_agentic.py  # Agent data collection followed by Stage-C or GRPO inference
```

The standard evaluation flow is still:

1) generating text outputs,
2) extracting numeric sequences,
3) computing MAE/MSE/RMSE/MAPE/wMAPE,
4) saving `labels_*.txt`, `preds_*.txt`, `metrics_*.json`, and per-rank `raw_*.jsonl` when supported by the selected script.

---

## License Notices

This repository may contain **multiple licenses**:

- See `LICENSE` for the repository-level license.
- See `LICENSE_NOTICES.md` for bundled/third-party components and their licenses.
- Some components (e.g., certain multimodal encoders) may be **non-commercial** — check carefully before commercial usage.

---

## Citation

If you use this repository in your work, you can cite it as:

```bibtex
@misc{trafficPrediction,
  title        = {trafficPrediction: SmartCityLLM Multimodal Forecasting},
  author       = {torchtorch Authors},
  year         = {2025},
  howpublished = {https://github.com/<your-org-or-user>/trafficPrediction},
}
```

---

## Acknowledgements

This codebase builds on open-source libraries such as:

- PyTorch
- Hugging Face Transformers / Datasets
- DeepSpeed (for distributed training)
- Accelerate / TRL for RL-related utilities

We thank the open-source community for these tools.