# trafficPrediction

This repository contains code and scripts for a **multi-stage training pipeline** (Stage A/B/C) and **GRPO-style RL fine-tuning** for a traffic prediction–related project, plus an optional `traffic_service/` module for serving/agent integration.

> The training scripts are provided as **templates**. You must point them to your own models, checkpoints, and JSONL datasets.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

Training scripts expect **JSONL** datasets (see the default paths in the `run_*.sh` scripts).
You will need to prepare these files yourself.

## Training

All training entrypoints are shell scripts in the repo root. They were made “open-source friendly” by:
- removing internal absolute paths
- using environment variables (or editable defaults) for model / data / checkpoint locations

### Stage A (multimodal warm-up / SFT)

```bash
# Required:
export DECODER_NAME_OR_PATH="your-decoder-model-id-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-id-or-local-path"

# Optional:
export TRAIN_FILES="dataset_stageA/train_understand_dataset.jsonl"
export VAL_FILES="dataset_stageA/test_understand_dataset.jsonl"
export OUTPUT_DIR="outputs/stageA"

bash run_stageA_ds.sh
```

### Stage B

```bash
export DECODER_NAME_OR_PATH="..."
export ENCODER_NAME_OR_PATH="..."
export FULL_MODEL_LOAD_DIR="outputs/stageA"
export OUTPUT_DIR="outputs/stageB"

bash run_stageB_ds.sh
```

### Stage C

```bash
export DECODER_NAME_OR_PATH="..."
export ENCODER_NAME_OR_PATH="..."
export FULL_MODEL_LOAD_DIR="outputs/stageB"
export OUTPUT_DIR="outputs/stageC"

bash run_stageC_ds.sh
```

### GRPO (RL fine-tuning)

```bash
export DECODER_NAME_OR_PATH="tiiuae/Falcon3-7B-Instruct"   # or your own
export ENCODER_NAME_OR_PATH="FacebookAI/roberta-large"     # or your own
export SFT_DIR="outputs_sft"
export TRAIN_JSONL_REASON="dataset/train_dataset_reason.jsonl"
export OUT_DIR="outputs_grpo"

bash run_grpo.sh
```

**Note:** `train_grpo.py` imports `dataprocessing_lora`. If you see `ModuleNotFoundError`, ensure `dataprocessing_lora.py` is included (or update the import to your local equivalent).

## Serving / integration

`traffic_service/` contains optional service/agent code (see `traffic_service/cli.py` for likely entrypoints).

## License

- Project code: **MIT** (see `LICENSE`)
- Third-party components bundled in this repo may use **different licenses** (see `THIRD_PARTY_NOTICES.md`)

