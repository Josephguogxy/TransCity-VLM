# trafficPrediction (SmartCityLLM) — Multimodal Forecasting with Stage A/B/C + GRPO

[![License](https://img.shields.io/badge/license-See%20LICENSE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](#environment)
[![PyTorch](https://img.shields.io/badge/pytorch-required-orange.svg)](#environment)

A research-oriented codebase for **multimodal forecasting** (e.g., traffic flow / electricity load) with a **multi-stage training pipeline**:

- **Stage-A**: multimodal warm-up / alignment (train encoders & adapters; optionally freeze decoder)
- **Stage-B**: supervised multi-task training (prediction + reasoning)
- **Stage-C**: further tuning (e.g., MixLoRA FFN-only; task-balanced sampling)
- **GRPO** (optional): RL-style fine-tuning for reasoning quality (Qwen3-aligned rules)

> ⚠️ Data and checkpoints are not shipped. You must prepare your own JSONL datasets and model weights.

---

## 🔥 News

- [2025.12.30] Initial open-source release.

---

## ✨ Demos / Figures

All figures are stored in `assets/`.

**Fig. 1 — Data-to-Reasoning Traceability**
<p align="center">
  <img src="assets/Data-To-Reasoning.svg" width="820" alt="Fig.1 Data-to-Reasoning Traceability"/>
</p>

**Fig. 2 — Expert-Level Interpretability**
<p align="center">
  <img src="assets/Expert-Interpretability.svg" width="820" alt="Fig.2 Expert-Level Interpretability"/>
</p>

**Fig. 3 — MoMExp Nets Interpretability**
<p align="center">
  <img src="assets/mapExplain.svg" width="820" alt="Fig.3 MoMExp Nets Interpretability"/>
</p>

**Fig. 4 — Agent-driven Evidence Workflow (Plan DAG + Traceable Records)**
<p align="center">
  <img src="assets/agent-framework-A1.svg" width="920" alt="Fig.4 Agent-driven Evidence Workflow"/>
</p>

**Fig. 5 — Safety Control System**
<p align="center">
  <img src="assets/SafetyControlSystem.svg" width="820" alt="Fig.5 Safety Control System"/>
</p>

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
  - [GRPO RL (optional)](#grpo-rl-optional)
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
TransCity-VLM
/
├─ LICENSE_NOTICES.md
├─ dataprocessing.py
├─ ds_config_zero3.json
├─ LICENSE
├─ LICENSE-APACHE-2.0.txt
├─ requirements.txt
├─ README.md
├─ ds_config_zero3_hfopt.json
├─ ds_config_zero2.json
├─ model/
│  ├─ builder.py
│  ├─ decoder_mixlora.py
│  ├─ mixlora_masked.py
│  ├─ language_model/
│  │  └─ smartcity_llm.py
│  ├─ multimodal_encoder/
│  │  ├─ chunk_text_encoder.py
│  │  ├─ imagebind_encoder.py
│  │  └─ imagebind/
│  │     ├─ data.py
│  │     ├─ requirements.txt
│  │     ├─ LICENSE-CC-BY-NC-4.0.txt
│  │     ├─ LICENSE
│  │     ├─ bpe/
│  │     │  └─ bpe_simple_vocab_16e6.txt.gz
│  │     └─ models/
│  │        ├─ helpers.py
│  │        ├─ imagebind_model.py
│  │        ├─ multimodal_preprocessors.py
│  │        └─ transformer.py
│  ├─ multimodal_projector/
│  │  ├─ group.py
│  │  ├─ projector.py
│  │  └─ vpma.py
│  └─ reinforcement_learning/
│     ├─ config.py
│     ├─ logprobs.py
│     ├─ loss.py
│     ├─ rewards.py
│     └─ rollout.py
├─ traffic_service/
│  ├─ cli.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ agents/
│  │  ├─ base.py
│  │  ├─ bootstrap/
│  │  │  ├─ geocode_forward.py
│  │  │  ├─ nl_parse.py
│  │  │  ├─ normalize.py
│  │  │  └─ time_window.py
│  │  ├─ demographics/
│  │  │  └─ demographics.py
│  │  ├─ geo/
│  │  │  └─ geocode_reverse.py
│  │  ├─ osm/
│  │  │  ├─ poi.py
│  │  │  └─ roads.py
│  │  ├─ record/
│  │  │  └─ record_builder.py
│  │  ├─ satellite/
│  │  │  ├─ fetch_gibs.py
│  │  │  └─ store_mysql.py
│  │  ├─ traffic/
│  │  │  ├─ nearest_sensor.py
│  │  │  └─ traffic_flow.py
│  │  ├─ weather/
│  │  │  └─ weather.py
│  │  └─ web/
│  │     ├─ content_fetcher.py
│  │     ├─ events.py
│  │     ├─ query_generator.py
│  │     ├─ scoring.py
│  │     └─ web_search_agent.py
│  ├─ clients/
│  │  ├─ cds.py
│  │  ├─ census.py
│  │  ├─ gibs.py
│  │  ├─ http.py
│  │  ├─ nominatim.py
│  │  ├─ open_meteo.py
│  │  ├─ overpass.py
│  │  ├─ web_search.py
│  │  └─ worldpop.py
│  ├─ core/
│  │  ├─ context.py
│  │  ├─ executor.py
│  │  ├─ logging.py
│  │  ├─ plan_builder.py
│  │  └─ runner.py
│  ├─ db/
│  │  ├─ flow_repo.py
│  │  ├─ mysql_pool.py
│  │  └─ satellite_images_repo.py
│  ├─ llm/
│  │  ├─ clients.py
│  │  └─ prompts/
│  │     └─ __init__.py
│  └─ utils/
│     ├─ geo.py
│     └─ time.py
├─ trl/
├─ utils/
│  └─ mixed_pr_sampler.py
├─ train/
│  ├─ train_mm_stageA.py
│  ├─ train_mm_stageB.py
│  ├─ train_mm_stageC.py
│  ├─ train_mm_stageD.py
│  ├─ run_grpo.sh
│  ├─ run_stageA_ds.sh
│  ├─ run_stageB_ds.sh
│  └─ run_stageC_ds.sh
└─ eval/
   ├─ inference_mm_stageB.py
   ├─ run_inference_stageB.sh
   ├─ inference_mm_stageA.py
   ├─ inference_mm_stageC.py
   ├─ run_inference_stageA.sh
   └─ run_inference_stageC.sh
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

> Tip: For Stage-A/B DeepSpeed training, you need a working CUDA + NCCL environment.

---

## Training

> All `run_*.sh` scripts are templates: they avoid internal absolute paths and rely on env vars / editable defaults.

### Stage A

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export TRAIN_FILES="dataset_stageA/train_understand_dataset.jsonl"
export VAL_FILES="dataset_stageA/test_understand_dataset.jsonl"
export OUTPUT_DIR="outputs/stageA"

bash run_stageA_ds.sh
```

### Stage B

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export FULL_MODEL_LOAD_DIR="outputs/stageA"
export TRAIN_FILES="dataset_stageB/train_*.jsonl"
export VAL_FILES="dataset_stageB/val_*.jsonl"
export OUTPUT_DIR="outputs/stageB"

bash run_stageB_ds.sh
```

### Stage C

```bash
export DECODER_NAME_OR_PATH="your-decoder-model-or-local-path"
export ENCODER_NAME_OR_PATH="your-encoder-model-or-local-path"
export FULL_MODEL_LOAD_DIR="outputs/stageB"
export TRAIN_FILES="dataset_stageC/train_*.jsonl"
export VAL_FILES="dataset_stageC/val_*.jsonl"
export OUTPUT_DIR="outputs/stageC"

bash run_stageC_ds.sh
```

### GRPO RL

```bash
export SFT_DIR="outputs/stageC"
export TRAIN_FILES="dataset_rl/train_reason.jsonl"
export OUTPUT_DIR="outputs/grpo"

bash run_grpo.sh
```

---

## Inference & Evaluation

This repo typically evaluates by:

1) generating text outputs,
2) extracting numeric sequences,
3) computing MAE/MSE/RMSE/MAPE/wMAPE,
4) saving `labels_*.txt`, `preds_*.txt`, `metrics_*.json`, and per-rank `raw_*.jsonl`.

---

## License Notices

This repository may contain **multiple licenses**:

- See `LICENSE` for the repository-level license.
- See `THIRD_PARTY_NOTICES.md` for bundled/third-party components and their licenses.
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
- (Optional) Accelerate / TRL for RL-related utilities

We thank the open-source community for these tools.