#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# - This is a template script for open-source use.
# - Replace model/data paths below (or set env vars) before running.

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$(pwd)"

# ====== SFT checkpoint dir ======
SFT_DIR="${SFT_DIR:-outputs_sft}"

# ====== RL training data (recommended: start with reason only) ======
TRAIN_JSONL_REASON="${TRAIN_JSONL_REASON:-dataset/train_dataset_reason.jsonl}"

# ====== output ======
OUT_DIR="${OUT_DIR:-outputs_grpo}"

# ====== base models ======
DECODER="${DECODER_NAME_OR_PATH:-tiiuae/Falcon3-7B-Instruct}"
ENCODER="${ENCODER_NAME_OR_PATH:-FacebookAI/roberta-large}"

python train_grpo.py \
  --sft_dir "${SFT_DIR}" \
  --train_files "${TRAIN_JSONL_REASON}" \
  --output_dir "${OUT_DIR}" \
  --only_task "reason" \
  --decoder_name_or_path "${DECODER}" \
  --encoder_name_or_path "${ENCODER}" \
  \
  --load_in_bits 4 \
  --torch_dtype auto \
  \
  --max_length 4096 \
  --batch_size 2 \
  --grad_accum 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --num_epochs 1 \
  --max_grad_norm 1.0 \
  --log_steps 5 \
  --save_steps 50 \
  --eval_steps 50 \
  \
  --num_generations 4 \
  --max_new_tokens 768 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 0 \
  --clip_eps 0.2 \
  --beta_kl 0.04 \
  \
  --micro_batch_logps 4
