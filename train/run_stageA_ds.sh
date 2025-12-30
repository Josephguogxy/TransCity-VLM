#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# - This is a template script for open-source use.
# - Replace model/data paths below (or set env vars) before running.

export PYTHONPATH="$(pwd)"

# ====== Environment ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-info}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 20000))}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"

# ====== Paths ======
# You can set these via env vars:
#   DECODER_NAME_OR_PATH / ENCODER_NAME_OR_PATH
#   TRAIN_FILES / VAL_FILES / OUTPUT_DIR / DS_CFG
DECODER="${DECODER_NAME_OR_PATH:-/path/to/decoder-or-hf-id}"
ENCODER="${ENCODER_NAME_OR_PATH:-/path/to/encoder-or-hf-id}"

# Local JSONL files (generate these first if needed)
TRAIN_FILES="${TRAIN_FILES:-dataset_stageA/train_understand_dataset.jsonl}"
VAL_FILES="${VAL_FILES:-dataset_stageA/test_understand_dataset.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/stageA}"
DS_CFG="${DS_CFG:-ds_config_zero3.json}"

if [[ "${DECODER}" == "/path/to/decoder-or-hf-id" ]] || [[ "${ENCODER}" == "/path/to/encoder-or-hf-id" ]]; then
  echo "ERROR: Please set DECODER_NAME_OR_PATH and ENCODER_NAME_OR_PATH (HF model id or local path)." >&2
  exit 1
fi

# ====== Training hyperparameters ======
PER_DEVICE_BS=8          # micro-batch per GPU
GRAD_ACC=8               # gradient accumulation steps
LR=3e-5                  # Stage A: train RoBERTa + projector only
WEIGHT_DECAY=0.1
EPOCHS=5
MAXLEN=1024
LOG_STEPS=5
SAVE_STEPS=500
EVAL_STEPS=500

# ====== Concept tokens (must match model & collator) ======
NUM_CHUNK_TOKENS=4
NUM_IMAGE_TOKENS=32

# ====== Launch (DeepSpeed + ZeRO-3) ======
deepspeed --num_gpus 8 train_mm_stageA.py \
  --deepspeed ${DS_CFG} \
  --output_dir ${OUTPUT_DIR} \
  --do_train true \
  --do_eval true \
  --eval_strategy "steps" \
  --eval_steps ${EVAL_STEPS} \
  --logging_steps ${LOG_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --per_device_train_batch_size ${PER_DEVICE_BS} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --learning_rate ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_train_epochs ${EPOCHS} \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --bf16 true \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters false \
  \
  --decoder_name_or_path ${DECODER} \
  --encoder_name_or_path ${ENCODER} \
  \
  --num_chunk_tokens ${NUM_CHUNK_TOKENS} \
  --num_image_tokens ${NUM_IMAGE_TOKENS} \
  --adapter_heads 4 \
  --encoder_max_length 512 \
  --imagebind_variant "google/siglip-so400m-patch14-384" \
  --group_layers 2 \
  --group_heads 8 \
  --group_tau 1.0 \
  \
  --open_roberta true \
  --open_vpma true \
  --open_image_projector true \
  --open_imagebind false \
  --freeze_decoder true \
  \
  --alpha_understand 1.0 \
  --beta_prediction 0.0 \
  --beta_reason 0.0 \
  --lambda_rat 0.3 \
  \
  --max_length ${MAXLEN} \
  --max_images_per_sample 1 \
  --image_size 384 \
  \
  $(for f in ${TRAIN_FILES}; do echo --train_files ${f}; done) \
  $(for f in ${VAL_FILES}; do echo --validation_files ${f}; done)

# After training, train_mm_stageA.py will:
# 1) Save full HF weights to ${OUTPUT_DIR}
# 2) If save_multimodal_checkpoint() is implemented, also export mm_checkpoint/ (RoBERTa + vPMA + ImageProjector only)
