#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# - This is a template script for open-source use.
# - Replace model/data paths below (or set env vars) before running.

export PYTHONPATH="$(pwd)"

# ====== Environment ======
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# ====== Paths ======
# You can set these via env vars:
#   DECODER_NAME_OR_PATH / ENCODER_NAME_OR_PATH
#   FULL_MODEL_LOAD_DIR / OUTPUT_DIR / DS_CFG
DECODER="${DECODER_NAME_OR_PATH:-/path/to/decoder-or-hf-id}"
ENCODER="${ENCODER_NAME_OR_PATH:-/path/to/encoder-or-hf-id}"
FULL_MODEL_LOAD_DIR="${FULL_MODEL_LOAD_DIR:-outputs/stageA}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/stageB}"
DS_CFG="${DS_CFG:-ds_config_zero3_hfopt.json}"

# Fallback if the referenced DS config is not in the repo
if [[ ! -f "${DS_CFG}" ]]; then
  echo "Warning: DeepSpeed config '${DS_CFG}' not found. Falling back to 'ds_config_zero3.json'." >&2
  DS_CFG="ds_config_zero3.json"
fi

if [[ "${DECODER}" == "/path/to/decoder-or-hf-id" ]] || [[ "${ENCODER}" == "/path/to/encoder-or-hf-id" ]]; then
  echo "ERROR: Please set DECODER_NAME_OR_PATH and ENCODER_NAME_OR_PATH (HF model id or local path)." >&2
  exit 1
fi

if [[ ! -d "${FULL_MODEL_LOAD_DIR}" ]]; then
  echo "Warning: FULL_MODEL_LOAD_DIR '${FULL_MODEL_LOAD_DIR}' does not exist yet. Make sure Stage A outputs are there." >&2
fi

deepspeed --num_gpus 8 train_mm_stageB.py \
  --deepspeed ${DS_CFG} \
  --output_dir ${OUTPUT_DIR} \
  --do_train true \
  --do_eval true \
  --eval_strategy "steps" \
  --eval_steps ${EVAL_STEPS} \
  --logging_steps ${LOG_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 2 \
  --per_device_train_batch_size ${PER_DEVICE_BS} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --learning_rate ${DEC_LR} \
  --mm_learning_rate ${MM_LR} \
  --warmup_ratio ${WARMUP} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_train_epochs ${EPOCHS} \
  --per_device_eval_batch_size 1 \
  --prediction_loss_only true \
  --eval_accumulation_steps 1 \
  --bf16 true \
  --bf16_full_eval true \
  --max_eval_samples ${EVAL_SAMPLES} \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters false \
  --remove_unused_columns false \
  \
  --decoder_name_or_path ${DECODER} \
  --encoder_name_or_path ${ENCODER} \
  --full_model_load_dir ${FULL_MODEL_LOAD_DIR} \
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
  --freeze_decoder false \
  \
  --alpha_understand 0.3 \
  --beta_prediction 0.7 \
  --beta_reason 0.0 \
  --lambda_rat 0.0 \
  \
  --max_length ${MAXLEN} \
  --max_images_per_sample 5 \
  --image_size 384 \
  \
  --train_files \
    dataset_stageB/train_prediction_dataset.jsonl \
    dataset_stageB/train_understand_dataset.jsonl \
    dataset_stageB/train_prediction_others_dataset.jsonl \
    dataset_stageB/MapQA_train.jsonl \
    dataset_stageB/maplm_train.jsonl \
    dataset_stageB/omni_train.jsonl \
    dataset_stageB/EarthVQA_train.jsonl \
  --validation_files \
    dataset_stageB/test_prediction_dataset.jsonl \
    dataset_stageB/test_understand_dataset.jsonl \
    dataset_stageB/EarthVQA_test.jsonl  \
    dataset_stageB/maplm_test.jsonl
