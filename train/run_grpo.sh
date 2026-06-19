#!/usr/bin/env bash
set -euo pipefail

########################
# 环境
########################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
unset NCCL_BLOCKING_WAIT
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 20000))}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$(pwd)"

# Stage D rollout/runtime defaults
export SMARTCITY_LOCAL_METRICS="${SMARTCITY_LOCAL_METRICS:-1}"
export SMARTCITY_DS3_GATHER_FOR_GENERATION="${SMARTCITY_DS3_GATHER_FOR_GENERATION:-1}"
export SMARTCITY_REASON_ONLY_GRAD_STATS="${SMARTCITY_REASON_ONLY_GRAD_STATS:-0}"
export SMARTCITY_REASON_ONLY_DELTA_STATS="${SMARTCITY_REASON_ONLY_DELTA_STATS:-0}"
export SMARTCITY_REASON_ONLY_RESTORE="${SMARTCITY_REASON_ONLY_RESTORE:-0}"
export SMARTCITY_REWARD_USE_SEMANTIC="${SMARTCITY_REWARD_USE_SEMANTIC:-0}"
export SMARTCITY_GRPO_IGNORE_LABEL_BUDGET="${SMARTCITY_GRPO_IGNORE_LABEL_BUDGET:-1}"

########################
# 路径
########################
DECODER="${DECODER:-/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B}"
ENCODER="${ENCODER:-/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large}"

SFTC_DIR="${SFTC_DIR:-checkpoint/stageC}"

TRAIN_FILES="${TRAIN_FILES:-dataset_grpo/train_dataset_reas.jsonl}"
EVAL_FILES="${EVAL_FILES:-dataset_grpo/test_dataset_reason.jsonl}"

OUT_DIR="${OUTPUT_DIR:-${OUT_DIR:-outputs_grpo}}"
DS_CFG="${DS_CFG:-configs/deepspeed_zero3.json}"

########################
# 结构（对齐 StageC）
########################
NUM_CHUNK_TOKENS="${NUM_CHUNK_TOKENS:-4}"
NUM_IMAGE_TOKENS="${NUM_IMAGE_TOKENS:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
MAXLEN="${MAXLEN:-4096}"

########################
# GRPO
########################
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEMP="${TEMP:-0.65}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
CLIP_EPS="${CLIP_EPS:-0.1}"
BETA_KL="${BETA_KL:-0.01}"

########################
# 训练超参
########################
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-4e-6}"
EPOCHS="${EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:-0}"
MICRO_BATCH_LOGPS="${MICRO_BATCH_LOGPS:-2}"

LOG_STEPS="${LOG_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-60}"
SAVE_FINAL="${SAVE_FINAL:-true}"
EVAL_STEPS="${EVAL_STEPS:-60}"
SAVE_FULL_MODEL="${SAVE_FULL_MODEL:-true}"
SAVE_FULL_EVERY_CHECKPOINT="${SAVE_FULL_EVERY_CHECKPOINT:-false}"
FULL_SAVE_MAX_SHARD_SIZE="${FULL_SAVE_MAX_SHARD_SIZE:-2GB}"
SAVE_TRAINABLE="${SAVE_TRAINABLE:-true}"
NUM_EVAL_BATCHES="${NUM_EVAL_BATCHES:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
NUM_GPUS="${NUM_GPUS:-8}"


########################
# Launch (DeepSpeed only)
########################
deepspeed --num_gpus "${NUM_GPUS}" train_grpo.py \
  --deepspeed "${DS_CFG}" \
  --sft_dir "${SFTC_DIR}" \
  --train_files ${TRAIN_FILES} \
  --eval_files ${EVAL_FILES} \
  --output_dir "${OUT_DIR}" \
  --only_task "reason" \
  \
  --decoder_name_or_path "${DECODER}" \
  --encoder_name_or_path "${ENCODER}" \
  --local_files_only true \
  --padding_token "<|pad|>" \
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
  --open_roberta false \
  --open_vpma false \
  --open_image_projector false \
  --open_imagebind false \
  --freeze_decoder true \
  \
  --use_mixlora true \
  --mixlora_num_universal 4 \
  --mixlora_num_pred 8 \
  --mixlora_num_reason 8 \
  --mixlora_rank_universal_mul 2.0 \
  --mixlora_r 8 \
  --mixlora_alpha 16 \
  --mixlora_share_router_for_wi true \
  --mixlora_enable_attention false \
  --mixlora_ensure_nonzero_gating false \
  \
  --alpha_understand 0.2 \
  --beta_prediction 1.0 \
  --beta_reason 1.0 \
  --lambda_rat 0.3 \
  \
  --max_length ${MAXLEN} \
  --max_images_per_sample 1 \
  --image_size ${IMAGE_SIZE} \
  \
  --num_generations ${NUM_GENERATIONS} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --temperature ${TEMP} \
  --top_p ${TOP_P} \
  --top_k ${TOP_K} \
  --clip_eps ${CLIP_EPS} \
  --beta_kl ${BETA_KL} \
  \
  --batch_size ${BATCH_SIZE} \
  --grad_accum ${GRAD_ACC} \
  --learning_rate ${LR} \
  --weight_decay 0.0 \
  --num_epochs ${EPOCHS} \
  --max_steps ${MAX_STEPS} \
  --max_grad_norm 1.0 \
  --micro_batch_logps ${MICRO_BATCH_LOGPS} \
  --freeze_mixlora_non_reason true \
  --default_reasoning_mode auto \
  --inject_no_think_token false \
  \
  --log_steps ${LOG_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --save_final ${SAVE_FINAL} \
  --eval_steps ${EVAL_STEPS} \
  --num_eval_batches ${NUM_EVAL_BATCHES} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --save_trainable ${SAVE_TRAINABLE} \
  --save_full_model ${SAVE_FULL_MODEL} \
  --save_full_every_checkpoint ${SAVE_FULL_EVERY_CHECKPOINT} \
  --full_save_max_shard_size ${FULL_SAVE_MAX_SHARD_SIZE} \
  \
  --use_wandb false
