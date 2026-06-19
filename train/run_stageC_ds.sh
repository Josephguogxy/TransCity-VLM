#!/usr/bin/env bash
set -e

########################
# 环境（关键修改）
########################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_BLOCKING_WAIT
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=$((10000 + RANDOM % 20000))

########################
# 路径
########################
DECODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"
ENCODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"
FULL_MODEL_LOAD_DIR="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/checkpoint/stageB"

TRAIN_FILES="dataset_stageC/train_dataset_pred.jsonl dataset_stageC/train_dataset_reas.jsonl dataset_stageC/train_dataset_unsd.jsonl"
VAL_FILES="dataset_stageC/val_dataset.jsonl"

OUTPUT_DIR="output_stageC"
DS_CFG="ds_config_zero3.json"

########################
# 超参（LoRA 用较高 LR）
########################
PER_DEVICE_BS=4
GRAD_ACC=8
LR=1e-4
WEIGHT_DECAY=0.01
EPOCHS=5
MAXLEN=4096
LOG_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500

NUM_CHUNK_TOKENS=4
NUM_IMAGE_TOKENS=32

deepspeed --num_gpus 8 train_mm_stageC.py \
  --deepspeed ${DS_CFG} \
  --output_dir ${OUTPUT_DIR} \
  --do_train true \
  --do_eval true \
  --eval_strategy "steps" \
  --eval_steps ${EVAL_STEPS} \
  --logging_steps ${LOG_STEPS} \
  --logging_first_step true \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 3 \
  --report_to "none" \
  --per_device_train_batch_size ${PER_DEVICE_BS} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --learning_rate ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_train_epochs ${EPOCHS} \
  --bf16 true \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters false \
  --max_eval_samples 128 \
  --per_device_eval_batch_size 2 \
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
  --image_size 384 \
  --train_files ${TRAIN_FILES} \
  --validation_files ${VAL_FILES}
