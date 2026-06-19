#!/usr/bin/env bash
set -e

# ====== 环境 ======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG_SUBSYS=

# ====== 路径 ======
DECODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"
ENCODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"
FULL_MODEL_LOAD_DIR="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/checkpoint/stageA"
RESUME_CKPT="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/output_stageB/checkpoint-5000"

OUTPUT_DIR="output_stageB"
DS_CFG="ds_config_zero3_hfopt.json"

# ====== 训练超参（只需要 decoder 的 LR）======
PER_DEVICE_BS=4
GRAD_ACC=8
WEIGHT_DECAY=0.01
EPOCHS=1
MAXLEN=4610
LOG_STEPS=10
SAVE_STEPS=400
EVAL_STEPS=400
EVAL_SAMPLES=1000
DEC_LR=5e-6
MM_LR=2e-4
WARMUP=0.03

# ====== 概念 token 个数 ======
NUM_CHUNK_TOKENS=4
NUM_IMAGE_TOKENS=32

deepspeed --num_gpus 8 train_mm_stageB.py \
  --deepspeed ${DS_CFG} \
  --output_dir ${OUTPUT_DIR} \
  --resume_from_checkpoint ${RESUME_CKPT} \
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
