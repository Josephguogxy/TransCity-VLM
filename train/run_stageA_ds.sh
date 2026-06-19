#!/usr/bin/env bash
set -e

# ====== 环境 ======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=info
export TORCH_NCCL_BLOCKING_WAIT=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE=0
export DS_BUILD_OPS=0

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export NCCL_TIMEOUT=600


# ====== 路径 ======
DECODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"
ENCODER="/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"

# 合并后的本地 JSONL（推荐先用 merge 脚本产出）
TRAIN_FILES="dataset_stageA/train_understand_dataset.jsonl"
VAL_FILES="dataset_stageA/test_understand_dataset.jsonl"

OUTPUT_DIR="output_stageA"
DS_CFG="ds_config_zero3.json"

# ====== 训练超参 ======
PER_DEVICE_BS=8          # 每卡 micro-batch
GRAD_ACC=8               # 累计步数
LR=3e-5                  # Stage-A：只训 RoBERTa + Projector，学习率可略大
WEIGHT_DECAY=0.1
EPOCHS=5
MAXLEN=1024
LOG_STEPS=5
SAVE_STEPS=500
EVAL_STEPS=500

# ====== 概念 token 个数（与模型&collator 保持一致） ======
NUM_CHUNK_TOKENS=4
NUM_IMAGE_TOKENS=32

# ====== 启动（deepspeed 双卡 + ZeRO-3）======
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
  --max_images_per_sample 4 \
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

# 训练完成后，train_mm_stageA.py 会：
# 1) 保存 HF 全量权重到 ${OUTPUT_DIR}
# 2) 若实现了 save_multimodal_checkpoint()，同时导出 mm_checkpoint/ （仅 RoBERTa + vPMA + ImageProjector）
