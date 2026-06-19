#!/usr/bin/env bash
set -euo pipefail

########################
# Env
########################
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
unset NCCL_BLOCKING_WAIT || true
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$((10000 + RANDOM % 20000))}
export TOKENIZERS_PARALLELISM=false

# StageD SFT warmup: keep labels as freeform answer, not JSON.
export SMARTCITY_REASON_KEEP_FREEFORM_ANSWER=${SMARTCITY_REASON_KEEP_FREEFORM_ANSWER:-1}
export SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE=${SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE:-1}
export SMARTCITY_REASON_ALWAYS_MARKERS=${SMARTCITY_REASON_ALWAYS_MARKERS:-1}
export SMARTCITY_Z3_LEAF_SUFFIXES=${SMARTCITY_Z3_LEAF_SUFFIXES:-mixlora}
export SMARTCITY_MAP_NUM_PROC=${SMARTCITY_MAP_NUM_PROC:-16}

########################
# Paths
########################
DECODER=${DECODER:-"/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"}
ENCODER=${ENCODER:-"/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"}

FULL_MODEL_LOAD_DIR=${FULL_MODEL_LOAD_DIR:-"checkpoint/stageC"}

TRAIN_FILES=${TRAIN_FILES:-"dataset_stageD_sft_warmup/train_dataset_reas.jsonl"}
VAL_FILES=${VAL_FILES:-"dataset_stageD_sft_warmup/val_dataset_reason.jsonl"}

OUTPUT_DIR=${OUTPUT_DIR:-"output_stageC_warmup/full_2epoch"}
DS_CFG=${DS_CFG:-"configs/deepspeed_zero3.json"}
NUM_GPUS=${NUM_GPUS:-8}

########################
# Hyperparams
########################
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-1e-5}
LAMBDA_RAT=${LAMBDA_RAT:-0.2}
BETA_REASON=${BETA_REASON:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
EPOCHS=${EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:--1}
MAXLEN=${MAXLEN:-2048}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}

LOG_STEPS=${LOG_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-256}

NUM_CHUNK_TOKENS=${NUM_CHUNK_TOKENS:-4}
NUM_IMAGE_TOKENS=${NUM_IMAGE_TOKENS:-32}
MAX_IMAGES_PER_SAMPLE=${MAX_IMAGES_PER_SAMPLE:-1}
IMAGE_SIZE=${IMAGE_SIZE:-384}

mkdir -p "${OUTPUT_DIR}"

########################
# Runtime DeepSpeed config
########################
# Important:
# HF Trainer checks DS config against TrainingArguments.
# The original configs/deepspeed_zero3.json may hard-code gradient_accumulation_steps=1.
# Here we generate a runtime config with batch-related fields set to auto.
RUNTIME_DS_CFG="${OUTPUT_DIR}/deepspeed_zero3.stageD_sft.runtime.json"

python - <<PY
import json
from pathlib import Path

src = Path("${DS_CFG}")
dst = Path("${RUNTIME_DS_CFG}")

if not src.is_file():
    raise FileNotFoundError(src)

cfg = json.loads(src.read_text(encoding="utf-8"))

# Let HF Trainer fill these from TrainingArguments.
cfg["train_micro_batch_size_per_gpu"] = "auto"
cfg["gradient_accumulation_steps"] = "auto"
cfg["train_batch_size"] = "auto"

# Avoid optimizer/scheduler conflicts with Trainer-created optimizer.
cfg.pop("optimizer", None)
cfg.pop("scheduler", None)

# Precision should also follow TrainingArguments.
if "fp16" in cfg and isinstance(cfg["fp16"], dict):
    cfg["fp16"]["enabled"] = "auto"
if "bf16" in cfg and isinstance(cfg["bf16"], dict):
    cfg["bf16"]["enabled"] = "auto"
else:
    cfg["bf16"] = {"enabled": "auto"}

# Do not use activation checkpointing for StageD SFT warmup under ZeRO-3.
# PyTorch checkpoint recompute can see ZeRO-3 partitioned tensors with shape [0].
cfg["activation_checkpointing"] = {
    "partition_activations": False,
    "contiguous_memory_optimization": False,
    "cpu_checkpointing": False,
    "synchronize_checkpoint_boundary": False,
}

# Some configs contain wall_clock_breakdown etc.; keep them.

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

print("[ds-runtime] source:", src)
print("[ds-runtime] written:", dst)
print("[ds-runtime] train_micro_batch_size_per_gpu:", cfg.get("train_micro_batch_size_per_gpu"))
print("[ds-runtime] gradient_accumulation_steps:", cfg.get("gradient_accumulation_steps"))
print("[ds-runtime] train_batch_size:", cfg.get("train_batch_size"))
print("[ds-runtime] optimizer_present:", "optimizer" in cfg)
print("[ds-runtime] scheduler_present:", "scheduler" in cfg)
PY

########################
# Version-compatible args
########################
MAX_STEPS_ARGS=()
if [[ "${MAX_STEPS}" != "-1" && "${MAX_STEPS}" != "0" ]]; then
  MAX_STEPS_ARGS=(--max_steps "${MAX_STEPS}")
fi

EVAL_STRATEGY_ARG=$(python - <<'PY'
from transformers import TrainingArguments
fields = getattr(TrainingArguments, "__dataclass_fields__", {})
print("--eval_strategy" if "eval_strategy" in fields else "--evaluation_strategy")
PY
)

SAVE_SAFE_ARGS=()

echo "[StageD SFT warmup]"
echo "  FULL_MODEL_LOAD_DIR=${FULL_MODEL_LOAD_DIR}"
echo "  TRAIN_FILES=${TRAIN_FILES}"
echo "  VAL_FILES=${VAL_FILES}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  global_batch=$((NUM_GPUS * PER_DEVICE_BS * GRAD_ACC))"
echo "  PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "  GRAD_ACC=${GRAD_ACC}"
echo "  EPOCHS=${EPOCHS}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "  LR=${LR}"
echo "  SAVE_STEPS=${SAVE_STEPS}"
echo "  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "  DS_CFG=${DS_CFG}"
echo "  RUNTIME_DS_CFG=${RUNTIME_DS_CFG}"
echo "  EVAL_STRATEGY_ARG=${EVAL_STRATEGY_ARG}"

deepspeed --num_gpus "${NUM_GPUS}" train_stageD_sft_warmup.py \
  --deepspeed "${RUNTIME_DS_CFG}" \
  --output_dir "${OUTPUT_DIR}" \
  --do_train true \
  --do_eval true \
  "${EVAL_STRATEGY_ARG}" "steps" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOG_STEPS}" \
  --logging_first_step true \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --report_to "none" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_train_epochs "${EPOCHS}" \
  "${MAX_STEPS_ARGS[@]}" \
  --bf16 true \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --ddp_find_unused_parameters false \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --per_device_eval_batch_size 1 \
  \
  --decoder_name_or_path "${DECODER}" \
  --encoder_name_or_path "${ENCODER}" \
  --full_model_load_dir "${FULL_MODEL_LOAD_DIR}" \
  \
  --num_chunk_tokens "${NUM_CHUNK_TOKENS}" \
  --num_image_tokens "${NUM_IMAGE_TOKENS}" \
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
  --freeze_mixlora_non_reason true \
  --restore_reason_only_after_step true \
  --freeze_router_hidden true \
  \
  --alpha_understand 0.2 \
  --beta_prediction 1.0 \
  --beta_reason "${BETA_REASON}" \
  --lambda_rat "${LAMBDA_RAT}" \
  \
  --max_length "${MAXLEN}" \
  --max_images_per_sample "${MAX_IMAGES_PER_SAMPLE}" \
  --image_size "${IMAGE_SIZE}" \
  --train_files ${TRAIN_FILES} \
  --validation_files ${VAL_FILES}
