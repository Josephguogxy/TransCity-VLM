#!/usr/bin/env bash
set -euo pipefail

# Optional: choose which GPU to use (single GPU example)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

# Make sure we are NOT accidentally in torchrun env for this single-GPU script
unset WORLD_SIZE || true
unset RANK || true
unset LOCAL_RANK || true

# ===== Paths (edit via env vars) =====
DECODER_NAME_OR_PATH="${DECODER_NAME_OR_PATH:-Qwen/Qwen3-8B-Instruct}"
ENCODER_NAME_OR_PATH="${ENCODER_NAME_OR_PATH:-FacebookAI/roberta-large}"
CKPT_DIR="${CKPT_DIR:-checkpoint/stageC}"          # your trained stage-C checkpoint dir
DATA_DIR="${DATA_DIR:-dataset_stageC}"             # your dataset folder
OUT_ROOT="${OUT_ROOT:-outputs/inference_stageC}"   # output root
mkdir -p "${OUT_ROOT}"

# ===== Inference settings =====
NUM_CHUNK_TOKENS="${NUM_CHUNK_TOKENS:-4}"
NUM_IMAGE_TOKENS="${NUM_IMAGE_TOKENS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAXLEN="${MAXLEN:-4096}"
EVAL_BS="${EVAL_BS:-2}"
MAX_INFER_SAMPLES="${MAX_INFER_SAMPLES:-}"         # optional; leave empty to disable

# ===== MixLoRA settings (override via env vars) =====
USE_MIXLORA="${USE_MIXLORA:-true}"
MIXLORA_NUM_EXPERTS="${MIXLORA_NUM_EXPERTS:-20}"
MIXLORA_R="${MIXLORA_R:-8}"
MIXLORA_ALPHA="${MIXLORA_ALPHA:-16}"
MIXLORA_SHARE_ROUTER_FOR_WI="${MIXLORA_SHARE_ROUTER_FOR_WI:-true}"
MIXLORA_ENABLE_ATTENTION="${MIXLORA_ENABLE_ATTENTION:-false}"
MIXLORA_ENSURE_NONZERO_GATING="${MIXLORA_ENSURE_NONZERO_GATING:-true}"

MIXLORA_NUM_UNIVERSAL="${MIXLORA_NUM_UNIVERSAL:-4}"
MIXLORA_NUM_PRED="${MIXLORA_NUM_PRED:-8}"
MIXLORA_NUM_REASON="${MIXLORA_NUM_REASON:-8}"
MIXLORA_RANK_UNIVERSAL_MUL="${MIXLORA_RANK_UNIVERSAL_MUL:-2.0}"

TEST_FILES=(
  "test_traffic_3h_prediction.jsonl"
  "test_traffic_12h_prediction.jsonl"
  "test_elec_pred_12h.jsonl"
)

for TEST_FILE in "${TEST_FILES[@]}"; do
  NAME="$(basename "${TEST_FILE}" .jsonl)"
  OUT_DIR="${OUT_ROOT}/${NAME}"
  mkdir -p "${OUT_DIR}"

  echo "==== Stage-C Inference: ${TEST_FILE} -> ${OUT_DIR} ===="

  python inference_mm_stageC.py \
    --decoder_name_or_path "${DECODER_NAME_OR_PATH}" \
    --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
    --full_model_load_dir "${CKPT_DIR}" \
    \
    --num_chunk_tokens "${NUM_CHUNK_TOKENS}" \
    --num_image_tokens "${NUM_IMAGE_TOKENS}" \
    --adapter_heads 4 \
    --encoder_max_length 512 \
    --imagebind_variant "huge" \
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
    --alpha_understand 0.2 \
    --beta_prediction 1.0 \
    --beta_reason 1.0 \
    --lambda_rat 0.3 \
    \
    --use_mixlora "${USE_MIXLORA}" \
    --mixlora_num_experts "${MIXLORA_NUM_EXPERTS}" \
    --mixlora_r "${MIXLORA_R}" \
    --mixlora_alpha "${MIXLORA_ALPHA}" \
    --mixlora_share_router_for_wi "${MIXLORA_SHARE_ROUTER_FOR_WI}" \
    --mixlora_enable_attention "${MIXLORA_ENABLE_ATTENTION}" \
    --mixlora_ensure_nonzero_gating "${MIXLORA_ENSURE_NONZERO_GATING}" \
    --mixlora_num_universal "${MIXLORA_NUM_UNIVERSAL}" \
    --mixlora_num_pred "${MIXLORA_NUM_PRED}" \
    --mixlora_num_reason "${MIXLORA_NUM_REASON}" \
    --mixlora_rank_universal_mul "${MIXLORA_RANK_UNIVERSAL_MUL}" \
    \
    --test_file "${DATA_DIR}/${TEST_FILE}" \
    --max_length "${MAXLEN}" \
    --max_images_per_sample 1 \
    --image_size "${IMAGE_SIZE}" \
    --gen_max_new_tokens 512 \
    --gen_num_beams 1 \
    --mape_eps 1e-3 \
    \
    --per_device_eval_batch_size "${EVAL_BS}" \
    --dataloader_num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --seed 42 \
    $(if [[ -n "${MAX_INFER_SAMPLES}" ]]; then echo --max_infer_samples "${MAX_INFER_SAMPLES}"; fi)

done
