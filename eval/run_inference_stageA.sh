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
CKPT_DIR="${CKPT_DIR:-checkpoint/stageA}"          # your trained stage-A checkpoint dir
DATA_DIR="${DATA_DIR:-dataset_stageA}"             # your dataset folder
OUT_ROOT="${OUT_ROOT:-outputs/inference_stageA}"   # output root
mkdir -p "${OUT_ROOT}"

# ===== Inference settings =====
NUM_CHUNK_TOKENS="${NUM_CHUNK_TOKENS:-4}"
NUM_IMAGE_TOKENS="${NUM_IMAGE_TOKENS:-32}"         # stage-A typical default
IMAGE_SIZE="${IMAGE_SIZE:-384}"                    # stage-A typical default
MAXLEN="${MAXLEN:-4096}"
EVAL_BS="${EVAL_BS:-2}"
MAX_INFER_SAMPLES="${MAX_INFER_SAMPLES:-}"         # optional; leave empty to disable

# If you have a different set of Stage-A tests, replace these names.
TEST_FILES=(
  "test_traffic_3h_prediction.jsonl"
  "test_traffic_12h_prediction.jsonl"
  "test_elec_pred_12h.jsonl"
)

for TEST_FILE in "${TEST_FILES[@]}"; do
  NAME="$(basename "${TEST_FILE}" .jsonl)"
  OUT_DIR="${OUT_ROOT}/${NAME}"
  mkdir -p "${OUT_DIR}"

  echo "==== Stage-A Inference: ${TEST_FILE} -> ${OUT_DIR} ===="

  python inference_mm_stageA.py \
    --decoder_name_or_path "${DECODER_NAME_OR_PATH}" \
    --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
    --full_model_load_dir "${CKPT_DIR}" \
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
    --freeze_decoder false \
    \
    --alpha_understand 1.0 \
    --beta_prediction 0.0 \
    --beta_reason 0.0 \
    --lambda_rat 0.3 \
    \
    --test_file "${DATA_DIR}/${TEST_FILE}" \
    --max_length "${MAXLEN}" \
    --max_images_per_sample 5 \
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
