# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from dataprocessing import (
    DataCollatorSmartCity,
    ensure_smartcity_features,
    load_jsonl_dataset,
)
from model.builder import build_smartcity_llm

logger = logging.getLogger(__name__)


# =========================
# Arguments (aligned with training scripts)
# =========================
@dataclass
class ModelArguments:
    decoder_name_or_path: str = field(
        metadata={"help": "LLM decoder name/path (e.g. Qwen/Qwen3-8B-Instruct or a local HF directory)."}
    )
    encoder_name_or_path: str = field(
        metadata={"help": "Chunk text encoder name/path (e.g. FacebookAI/roberta-large)."}
    )

    # ---- multimodal structure ----
    num_chunk_tokens: int = 4
    adapter_heads: int = 4
    encoder_max_length: int = 512

    num_image_tokens: int = 4
    imagebind_variant: str = "huge"
    group_layers: int = 2
    group_heads: int = 8
    group_tau: float = 1.0

    # These flags are mainly used to set requires_grad in training.
    # For inference, they can stay False.
    open_roberta: bool = False
    open_vpma: bool = False
    open_imagebind: bool = False
    open_image_projector: bool = False
    freeze_decoder: bool = False

    # Loss weights (kept for API compatibility; does not change generation behavior).
    alpha_understand: float = 0.2
    beta_prediction: float = 1.0
    beta_reason: float = 1.0
    lambda_rat: float = 0.3

    # Optional: legacy stage-A mm checkpoint root (usually unused in this script).
    mm_load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional: stage-A mm checkpoint root (usually unused for stage-B inference)."},
    )

    padding_token: Optional[str] = field(
        default="<|pad|>",
        metadata={
            "help": (
                "If the tokenizer has no pad_token, try to set pad_token to this EXISTING token. "
                "No new tokens will be added (no vocab expansion)."
            )
        },
    )

    full_model_load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of a trained checkpoint to load via model.load_full_state_dict()."},
    )


@dataclass
class DataInferenceArguments:
    test_file: str = field(metadata={"help": "Test set jsonl path."})
    max_eval_samples: Optional[int] = None     # legacy cap (kept)
    max_infer_samples: Optional[int] = None    # actual cap

    # Keep consistent with training
    max_length: int = 2048
    max_images_per_sample: int = 1
    image_size: int = 224

    # Generation
    gen_max_new_tokens: int = 512
    gen_num_beams: int = 1  # kept for compatibility (custom generate may not use beam search)

    # MAPE
    mape_eps: float = 1e-3


# =========================
# Metrics
# =========================
def compute_metrics_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_eps: float = 1e-3,
) -> Dict[str, float]:
    """
    Compute basic metrics on MW scale.
    """
    yt = y_true.astype(np.float64).reshape(-1)
    yp = y_pred.astype(np.float64).reshape(-1)
    diff = yp - yt

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))

    denom = np.abs(yt)
    mask = denom >= mape_eps
    if np.any(mask):
        mape = float(np.mean(np.abs(diff[mask] / denom[mask])) * 100.0)
    else:
        mape = float("nan")

    wden = float(np.sum(np.abs(yt)))
    wmape = float(np.sum(np.abs(diff)) / wden * 100.0) if wden > 0 else float("nan")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE%": mape, "wMAPE%": wmape}


# =========================
# Distributed helpers
# =========================
def get_dist_info() -> Tuple[int, int, int]:
    """
    Return (rank, world_size, local_rank).
    If not launched by torchrun, it falls back to single-process: rank=0, world_size=1, local_rank=0.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


# =========================
# Dataset
# =========================
def load_test_dataset(path: str, seed: int) -> HFDataset:
    assert path.endswith(".jsonl"), f"Expect *.jsonl test file, got {path!r}"
    ds = load_jsonl_dataset(path, split_name="test")

    # Prefer a configurable num_proc for open-source environments
    num_proc = int(os.environ.get("SMARTCITY_MAP_NUM_PROC", "16"))
    try:
        sig = inspect.signature(ensure_smartcity_features)
        kwargs = {}
        if "num_proc" in sig.parameters:
            kwargs["num_proc"] = num_proc
        ds = ensure_smartcity_features(ds, **kwargs)
    except Exception:
        ds = ensure_smartcity_features(ds, num_proc=num_proc)

    ds = ds.shuffle(seed=seed)
    return ds


# =========================
# Task type inference (traffic vs electricity)
# =========================
def infer_task_type(test_file: str) -> str:
    name = os.path.basename(test_file).lower()
    if "elec" in name or "electric" in name or "power" in name:
        return "elec"
    if "traffic" in name or "flow" in name:
        return "traffic"
    return "generic"


# =========================
# Number extraction
# =========================
_NUM_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")  # float (+ scientific notation)


def _extract_segment(text: str, task: str) -> str:
    """
    Extraction priority:
      1) JSON-like snippet: {"Next 12 flow values": [ ... ]}
      2) keyword + [ ... ]
      3) any [ ... ]
      4) full text
    """
    # 1) Task-specific JSON keys
    if task == "traffic":
        json_keys = [
            r"Next\s+\d+\s+flow\s+values",
            r"Next\s+\d+\s+traffic\s+values",
        ]
    elif task == "elec":
        json_keys = [
            r"Next\s+\d+\s+elec(?:tric(?:ity)?)?\s+values",
            r"Next\s+\d+\s+power\s+values",
            r"Next\s+\d+\s+load\s+values",
        ]
    else:
        json_keys = []

    for key_pat in json_keys:
        pat = r'\{\s*"' + key_pat + r'"\s*:\s*\[([^\]]+)\]\s*\}'
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            inner = m.group(1)
            if inner and inner.strip():
                return inner

    # 2) keyword + bracket
    kw_patterns: List[str] = []
    if task == "elec":
        kw_patterns = [
            r"(?:电力|用电|负荷|负载|load|power|electric(?:ity)?)[^:\n]*[:：]\s*\[([^\]]+)\]",
            r"(?:电力|用电|负荷|负载|load|power|electric(?:ity)?)[^:\n]*[:：](.*)$",
        ]
    elif task == "traffic":
        kw_patterns = [
            r"(?:交通|车流|流量|traffic|flow)[^:\n]*[:：]\s*\[([^\]]+)\]",
            r"(?:交通|车流|流量|traffic|flow)[^:\n]*[:：](.*)$",
        ]

    for pat in kw_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            seg = m.group(1)
            if seg and seg.strip():
                return seg

    # 3) any bracket
    m = re.search(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if m:
        seg = m.group(1)
        if seg and seg.strip():
            return seg

    # 4) fallback
    return text


def extract_numbers(text: str, task: str) -> List[float]:
    """
    Extract float numbers from the model output / target text.
    """
    seg = _extract_segment(text or "", task)
    nums = _NUM_PATTERN.findall(seg)
    return [float(x) for x in nums]


# =========================
# Tokenizer (NO new tokens / NO vocab expansion)
# =========================
def setup_tokenizer_no_modify(model_args: ModelArguments) -> AutoTokenizer:
    """
    Strict tokenizer rule for open-source reproducibility:
      - DO NOT call add_special_tokens()
      - DO NOT resize embeddings
      - pad_token must be chosen from an EXISTING token in vocab; otherwise fallback to eos_token
      - Prefer loading tokenizer from full_model_load_dir if provided (to match checkpoints)
    """
    primary = model_args.full_model_load_dir or model_args.decoder_name_or_path
    fallback = model_args.decoder_name_or_path

    try:
        tok = AutoTokenizer.from_pretrained(
            primary,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )
        tok_src = primary
    except Exception as e:
        logger.warning("Failed to load tokenizer from %r (%s). Fallback to %r.", primary, e, fallback)
        tok = AutoTokenizer.from_pretrained(
            fallback,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )
        tok_src = fallback

    tok.padding_side = "left"

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run safely.")

    if tok.pad_token_id is None:
        # Candidate order: user-specified -> common tokens -> eos
        candidates = []
        if model_args.padding_token:
            candidates.append(model_args.padding_token)
        candidates += ["<|pad|>", "<|endoftext|>", "<|im_end|>"]

        chosen = None
        for cand in candidates:
            if not cand:
                continue
            if cand not in tok.get_vocab():
                continue  # do NOT add new tokens
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tok.eos_token_id:
                chosen = cand
                break

        if chosen is not None:
            tok.pad_token = chosen
        else:
            tok.pad_token = tok.eos_token  # last resort (still no vocab expansion)

    if tok.pad_token_id is None:
        raise ValueError("Tokenizer still has no pad_token_id after setup_tokenizer_no_modify().")

    if tok.pad_token_id == tok.eos_token_id:
        logger.warning(
            "pad_token_id == eos_token_id == %s. This is allowed in no-vocab-expansion mode, "
            "but ensure your attention_mask/padding logic is compatible.",
            str(tok.eos_token_id),
        )

    logger.info("[tok] loaded from: %s | pad=%r(%s) eos=%r(%s)",
                tok_src, tok.pad_token, tok.pad_token_id, tok.eos_token, tok.eos_token_id)
    return tok


def assert_vocab_match_no_resize(model, tok: AutoTokenizer) -> None:
    """
    Fail fast if model embedding vocab_size != tokenizer vocab_size.
    We do NOT call resize_token_embeddings() in this repo mode.
    """
    dec = getattr(model, "decoder", model)
    try:
        emb = dec.get_input_embeddings()
    except Exception:
        emb = None

    if emb is None or (not hasattr(emb, "weight")):
        return

    cur = int(emb.weight.size(0))
    tgt = int(len(tok))
    if cur != tgt:
        raise ValueError(
            f"Vocab mismatch: model embedding vocab_size={cur} vs tokenizer vocab_size={tgt}.\n"
            "This script is configured to NOT add tokens and NOT resize embeddings.\n"
            "Fix: load tokenizer from the exact checkpoint directory that matches the weights "
            "(prefer --full_model_load_dir if it contains tokenizer files), or use a consistent base model+tokenizer."
        )


# =========================
# Collator (signature-compatible)
# =========================
def build_collator(
    tok: AutoTokenizer,
    enc_tok: AutoTokenizer,
    model_args: ModelArguments,
    data_args: DataInferenceArguments,
) -> DataCollatorSmartCity:
    kwargs: Dict = dict(
        max_length=data_args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=model_args.num_chunk_tokens,
        image_token_factor=model_args.num_image_tokens,
        max_images_per_sample=data_args.max_images_per_sample,
        image_size=data_args.image_size,
    )

    # Pass optional chunking controls only if supported by your DataCollatorSmartCity version.
    try:
        sig = inspect.signature(DataCollatorSmartCity.__init__)
        if "chunk_tokenizer" in sig.parameters:
            kwargs["chunk_tokenizer"] = enc_tok
        if "chunk_size_tokens" in sig.parameters:
            kwargs["chunk_size_tokens"] = 510
        if "chunk_overlap" in sig.parameters:
            kwargs["chunk_overlap"] = 0
        if "join_fields_before_chunk" in sig.parameters:
            kwargs["join_fields_before_chunk"] = True
        if "add_field_header" in sig.parameters:
            kwargs["add_field_header"] = True
    except Exception:
        pass

    return DataCollatorSmartCity(tok, **kwargs)


# =========================
# Main
# =========================
def main():
    parser = HfArgumentParser((ModelArguments, DataInferenceArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank, world_size, local_rank = get_dist_info()
    is_main = (rank == 0)

    # logging & seed
    logging.basicConfig(
        format=f"[%(asctime)s] [rank {rank}/{world_size}] %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level() if is_main else logging.ERROR)
    set_seed(int(training_args.seed) + int(rank))

    # tokenizer (no new tokens / no resize)
    tok = setup_tokenizer_no_modify(model_args)

    # encoder tokenizer (for chunking, if collator supports it)
    enc_tok = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    # model
    model = build_smartcity_llm(
        decoder_name=model_args.decoder_name_or_path,
        encoder_name=model_args.encoder_name_or_path,
        alpha_understand=model_args.alpha_understand,
        beta_prediction=model_args.beta_prediction,
        beta_reason=model_args.beta_reason,
        lambda_rat=model_args.lambda_rat,
        num_chunk_tokens=model_args.num_chunk_tokens,
        adapter_heads=model_args.adapter_heads,
        encoder_max_length=model_args.encoder_max_length,
        imagebind_variant=model_args.imagebind_variant,
        num_image_tokens=model_args.num_image_tokens,
        group_layers=model_args.group_layers,
        group_heads=model_args.group_heads,
        group_tau=model_args.group_tau,
        open_roberta=model_args.open_roberta,
        open_vpma=model_args.open_vpma,
        open_imagebind=model_args.open_imagebind,
        open_image_projector=model_args.open_image_projector,
        freeze_decoder=model_args.freeze_decoder,
    )

    # load trained weights (e.g. stage-B output_dir)
    if model_args.full_model_load_dir:
        if is_main:
            logger.info("Loading full model weights from %s", model_args.full_model_load_dir)
        model.load_full_state_dict(model_args.full_model_load_dir, cast_to_target_dtype=True)

    # sync pad_token_id to config
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    # strict check: no resizing allowed
    assert_vocab_match_no_resize(model, tok)

    # device placement
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    # dataset & manual rank split
    full_ds = load_test_dataset(data_args.test_file, int(training_args.seed))

    if data_args.max_eval_samples is not None:
        full_ds = full_ds.select(range(min(len(full_ds), int(data_args.max_eval_samples))))
    if data_args.max_infer_samples is not None:
        full_ds = full_ds.select(range(min(len(full_ds), int(data_args.max_infer_samples))))

    total_samples = len(full_ds)
    if total_samples == 0:
        raise ValueError(f"Empty test set: {data_args.test_file}")

    indices = [i for i in range(total_samples) if i % world_size == rank]
    test_ds = full_ds.select(indices)

    base = os.path.basename(data_args.test_file).replace(".jsonl", "")
    task_type = infer_task_type(data_args.test_file)

    if is_main:
        logger.info("Total=%d, world_size=%d, rank0_split≈%d", total_samples, world_size, total_samples // max(1, world_size))
        logger.info("Inferred task type from filename: %s -> %s", base, task_type)
    logger.info("Rank %d has %d samples.", rank, len(test_ds))

    collator = build_collator(tok, enc_tok, model_args, data_args)

    eval_bs = max(1, int(training_args.per_device_eval_batch_size))
    test_dl = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collator,
        num_workers=int(training_args.dataloader_num_workers),
        pin_memory=bool(training_args.dataloader_pin_memory),
    )

    # inference + number reconstruction
    sample_true_seqs: List[List[float]] = []
    sample_pred_seqs: List[List[float]] = []
    raw_records: List[Dict] = []

    pbar = tqdm(total=total_samples, desc=base, position=0, disable=(not is_main))

    # Use decoder embedding device/dtype as reference for modality encoders
    dec = getattr(model, "decoder", model)
    dec_emb = dec.get_input_embeddings()
    dev_ref, dtype_ref = dec_emb.weight.device, dec_emb.weight.dtype

    for batch in test_dl:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        B = int(batch["prefix_ids"].size(0))

        with torch.no_grad():
            chunk_embeds, chunk_mask, image_embeds, image_mask, _ = model.encode_modalities(
                chunk_txts_all=batch.get("chunk_txts", None),
                pixel_values=batch.get("pixel_values", None),
                n_images=batch.get("n_images", None),
                batch_size=B,
                device=dev_ref,
                dtype=dtype_ref,
            )

            gen_only_ids = model.generate_with_modalities_cached(
                prefix_ids=batch["prefix_ids"],
                prefix_mask=batch["prefix_mask"],
                suffix_ids=batch["suffix_ids"],
                suffix_mask=batch["suffix_mask"],
                chunk_embeds=chunk_embeds,
                chunk_mask=chunk_mask,
                image_embeds=image_embeds,
                image_mask=image_mask,
                task_ids=batch.get("task_ids", None),
                max_new_tokens=int(data_args.gen_max_new_tokens),
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        pred_texts = tok.batch_decode(gen_only_ids, skip_special_tokens=True)

        labels_ids = batch["labels"].detach().clone()
        labels_ids[labels_ids == -100] = tok.pad_token_id
        target_texts = tok.batch_decode(labels_ids, skip_special_tokens=True)

        for true_text, pred_text in zip(target_texts, pred_texts):
            vals_true = extract_numbers(true_text, task_type)
            vals_pred = extract_numbers(pred_text, task_type)

            raw_records.append(
                {
                    "target_text": true_text,
                    "pred_text": pred_text,
                    "target_values": vals_true,
                    "pred_values": vals_pred,
                }
            )

            if len(vals_true) == 0 or len(vals_pred) == 0:
                continue

            L = min(len(vals_true), len(vals_pred))
            sample_true_seqs.append(vals_true[:L])
            sample_pred_seqs.append(vals_pred[:L])

        # rank0 progress is approximate (B * world_size)
        if is_main:
            inc = B * world_size
            remaining = total_samples - pbar.n
            if remaining > 0:
                pbar.update(min(inc, remaining))

    pbar.close()

    # write per-rank raw jsonl
    os.makedirs(training_args.output_dir, exist_ok=True)
    raw_path_rank = os.path.join(training_args.output_dir, f"raw_{base}_rank{rank}.jsonl")
    with open(raw_path_rank, "w", encoding="utf-8") as f:
        for rec in raw_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    logger.info("Rank %d saved %d raw records to %s", rank, len(raw_records), raw_path_rank)

    # gather sequences across ranks (optional)
    merged_true_seqs = sample_true_seqs
    merged_pred_seqs = sample_pred_seqs

    if world_size > 1:
        try:
            import torch.distributed as dist

            initialized_here = False
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available")

            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
                initialized_here = True

            obj_local = (sample_true_seqs, sample_pred_seqs)
            obj_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj_list, obj_local)

            if rank == 0:
                merged_true_seqs = []
                merged_pred_seqs = []
                for t_list, p_list in obj_list:
                    if not t_list or not p_list:
                        continue
                    merged_true_seqs.extend(t_list)
                    merged_pred_seqs.extend(p_list)

            dist.barrier()
            if initialized_here:
                dist.destroy_process_group()

        except Exception as e:
            if is_main:
                logger.warning("Distributed gather failed; metrics will use rank0/local data only. Error: %s", e)

    # rank0 metrics + save labels/preds
    if rank == 0:
        if not merged_true_seqs:
            logger.warning("No numeric sequences extracted. Metrics and labels/preds will be empty/NaN.")
            metrics = {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "wMAPE%": float("nan")}
        else:
            T = min(len(s) for s in merged_true_seqs)
            labels_arr = np.asarray([s[:T] for s in merged_true_seqs], dtype=np.float64)
            preds_arr = np.asarray([s[:T] for s in merged_pred_seqs], dtype=np.float64)

            labels_path = os.path.join(training_args.output_dir, f"labels_{base}.txt")
            preds_path = os.path.join(training_args.output_dir, f"preds_{base}.txt")
            np.savetxt(labels_path, labels_arr, fmt="%.8f")
            np.savetxt(preds_path, preds_arr, fmt="%.8f")
            logger.info("Saved labels to %s", labels_path)
            logger.info("Saved preds  to %s", preds_path)

            metrics = compute_metrics_np(labels_arr, preds_arr, mape_eps=float(data_args.mape_eps))

        metrics_path = os.path.join(training_args.output_dir, f"metrics_{base}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info("==== Final metrics on %s ====", data_args.test_file)
        for k, v in metrics.items():
            logger.info("  %s = %.6f", k, v)
        logger.info("Saved metrics to %s", metrics_path)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
