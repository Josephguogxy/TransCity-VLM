# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import os, sys, math, json, logging, inspect
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import concatenate_datasets, Dataset as HFDataset
from transformers import (
    AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer, set_seed
)

from utils.mixed_pr_sampler import PRBalancedBatchSampler
from dataprocessing import load_jsonl_dataset, ensure_smartcity_features, DataCollatorSmartCity
from model.builder import build_smartcity_llm

logger = logging.getLogger(__name__)

# =========================
# Arguments
# =========================
@dataclass
class ModelArguments:
    decoder_name_or_path: str = field(metadata={"help": "LLM decoder (e.g. Qwen/Qwen3-8B-Instruct)"})
    encoder_name_or_path: str = field(metadata={"help": "Chunk text encoder (e.g. roberta-large)"})
    # Multimodal structure
    num_chunk_tokens: int = 4
    adapter_heads: int = 4
    encoder_max_length: int = 512
    num_image_tokens: int = 4
    imagebind_variant: str = "huge"
    group_layers: int = 2
    group_heads: int = 8
    group_tau: float = 1.0

    # Stage-C: train MixLoRA (FFN only); freeze decoder backbone
    open_roberta: bool = False
    open_vpma: bool = False
    open_imagebind: bool = False
    open_image_projector: bool = False
    freeze_decoder: bool = True

    # Loss weights
    alpha_understand: float = 0.2
    beta_prediction: float = 1.0
    beta_reason: float = 1.0
    lambda_rat: float = 0.3

    # Legacy loading entrypoints (kept but skipped when full_model_load_dir is set)
    mm_load_dir: Optional[str] = field(default=None)
    stageb_ckpt_dir: Optional[str] = field(default=None)

    # Full-model load directory (load_full_state_dict mainly supports ZeRO-3 sharded *.index.json)
    full_model_load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a Trainer.save_model() directory (ideally contains *.index.json for load_full_state_dict)"}
    )

    # Tokenizer pad setup (⚠️ no new tokens: only allow setting an existing vocab token as pad)
    padding_token: Optional[str] = field(default="<|pad|>")

    # MixLoRA (FFN only)
    use_mixlora: bool = True
    mixlora_num_experts: int = 20
    mixlora_r: int = 8
    mixlora_alpha: float = 16.0
    mixlora_share_router_for_wi: bool = True
    mixlora_enable_attention: bool = False
    mixlora_ensure_nonzero_gating: bool = True

    # Configure by counts only; backend auto-assigns indices; supports multiplying U rank
    mixlora_num_universal: Optional[int] = 4
    mixlora_num_pred: Optional[int] = 8
    mixlora_num_reason: Optional[int] = 8
    mixlora_rank_universal_mul: float = 2.0


@dataclass
class DataTrainingArguments:
    train_files: List[str] = field(default_factory=list)
    validation_files: List[str] = field(default_factory=list)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_length: int = 2048
    max_images_per_sample: int = 1
    image_size: int = 224

    def __post_init__(self):
        for flist in (self.train_files, self.validation_files):
            for fp in flist:
                assert fp.endswith(".jsonl"), f"Expect *.jsonl data files, got {fp!r}"


# =========================
# tokenizer helpers
# =========================
def _token_exists(tok, token: str) -> bool:
    tid = tok.convert_tokens_to_ids(token)
    if tid is None:
        return False
    unk = getattr(tok, "unk_token_id", None)
    return (unk is None) or (tid != unk)


def _setup_tokenizer(model_args: ModelArguments) -> AutoTokenizer:
    """
    Do not add new tokens / do not expand the vocab:
    - prefer loading tokenizer from full_model_load_dir (to avoid vocab mismatch)
    - pad_token: prefer the provided padding_token (must already exist in vocab); otherwise pad=eos
    """
    tok_src = model_args.full_model_load_dir or model_args.stageb_ckpt_dir or model_args.decoder_name_or_path
    tok = AutoTokenizer.from_pretrained(
        tok_src,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run safely.")

    # Do not add a new pad token: only set to an existing token or fall back to eos
    if tok.pad_token_id is None:
        pad_str = (model_args.padding_token or "").strip()
        if pad_str and (pad_str in tok.get_vocab()) and _token_exists(tok, pad_str):
            tok.pad_token = pad_str
        else:
            tok.pad_token = tok.eos_token  # do not change vocab_size

    # (Optional) sanity check: Qwen3 often has <think></think>, but they may be split into multiple tokens
    for t in ("<think>", "</think>"):
        if not _token_exists(tok, t):
            logger.warning("Tokenizer vocab doesn't contain %r as a single token; it may be split into sub-tokens.", t)

    print("[tok] src:", tok_src)
    print("[tok] pad:", tok.pad_token, tok.pad_token_id, "| eos:", tok.eos_token, tok.eos_token_id)
    return tok


# =========================
# model helpers
# =========================
def _maybe_resize_to_tokenizer(model, tok):
    """
    NO vocab expansion mode:
    - We DO NOT call resize_token_embeddings().
    - If embedding vocab_size != len(tokenizer), we fail fast to avoid silent mismatch.
    """
    dec = getattr(model, "decoder", model)
    emb = dec.get_input_embeddings() if hasattr(dec, "get_input_embeddings") else None
    if emb is None or (not hasattr(emb, "weight")):
        return
    cur = int(emb.weight.size(0))
    tgt = int(len(tok))
    if cur != tgt:
        raise ValueError(
            f"Vocab mismatch: model embedding vocab_size={cur} vs tokenizer vocab_size={tgt}.\n"
            "This repo is configured to NOT add tokens and NOT resize embeddings.\n"
            "Fix: load tokenizer from the exact checkpoint directory that matches the decoder weights "
            "(use --full_model_load_dir or --stageb_ckpt_dir), or use a base model+tokenizer pair with matching vocab."
        )


def _freeze_embeddings(model):
    """
    Ensure embedding / lm_head are not trained (safety).
    """
    try:
        dec = getattr(model, "decoder", model)
        emb = dec.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad_(False)

        head = dec.get_output_embeddings()
        if head is None:
            head = getattr(dec, "lm_head", None)
        if head is not None and hasattr(head, "weight"):
            head.weight.requires_grad_(False)
    except Exception as e:
        logger.warning("freeze_embeddings failed: %r", e)


# =========================
# dataset load (compat with old/new dataprocessing signatures)
# =========================
def _call_load_jsonl_dataset(path: str, split_name: str):
    """
    Compatibility helper for signature differences in dataprocessing.load_jsonl_dataset.
    """
    try:
        sig = inspect.signature(load_jsonl_dataset)
        if "split_name" in sig.parameters:
            return load_jsonl_dataset(path, split_name=split_name)
        return load_jsonl_dataset(path)
    except Exception:
        return load_jsonl_dataset(path, split_name=split_name)


def _call_ensure_features(ds: HFDataset, num_proc: int = 16) -> HFDataset:
    """
    Compatibility helper for newer ensure_smartcity_features parameters (prompt_style / ensure_markers).
    """
    sig = inspect.signature(ensure_smartcity_features)
    kwargs = {"num_proc": num_proc}
    if "prompt_style" in sig.parameters:
        pass
    if "ensure_markers" in sig.parameters:
        kwargs["ensure_markers"] = False
    return ensure_smartcity_features(ds, **kwargs)


def _load_and_merge(files: List[str], split_name: str, seed: int) -> HFDataset:
    assert len(files) > 0, f"No {split_name} files provided."
    dss = []
    for p in files:
        ds_i = _call_load_jsonl_dataset(p, split_name=split_name)

        name = os.path.basename(p).lower()
        if "pred" in name:
            tag = "prediction"
        elif "reas" in name or "reason" in name:
            tag = "reason"
        else:
            if "task" not in ds_i.column_names:
                raise ValueError(
                    f"File {p} has no 'task' column and filename doesn't contain 'pred'/'reas'. "
                    "Either add a 'task' field per row or rename file with 'pred'/'reas'."
                )
            tag = None

        if tag is not None:
            def _set_task(batch, tag):
                first_key = next(iter(batch))
                n = len(batch[first_key])
                return {"task": [tag] * n}
            ds_i = ds_i.map(_set_task, fn_kwargs={"tag": tag}, batched=True, batch_size=10000)

        dss.append(ds_i)

    ds = dss[0] if len(dss) == 1 else concatenate_datasets(dss)
    try:
        from collections import Counter
        cnt = Counter(ds["task"])
        print(f"[data] merged task histogram: {dict(cnt)} | total={len(ds)}")
    except Exception:
        pass

    ds = _call_ensure_features(ds, num_proc=16)
    ds = ds.shuffle(seed=seed)
    return ds


# =========================
# Custom Trainer (DDP sampling)
# =========================
class TrafficTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None

        if self.args.process_index == 0:
            try:
                from collections import Counter
                hist = dict(Counter(self.train_dataset["task"]))
                print(f"[pre-train CHECK] task histogram(before DDP): {hist} | total={len(self.train_dataset)}")
                keys = set(hist.keys())
                if (len(keys) == 0) or (keys == {""}):
                    raise RuntimeError(
                        "Dataset 'task' column appears empty after ensure_smartcity_features(). "
                        "Check task mapping / normalize logic."
                    )
            except Exception as e:
                print("[pre-train CHECK] failed to compute task histogram:", repr(e))

        dist_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
            seed=self.args.seed,
            drop_last=False,
        )

        batch_sampler = PRBalancedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            pr_per_batch=(3, 1),
            sampler=dist_sampler,
            shuffle=True,
            drop_last=False,
            seed=self.args.seed,
            epoch_batches=None,
            minority_repeat_cap=10,
            task_field="task",
            label_pred="prediction",
            label_reason="reason",
        )

        dl = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return dl


# =========================
# main
# =========================
def main():
    parser_all = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser_all.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser_all.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    set_seed(training_args.seed)

    tok = _setup_tokenizer(model_args)

    enc_tok = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    _u = model_args.mixlora_num_universal
    _p = model_args.mixlora_num_pred
    _r = model_args.mixlora_num_reason
    _sum_counts = ((_u or 0) + (_p or 0) + (_r or 0)) or None

    mix_cfg_overrides: Dict[str, Any] = dict(
        num_experts=(_sum_counts or model_args.mixlora_num_experts),
        lora_r=model_args.mixlora_r,
        lora_alpha=model_args.mixlora_alpha,
        enable_attention=model_args.mixlora_enable_attention,
        enable_ffn=True,
        share_router_for_wi=model_args.mixlora_share_router_for_wi,
        num_tasks=3,
        ensure_nonzero_gating=bool(model_args.mixlora_ensure_nonzero_gating),
        num_universal=model_args.mixlora_num_universal,
        num_pred=model_args.mixlora_num_pred,
        num_reason=model_args.mixlora_num_reason,
        rank_universal_mul=model_args.mixlora_rank_universal_mul,
        target_modules_override={
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [], "k": [], "v": [], "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        },
    )

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
        use_mixlora=model_args.use_mixlora,
        mixlora_cfg_overrides=mix_cfg_overrides,
    )

    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    _maybe_resize_to_tokenizer(model, tok)

    if model_args.full_model_load_dir:
        model.load_full_state_dict(
            model_args.full_model_load_dir,
            cast_to_target_dtype=True
        )

    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    _freeze_embeddings(model)

    if training_args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            setattr(model.decoder.config, "use_cache", False)
        if hasattr(model.decoder, "enable_input_require_grads"):
            model.decoder.enable_input_require_grads()

    assert len(data_args.train_files) > 0, "Please provide at least one *.jsonl via --train_files"
    train_ds = _load_and_merge(data_args.train_files, "train", training_args.seed)
    eval_ds = _load_and_merge(data_args.validation_files, "validation", training_args.seed) if data_args.validation_files else None

    if data_args.max_train_samples:
        train_ds = train_ds.select(range(min(len(train_ds), data_args.max_train_samples)))
    if data_args.max_eval_samples and (eval_ds is not None):
        eval_ds = eval_ds.select(range(min(len(eval_ds), data_args.max_eval_samples)))

    if len(train_ds) == 0:
        raise ValueError("Training set is empty; check --train_files")
    if (eval_ds is None) or (len(eval_ds) == 0):
        training_args.evaluation_strategy = "no"
        logger.warning("Evaluation set is empty: evaluation_strategy has been set to 'no'.")

    collator_kwargs: Dict[str, Any] = dict(
        max_length=data_args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=model_args.num_chunk_tokens,
        image_token_factor=model_args.num_image_tokens,
        max_images_per_sample=data_args.max_images_per_sample,
        image_size=data_args.image_size,
        add_eos_to_labels=True,
        wrap_reason_with_tags=True,
    )

    try:
        sig = inspect.signature(DataCollatorSmartCity.__init__)
        if "chunk_tokenizer" in sig.parameters:
            collator_kwargs["chunk_tokenizer"] = enc_tok
        if "chunk_size_tokens" in sig.parameters:
            collator_kwargs["chunk_size_tokens"] = 510
        if "chunk_overlap" in sig.parameters:
            collator_kwargs["chunk_overlap"] = 0
        if "join_fields_before_chunk" in sig.parameters:
            collator_kwargs["join_fields_before_chunk"] = True
        if "add_field_header" in sig.parameters:
            collator_kwargs["add_field_header"] = True
    except Exception:
        pass

    collator = DataCollatorSmartCity(tok, **collator_kwargs)

    trainer = TrafficTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if (training_args.do_eval and eval_ds is not None) else None,
        tokenizer=tok,
        data_collator=collator,
    )

    try:
        if training_args.do_train:
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            if trainer.is_world_process_zero():
                tok.save_pretrained(training_args.output_dir)

        if training_args.do_eval and (eval_ds is not None):
            metrics = trainer.evaluate()
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except Exception:
                metrics["perplexity"] = float("inf")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
