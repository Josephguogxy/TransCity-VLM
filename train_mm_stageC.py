# train_mm_stageC.py
# Stage-C finetune: MixLoRA(FFN) only, decoder backbone frozen.
# - NO token adding, NO tokenizer resize, NO embedding resize
# - Keep StageB structure hyperparams consistent (num_image_tokens=32, etc.)
# - Use HF default sampler (Trainer default)

from __future__ import annotations

import os
import sys
import math
import logging
import inspect
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from datasets import concatenate_datasets, Dataset as HFDataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer, set_seed

from dataprocessing import load_jsonl_dataset, ensure_smartcity_features, DataCollatorSmartCity
from model.builder import build_smartcity_llm

logger = logging.getLogger(__name__)


# =========================
# Args
# =========================
@dataclass
class ModelArguments:
    decoder_name_or_path: str = field(metadata={"help": "LLM decoder (e.g. Qwen3-8B) local path"})
    encoder_name_or_path: str = field(metadata={"help": "Chunk text encoder (e.g. roberta-large) local path"})

    # multimodal structure (MUST match StageB if you want load_full_state_dict to load projector/vpma weights)
    num_chunk_tokens: int = 4
    adapter_heads: int = 4
    encoder_max_length: int = 512

    # ✅ StageC: keep consistent with StageB (was 32 in your StageB)
    num_image_tokens: int = 32
    imagebind_variant: str = "google/siglip-so400m-patch14-384"
    group_layers: int = 2
    group_heads: int = 8
    group_tau: float = 1.0

    # Stage‑C：only train MixLoRA FFN, freeze decoder backbone + MM modules
    open_roberta: bool = False
    open_vpma: bool = False
    open_imagebind: bool = False
    open_image_projector: bool = False
    freeze_decoder: bool = True

    # loss weights
    alpha_understand: float = 0.2
    beta_prediction: float = 1.0
    beta_reason: float = 1.0
    lambda_rat: float = 0.3

    # legacy load args (kept)
    mm_load_dir: Optional[str] = field(default=None)
    stageb_ckpt_dir: Optional[str] = field(default=None)

    # ✅ full load dir (StageB checkpoint dir with *.index.json)
    full_model_load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "StageB checkpoint dir (contains model.safetensors.index.json or pytorch_model.bin.index.json)"},
    )

    # tokenizer pad (NO add token; must exist in vocab, otherwise fallback eos)
    padding_token: Optional[str] = field(default="<|pad|>")

    # MixLoRA (FFN only)
    use_mixlora: bool = True
    mixlora_num_experts: int = 20
    mixlora_r: int = 8
    mixlora_alpha: float = 16.0
    mixlora_share_router_for_wi: bool = True
    mixlora_enable_attention: bool = False
    mixlora_ensure_nonzero_gating: bool = True

    # experts split by counts (U/P/R)
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
    max_chunks_per_sample: Optional[int] = None

    def __post_init__(self):
        for flist in (self.train_files, self.validation_files):
            for fp in flist:
                assert fp.endswith(".jsonl"), f"Expect *.jsonl data files, got {fp!r}"


# =========================
# Tokenizer helpers (NO add, NO resize)
# =========================
def _token_exists(tok, token: str) -> bool:
    tid = tok.convert_tokens_to_ids(token)
    if tid is None:
        return False
    unk = getattr(tok, "unk_token_id", None)
    return (unk is None) or (tid != unk)


def _setup_tokenizer(model_args: ModelArguments) -> AutoTokenizer:
    """
    ✅ NO add token / NO resize
    - tokenizer source: prefer full_model_load_dir (StageB) to keep consistent
    - pad_token: only set to existing token; else fallback to eos
    """
    tok_src = model_args.full_model_load_dir or model_args.stageb_ckpt_dir or model_args.decoder_name_or_path
    tok = AutoTokenizer.from_pretrained(
        tok_src,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True,
    )

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run safely.")

    if tok.pad_token_id is None:
        pad_str = (model_args.padding_token or "").strip()
        if pad_str and (pad_str in tok.get_vocab()) and _token_exists(tok, pad_str):
            tok.pad_token = pad_str
        else:
            tok.pad_token = tok.eos_token  # no vocab change

    # optional warning: qwen <think> might be multi-token
    for t in ("<think>", "</think>"):
        if not _token_exists(tok, t):
            logger.warning("Tokenizer doesn't contain %r as a single token; it may be split.", t)

    logger.info("[tok] src=%s | class=%s fast=%s | len=%d vocab_size=%s | pad=%r(%s) eos=%r(%s)",
                tok_src, tok.__class__.__name__, getattr(tok, "is_fast", None),
                len(tok), getattr(tok, "vocab_size", None),
                tok.pad_token, tok.pad_token_id, tok.eos_token, tok.eos_token_id)
    return tok


def _tokenizer_effective_vocab(tok: AutoTokenizer) -> int:
    vocab = tok.get_vocab()
    max_id = max(vocab.values()) if vocab else -1
    for t in tok.all_special_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            max_id = max(max_id, tid)
    return int(max_id + 1)


def _assert_vocab_safe_no_resize(model, tok: AutoTokenizer) -> None:
    # ZeRO-3 下很多 rank 的参数可能是分片/空的，这个检查只在 rank0 做
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    dec = getattr(model, "decoder", model)
    emb = dec.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        return

    # tokenizer 的有效 vocab
    tgt_eff = int(_tokenizer_effective_vocab(tok))

    # 先用 embedding 行数；如果因为 ZeRO-3 读到 0，就用 config.vocab_size 兜底
    cur = int(emb.weight.size(0))
    if cur == 0:
        cfg = getattr(dec, "config", None)
        cfg_vocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
        if cfg_vocab is not None:
            cur = int(cfg_vocab)

    if cur <= 0:
        logger.warning("[skip] cannot infer model vocab rows under ZeRO-3 (got %s).", cur)
        return

    if tgt_eff > cur:
        raise ValueError(
            f"[FATAL] tokenizer_effective_vocab({tgt_eff}) > model_embedding_rows({cur}) -> token id OOB risk. "
            "You said NO resize, so please fix tokenizer/model pairing."
        )

    if cur != tgt_eff:
        logger.warning(
            "[OK] padded vocab: model_embedding_rows=%d, tokenizer_effective_vocab=%d (len(tok)=%d). No resize performed.",
            cur, tgt_eff, len(tok)
        )



# =========================
# Model helpers
# =========================
def _freeze_embeddings(model) -> None:
    """Extra safety: never train embeddings / lm_head."""
    try:
        dec = getattr(model, "decoder", model)
        emb = dec.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad_(False)

        head = None
        try:
            head = dec.get_output_embeddings()
        except Exception:
            head = getattr(dec, "lm_head", None)
        if head is not None and hasattr(head, "weight"):
            head.weight.requires_grad_(False)
    except Exception as e:
        logger.warning("freeze_embeddings failed: %r", e)


# =========================
# Dataset helpers
# =========================
def _call_load_jsonl_dataset(path: str, split_name: str):
    try:
        sig = inspect.signature(load_jsonl_dataset)
        if "split_name" in sig.parameters:
            return load_jsonl_dataset(path, split_name=split_name)
        return load_jsonl_dataset(path)
    except Exception:
        return load_jsonl_dataset(path, split_name=split_name)


def _call_ensure_features(ds: HFDataset, num_proc: int) -> HFDataset:
    """
    Keep user's prompts; avoid double marker injection if your data already has markers.
    """
    sig = inspect.signature(ensure_smartcity_features)
    kwargs: Dict[str, Any] = {"num_proc": num_proc}
    if "ensure_markers" in sig.parameters:
        kwargs["ensure_markers"] = False
    # do not force prompt_style here
    return ensure_smartcity_features(ds, **kwargs)


def _infer_task_tag_from_filename(filename: str) -> Optional[str]:
    """
    StageC: 3 tasks
      - prediction
      - reason
      - understand
    """
    name = os.path.basename(filename).lower()

    # prediction
    if ("pred" in name) or ("predict" in name):
        return "prediction"
    # reason
    if ("reas" in name) or ("reason" in name):
        return "reason"
    # understand (handle your 'unsd' file)
    if ("unsd" in name) or ("understand" in name) or ("uns" in name):
        return "understand"

    return None


def _load_and_merge(files: List[str], split_name: str, seed: int) -> HFDataset:
    assert len(files) > 0, f"No {split_name} files provided."
    dss: List[HFDataset] = []

    for p in files:
        ds_i = _call_load_jsonl_dataset(p, split_name=split_name)

        tag = _infer_task_tag_from_filename(p)

        # if filename doesn't imply, require column `task`
        if tag is None and ("task" not in ds_i.column_names):
            raise ValueError(
                f"File {p} has no 'task' column and filename doesn't contain pred/reas/unsd keywords. "
                "Either add a 'task' field per row or rename file with 'pred'/'reas'/'unsd'."
            )

        # set/overwrite task when tag inferred from filename
        if tag is not None:
            def _set_task(batch, tag: str):
                first_key = next(iter(batch))
                n = len(batch[first_key])
                return {"task": [tag] * n}
            ds_i = ds_i.map(_set_task, fn_kwargs={"tag": tag}, batched=True, batch_size=10000)

        dss.append(ds_i)

    ds = dss[0] if len(dss) == 1 else concatenate_datasets(dss)

    # histogram before ensure_features
    try:
        from collections import Counter
        cnt = Counter(ds["task"])
        logger.info("[data] merged task histogram(before ensure): %s | total=%d", dict(cnt), len(ds))
    except Exception:
        pass

    num_proc = int(os.environ.get("SMARTCITY_MAP_NUM_PROC", "16"))
    ds = _call_ensure_features(ds, num_proc=num_proc)
    ds = ds.shuffle(seed=seed)

    # histogram after ensure_features
    try:
        from collections import Counter
        cnt = Counter(ds["task"])
        logger.info("[data] merged task histogram(after ensure): %s | total=%d", dict(cnt), len(ds))
    except Exception:
        pass

    return ds


# =========================
# Trainer (use HF default sampler)
# =========================
class TrafficTrainer(Trainer):
    """
    Do NOT override sampler/batch_sampler.
    This keeps HF default (DistributedSampler in DDP).
    """
    pass


# =========================
# Main
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

    # ===== tokenizer (NO add / NO resize) =====
    tok = _setup_tokenizer(model_args)

    # ===== encoder tokenizer (collator chunking optional) =====
    enc_tok = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ===== MixLoRA config overrides =====
    _u = model_args.mixlora_num_universal
    _p = model_args.mixlora_num_pred
    _r = model_args.mixlora_num_reason
    _sum_counts = ((_u or 0) + (_p or 0) + (_r or 0)) or None

    mix_cfg_overrides: Dict[str, Any] = dict(
        num_experts=(_sum_counts or model_args.mixlora_num_experts),
        lora_r=model_args.mixlora_r,
        lora_alpha=model_args.mixlora_alpha,
        enable_attention=bool(model_args.mixlora_enable_attention),
        enable_ffn=True,
        share_router_for_wi=bool(model_args.mixlora_share_router_for_wi),
        num_tasks=3,
        ensure_nonzero_gating=bool(model_args.mixlora_ensure_nonzero_gating),
        num_universal=model_args.mixlora_num_universal,
        num_pred=model_args.mixlora_num_pred,
        num_reason=model_args.mixlora_num_reason,
        rank_universal_mul=model_args.mixlora_rank_universal_mul,
        # Qwen/Llama-style FFN module names
        target_modules_override={
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [], "k": [], "v": [], "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        },
    )

    # ===== build model (structure MUST match StageB) =====
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

    # sync decoder pad/eos (no vocab change)
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    # ===== load StageB full weights =====
    if model_args.full_model_load_dir:
        logger.info("Loading full model weights from: %s", model_args.full_model_load_dir)
        model.load_full_state_dict(model_args.full_model_load_dir, cast_to_target_dtype=True)

    # sync again (safety)
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    # ✅ NO resize, only safety check
    _assert_vocab_safe_no_resize(model, tok)

    # ✅ never train embeddings / head
    _freeze_embeddings(model)

    # gradient checkpointing
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
            try:
                model.decoder.enable_input_require_grads()
            except Exception:
                pass

    # ===== dataset =====
    assert len(data_args.train_files) > 0, "请通过 --train_files 提供至少一个 *.jsonl"
    train_ds = _load_and_merge(data_args.train_files, "train", training_args.seed)
    eval_ds = _load_and_merge(data_args.validation_files, "validation", training_args.seed) if data_args.validation_files else None

    if data_args.max_train_samples:
        train_ds = train_ds.select(range(min(len(train_ds), int(data_args.max_train_samples))))
    if data_args.max_eval_samples and (eval_ds is not None):
        eval_ds = eval_ds.select(range(min(len(eval_ds), int(data_args.max_eval_samples))))

    if len(train_ds) == 0:
        raise ValueError("训练集为空，请检查 --train_files")
    if (eval_ds is None) or (len(eval_ds) == 0):
        training_args.evaluation_strategy = "no"
        logger.warning("评估集为空：已自动将 evaluation_strategy='no'。")

    # ===== collator (signature-compatible) =====
    collator_kwargs: Dict[str, Any] = dict(
        max_length=data_args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=model_args.num_chunk_tokens,
        image_token_factor=model_args.num_image_tokens,
        max_images_per_sample=data_args.max_images_per_sample,
        image_size=data_args.image_size,
        max_chunks_per_sample=data_args.max_chunks_per_sample,
    )

    # optional args: only pass if DataCollatorSmartCity supports them
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

        # ✅ the two you mentioned
        if "add_eos_to_labels" in sig.parameters:
            collator_kwargs["add_eos_to_labels"] = True
        if "wrap_reason_with_tags" in sig.parameters:
            collator_kwargs["wrap_reason_with_tags"] = True
    except Exception:
        pass

    collator = DataCollatorSmartCity(tok, **collator_kwargs)

    # ===== trainer =====
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
        # graceful shutdown
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
