# train_stageD_sft_warmup.py
# StageD-format SFT warmup:
# - Start from StageC full checkpoint
# - Train only reason experts + reason router
# - Teach output format: optional <think>...</think> + natural answer + [E_*] citations
# - Do NOT train JSON output

from __future__ import annotations

import os
import sys
import math
import logging
import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, TrainerCallback, set_seed

from dataprocessing import DataCollatorSmartCity
from model.builder import build_smartcity_llm

# Reuse StageC helpers so the training style/checkpoint behavior stays close to StageC.
from train_mm_stageC import (
    ModelArguments,
    DataTrainingArguments,
    TrafficTrainer,
    _setup_tokenizer,
    _assert_vocab_safe_no_resize,
    _freeze_embeddings,
    _load_and_merge,
)

# Reuse StageD reason-only masks.
from train_grpo import (
    apply_reason_only_mixlora_gradient_mask,
    restore_reason_only_frozen_slices,
    configure_zero3_leaf_modules_for_mixlora,
)

logger = logging.getLogger(__name__)


@dataclass
class StageDSFTWarmupArguments:
    freeze_mixlora_non_reason: bool = field(
        default=True,
        metadata={"help": "Train only R experts and reason router rows via gradient masks."},
    )
    restore_reason_only_after_step: bool = field(
        default=True,
        metadata={"help": "Restore frozen U/P slices after optimizer steps as extra safety."},
    )
    freeze_router_hidden: bool = field(
        default=True,
        metadata={"help": "Freeze hidden linears inside MaskedTokenRouter; train only final R rows."},
    )
    print_trainable_names: bool = field(default=False)
    max_trainable_name_print: int = field(default=80)


def _is_main_process() -> bool:
    rk = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    try:
        return int(rk) in (-1, 0)
    except Exception:
        return True



def _env_flag_false(name: str, default: str = "false") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"0", "false", "no", "off"}


def _force_disable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    ZeRO-3 + torch activation checkpoint recompute can see partitioned params
    with shape [0]. For this StageD SFT warmup, disable all checkpointing.
    """
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception:
        pass

    dec = getattr(model, "decoder", None)
    try:
        if dec is not None and hasattr(dec, "gradient_checkpointing_disable"):
            dec.gradient_checkpointing_disable()
    except Exception:
        pass

    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            try:
                setattr(m, "gradient_checkpointing", False)
            except Exception:
                pass
        if hasattr(m, "activation_checkpointing"):
            try:
                setattr(m, "activation_checkpointing", False)
            except Exception:
                pass
        if hasattr(m, "config"):
            try:
                setattr(m.config, "use_cache", False)
            except Exception:
                pass


def _count_trainable(model: torch.nn.Module) -> Dict[str, int]:
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "trainable": int(trainable),
        "total": int(total),
        "pct_x100": float(trainable / max(1, total) * 100.0),
    }


class ReasonOnlyRestoreCallback(TrainerCallback):
    def __init__(self, active: bool):
        self.active = bool(active)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return control
        try:
            m = getattr(model, "module", model)
            restore_reason_only_frozen_slices(m)
        except Exception as e:
            if _is_main_process():
                print(f"[warn] restore_reason_only_frozen_slices failed: {repr(e)}", flush=True)
        return control



def _stageD_dist_is_initialized():
    import torch
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _stageD_dist_rank():
    import torch
    if _stageD_dist_is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _stageD_dist_broadcast_object(obj, src=0):
    import torch
    if not _stageD_dist_is_initialized():
        return obj
    holder = [obj]
    torch.distributed.broadcast_object_list(holder, src=src)
    return holder[0]


def _stageD_dist_barrier():
    import torch
    if _stageD_dist_is_initialized():
        torch.distributed.barrier()


def _stageD_has_zero3_partitioned_params(model) -> bool:
    """
    Detect DeepSpeed ZeRO-3 partitioned parameters.

    Under ZeRO-3, non-local params can appear as torch.Size([0]).
    Loading a full model.safetensors with plain model.load_state_dict(...)
    then raises size mismatch errors.
    """
    import os

    if os.environ.get("SMARTCITY_STAGE_D_FORCE_ZERO3_GATHER_LOAD", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return True

    for p in model.parameters():
        if hasattr(p, "ds_id") or hasattr(p, "ds_status") or hasattr(p, "ds_numel"):
            return True
        try:
            if tuple(p.shape) == (0,):
                return True
        except Exception:
            pass
    return False


def _stageD_load_safetensors_zero3_aware(model, safetensors_path, verbose=True):
    """
    Load a full root model.safetensors into a model whose parameters are already
    ZeRO-3 partitioned.

    All ranks enter GatheredParameters for each param. Rank 0 copies the full
    checkpoint tensor into the gathered param; DeepSpeed repartitions on exit.
    """
    from pathlib import Path
    import gc
    import torch
    from safetensors.torch import load_file

    try:
        from deepspeed import zero
    except Exception as e:
        raise RuntimeError(
            "ZeRO-3 partitioned parameters were detected, but deepspeed.zero "
            "could not be imported for gathered-parameter loading."
        ) from e

    sf = Path(safetensors_path)
    rank = _stageD_dist_rank()
    is_main = rank == 0

    if verbose and is_main:
        print(f"[stageD-sft-load] ZeRO-3 detected; loading model.safetensors with gathered-parameter copy: {sf}", flush=True)

    # Load the full checkpoint only on rank 0. Other ranks receive key order.
    sd = load_file(str(sf), device="cpu") if is_main else {}
    ckpt_keys = list(sd.keys()) if is_main else None
    ckpt_keys = _stageD_dist_broadcast_object(ckpt_keys, src=0) or []

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    loaded_params = 0
    loaded_buffers = 0
    unexpected = 0
    mismatched = 0
    mismatch_examples = []
    loaded_names = set()

    def resolve_name(name):
        candidates = [name]
        if name.startswith("module."):
            candidates.append(name[len("module."):])
        else:
            candidates.append("module." + name)
        for c in candidates:
            if c in params or c in buffers:
                return c
        return None

    for ckpt_name in ckpt_keys:
        model_name = resolve_name(ckpt_name)

        if model_name is None:
            if is_main:
                unexpected += 1
            continue

        loaded_names.add(model_name)

        if model_name in params:
            param = params[model_name]
            # All ranks must enter this context, even though only rank 0 copies.
            with zero.GatheredParameters([param], modifier_rank=0):
                if is_main:
                    tensor = sd[ckpt_name]
                    got = tuple(param.data.shape)
                    want = tuple(tensor.shape)
                    if got != want:
                        mismatched += 1
                        if len(mismatch_examples) < 12:
                            mismatch_examples.append((ckpt_name, want, got))
                    else:
                        param.data.copy_(tensor.to(device=param.data.device, dtype=param.data.dtype))
                        loaded_params += 1

        elif model_name in buffers:
            buf = buffers[model_name]
            ok = True
            if is_main:
                tensor = sd[ckpt_name]
                got = tuple(buf.data.shape)
                want = tuple(tensor.shape)
                if got != want:
                    ok = False
                    mismatched += 1
                    if len(mismatch_examples) < 12:
                        mismatch_examples.append((ckpt_name, want, got))
                else:
                    buf.data.copy_(tensor.to(device=buf.data.device, dtype=buf.data.dtype))
                    loaded_buffers += 1

            ok = _stageD_dist_broadcast_object(ok, src=0)
            if ok and _stageD_dist_is_initialized():
                torch.distributed.broadcast(buf.data, src=0)

    if is_main:
        model_keys = set(params.keys()) | set(buffers.keys())
        missing = sorted(model_keys - loaded_names)
        print(
            f"[stageD-sft-load] loaded ZeRO-3 model.safetensors: "
            f"params={loaded_params} buffers={loaded_buffers} "
            f"missing={len(missing)} unexpected={unexpected} mismatched={mismatched}",
            flush=True,
        )
        if missing[:10]:
            print(f"[stageD-sft-load][missing:first10] {missing[:10]}", flush=True)
        if mismatch_examples:
            print(f"[stageD-sft-load][mismatch:first12] {mismatch_examples}", flush=True)

    if is_main:
        sd.clear()
    del sd
    gc.collect()
    _stageD_dist_barrier()


def _stageD_load_full_or_safetensors(model, ckpt_dir, verbose=True):
    """
    Flexible loader for StageD SFT warmup.

    Supports:
      1. root model.safetensors
      2. root pytorch_model.bin
      3. native SmartCity sharded/full checkpoint via model.load_full_state_dict(...)

    Important:
      Under DeepSpeed ZeRO-3, current rank params may have shape torch.Size([0]).
      In that case a plain model.load_state_dict(full_sd) is invalid; use gathered
      parameter copy instead.
    """
    from pathlib import Path
    import gc
    import torch

    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {ckpt}")

    def log(msg):
        if verbose and _stageD_dist_rank() == 0:
            print(msg, flush=True)

    sf = ckpt / "model.safetensors"
    if sf.is_file():
        if _stageD_has_zero3_partitioned_params(model):
            _stageD_load_safetensors_zero3_aware(model, sf, verbose=verbose)
            return

        from safetensors.torch import load_file
        log(f"[stageD-sft-load] loading model.safetensors: {sf}")
        sd = load_file(str(sf), device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"[stageD-sft-load] loaded model.safetensors: tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        if list(missing)[:10]:
            log(f"[stageD-sft-load][missing:first10] {list(missing)[:10]}")
        if list(unexpected)[:10]:
            log(f"[stageD-sft-load][unexpected:first10] {list(unexpected)[:10]}")
        del sd
        gc.collect()
        return

    pt = ckpt / "pytorch_model.bin"
    if pt.is_file():
        log(f"[stageD-sft-load] loading pytorch_model.bin: {pt}")
        sd = torch.load(str(pt), map_location="cpu")
        if isinstance(sd, dict):
            for key in ("state_dict", "model", "module"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break

        if _stageD_has_zero3_partitioned_params(model):
            # Reuse a temporary safetensors-like code path is not convenient for bin;
            # fail clearly rather than plain-loading into torch.Size([0]) params.
            raise RuntimeError(
                "pytorch_model.bin loading under ZeRO-3 is not implemented in this helper. "
                "Use model.safetensors or load before ZeRO-3 partitioning."
            )

        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"[stageD-sft-load] loaded pytorch_model.bin: tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        del sd
        gc.collect()
        return

    if hasattr(model, "load_full_state_dict"):
        log(f"[stageD-sft-load] loading by model.load_full_state_dict: {ckpt}")
        try:
            model.load_full_state_dict(str(ckpt), cast_to_target_dtype=True, verbose=verbose)
        except TypeError:
            model.load_full_state_dict(str(ckpt), cast_to_target_dtype=True)
        log(f"[stageD-sft-load] loaded by model.load_full_state_dict: {ckpt}")
        return

    raise FileNotFoundError(
        f"Cannot load checkpoint from {ckpt}; expected model.safetensors, pytorch_model.bin, or load_full_state_dict support."
    )

def main():
    # Critical for SFT warmup: do not turn freeform answer back into JSON.
    os.environ.setdefault("SMARTCITY_REASON_KEEP_FREEFORM_ANSWER", "1")
    os.environ.setdefault("SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE", "1")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, StageDSFTWarmupArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, warmup_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, warmup_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    # StageD SFT warmup under ZeRO-3 must not use activation checkpointing.
    # If this remains True, backward can fail with recomputed tensor shape [0].
    if _env_flag_false("GRADIENT_CHECKPOINTING", "false"):
        training_args.gradient_checkpointing = False
        if hasattr(training_args, "gradient_checkpointing_kwargs"):
            try:
                training_args.gradient_checkpointing_kwargs = None
            except Exception:
                pass

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    set_seed(training_args.seed)

    if _is_main_process():
        print("[stageD-sft] SMARTCITY_REASON_KEEP_FREEFORM_ANSWER=", os.environ.get("SMARTCITY_REASON_KEEP_FREEFORM_ANSWER"))
        print("[stageD-sft] SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE=", os.environ.get("SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE"))
        print("[stageD-sft] full_model_load_dir=", model_args.full_model_load_dir)
        print("[stageD-sft] train_files=", data_args.train_files)
        print("[stageD-sft] validation_files=", data_args.validation_files)

    tok = _setup_tokenizer(model_args)

    enc_tok = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

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
        target_modules_override={
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [],
            "k": [],
            "v": [],
            "o": [],
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

    if not model_args.full_model_load_dir:
        raise ValueError("--full_model_load_dir is required; use output_stageC_warmup/full_2epoch for think-tag repair.")

    logger.info("Loading SFT repair base/full model weights from: %s", model_args.full_model_load_dir)
    _stageD_load_full_or_safetensors(model, model_args.full_model_load_dir, verbose=_is_main_process())

    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    _assert_vocab_safe_no_resize(model, tok)
    _freeze_embeddings(model)

    if model_args.use_mixlora and warmup_args.freeze_mixlora_non_reason:
        stats = apply_reason_only_mixlora_gradient_mask(
            model,
            num_universal=int(model_args.mixlora_num_universal),
            num_pred=int(model_args.mixlora_num_pred),
            num_reason=int(model_args.mixlora_num_reason),
            freeze_router_hidden=bool(warmup_args.freeze_router_hidden),
            verbose=_is_main_process(),
        )
        if _is_main_process():
            print("[reason-only-freeze] stats=", stats)
            print("[reason-only-freeze] restore_after_step=", warmup_args.restore_reason_only_after_step)
    else:
        if _is_main_process():
            print("[reason-only-freeze] disabled")

    if _is_main_process():
        c = _count_trainable(model)
        print(f"[stageD-sft] trainable={c['trainable']} total={c['total']} pct={c['pct_x100']:.2f}%")

        if warmup_args.print_trainable_names:
            print("[stageD-sft] trainable parameter names:")
            n_print = 0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    print("  ", name, tuple(p.shape))
                    n_print += 1
                    if n_print >= warmup_args.max_trainable_name_print:
                        print("  ...")
                        break

    # ZeRO-3 leaf modules for MixLoRA.
    try:
        configure_zero3_leaf_modules_for_mixlora(model, is_main=_is_main_process())
    except Exception as e:
        if _is_main_process():
            print("[warn] configure_zero3_leaf_modules_for_mixlora failed:", repr(e))

    if training_args.gradient_checkpointing:
        if _is_main_process():
            print("[stageD-sft] gradient_checkpointing=True requested", flush=True)
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
    else:
        _force_disable_gradient_checkpointing(model)
        if _is_main_process():
            print("[stageD-sft] gradient_checkpointing=False; checkpointing forcibly disabled", flush=True)

    assert len(data_args.train_files) > 0, "Please provide --train_files"
    train_ds = _load_and_merge(data_args.train_files, "train", training_args.seed)
    eval_ds = _load_and_merge(data_args.validation_files, "validation", training_args.seed) if data_args.validation_files else None

    if data_args.max_train_samples:
        train_ds = train_ds.select(range(min(len(train_ds), int(data_args.max_train_samples))))
    if data_args.max_eval_samples and eval_ds is not None:
        eval_ds = eval_ds.select(range(min(len(eval_ds), int(data_args.max_eval_samples))))

    if len(train_ds) == 0:
        raise ValueError("Empty training set.")

    if eval_ds is None or len(eval_ds) == 0:
        training_args.evaluation_strategy = "no"
        try:
            training_args.eval_strategy = "no"
        except Exception:
            pass
        logger.warning("Empty validation set; evaluation disabled.")

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
    if "add_eos_to_labels" in sig.parameters:
        collator_kwargs["add_eos_to_labels"] = True
    if "wrap_reason_with_tags" in sig.parameters:
        collator_kwargs["wrap_reason_with_tags"] = True

    collator = DataCollatorSmartCity(tok, **collator_kwargs)

    callbacks = [
        ReasonOnlyRestoreCallback(
            active=bool(warmup_args.restore_reason_only_after_step and warmup_args.freeze_mixlora_non_reason)
        )
    ]

    # transformers version compatibility:
    # - older Trainer accepts tokenizer=
    # - newer Trainer removed tokenizer= and uses processing_class=
    # - some custom Trainer signatures may accept neither
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if (training_args.do_eval and eval_ds is not None) else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer_sig = inspect.signature(TrafficTrainer.__init__)
    trainer_params = trainer_sig.parameters

    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tok
        if _is_main_process():
            print("[trainer-compat] using tokenizer=tok", flush=True)
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
        if _is_main_process():
            print("[trainer-compat] using processing_class=tok", flush=True)
    else:
        if _is_main_process():
            print("[trainer-compat] Trainer accepts neither tokenizer nor processing_class; skipping both", flush=True)

    trainer = TrafficTrainer(**trainer_kwargs)

    try:
        if training_args.do_train:
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            if trainer.is_world_process_zero():
                tok.save_pretrained(training_args.output_dir)
                with open(os.path.join(training_args.output_dir, "stageD_sft_warmup_manifest.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump(
                        {
                            "base": model_args.full_model_load_dir,
                            "train_files": data_args.train_files,
                            "validation_files": data_args.validation_files,
                            "output_format": "qwen3_freeform_evidence_sft_warmup_v1",
                            "train_scope": "reason experts + reason router",
                            "model_should_output_json": False,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        if training_args.do_eval and eval_ds is not None and len(eval_ds) > 0:
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


if __name__ == "__main__":
    main()
