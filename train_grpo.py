#!/usr/bin/env python
# coding: utf-8
"""
GRPO training for SmartCityLLM (StageC -> GRPO), deepspeed-launch friendly.

- Accept --local_rank injected by `deepspeed` launcher.
- Reuse your SmartCityLLM framework + MixLoRA architecture.
- Load StageC weights via `model.load_full_state_dict()` (NO tokenizer resize).
- Freezing is controlled ONLY by:
    --open_roberta / --open_vpma / --open_image_projector / --open_imagebind / --freeze_decoder
- For reason task: inject '/think' into USER message (same as your Qwen3 Route-A).
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, set_seed
from tqdm.auto import tqdm

# We use the Accelerate *library* for distributed utilities (DDP, gather, etc.)
# but you can launch with `deepspeed --num_gpus ... train_grpo.py ...` (no accelerate launcher needed).
from accelerate import Accelerator

from dataprocessing import load_jsonl_dataset, ensure_smartcity_features, DataCollatorSmartCity
from model.builder import build_smartcity_llm

from model.reinforcement_learning.config import GRPOConfig
from model.reinforcement_learning.logprobs import get_per_token_logps_smartcity
from model.reinforcement_learning.loss import grpo_loss
from model.reinforcement_learning.rewards import ExampleMeta
from model.reinforcement_learning.rollout import rollout_and_cache


THINK_SOFT = "/think"


# ---------------------------
# small utils
# ---------------------------
class HFWrap(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "on"):
        return True
    if s in ("0", "false", "no", "n", "f", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def _token_exists(tok, token: str) -> bool:
    vocab = tok.get_vocab()
    if token not in vocab:
        return False
    tid = vocab.get(token, None)
    if tid is None:
        return False
    unk = getattr(tok, "unk_token_id", None)
    return (unk is None) or (tid != unk)


def _effective_vocab_size(tok) -> int:
    vocab = tok.get_vocab()
    max_id = max(vocab.values()) if vocab else -1
    for t in tok.all_special_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            max_id = max(max_id, tid)
    return int(max_id + 1)


def _assert_vocab_safe_no_resize(model, tok) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    dec = getattr(model, "decoder", model)
    emb = None
    try:
        emb = dec.get_input_embeddings()
    except Exception:
        emb = None
    if emb is None or not hasattr(emb, "weight"):
        return

    tgt = int(_effective_vocab_size(tok))
    cur = int(emb.weight.size(0)) if emb.weight is not None else -1
    if cur <= 0:
        cfg = getattr(dec, "config", None)
        cfg_vocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
        if cfg_vocab is not None:
            cur = int(cfg_vocab)

    if cur <= 0:
        print("[warn] cannot infer model vocab rows; skip vocab safety check.")
        return

    if tgt > cur:
        raise ValueError(
            f"[FATAL] tokenizer_effective_vocab({tgt}) > model_embedding_rows({cur}). "
            "NO resize is allowed; fix tokenizer/model pairing."
        )

    if cur != tgt:
        print(f"[info] padded vocab: model_embedding_rows={cur}, tokenizer_effective_vocab={tgt}, len(tok)={len(tok)}")


def find_latest_ckpt_dir(root_dir: str) -> str:
    p = Path(root_dir)
    if not p.exists():
        raise FileNotFoundError(root_dir)

    direct = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for fn in direct:
        if (p / fn).exists():
            return str(p)

    ckpts: List[Tuple[int, Path]] = []
    for d in p.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", d.name)
        step = int(m.group(1)) if m else -1
        ckpts.append((step, d))
    if ckpts:
        ckpts.sort(key=lambda x: x[0])
        return str(ckpts[-1][1])

    return str(p)


def load_tokenizer_no_modify(ckpt_dir: str, fallback_name: str, padding_token: str, local_files_only: bool):
    tok_src = ckpt_dir if Path(ckpt_dir).exists() else fallback_name
    tok = AutoTokenizer.from_pretrained(
        tok_src,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=bool(local_files_only),
    )

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    if tok.pad_token_id is None:
        pad_str = (padding_token or "").strip()
        if pad_str and _token_exists(tok, pad_str):
            tok.pad_token = pad_str
        else:
            tok.pad_token = tok.eos_token

    print(f"[tok] src={tok_src} | pad={tok.pad_token}({tok.pad_token_id}) eos={tok.eos_token}({tok.eos_token_id}) len={len(tok)}")
    return tok


def build_stop_token_ids(tok) -> Optional[List[int]]:
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    unk = getattr(tok, "unk_token_id", None)

    extra: List[int] = []
    for s in ["<|im_end|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(s)
        if not isinstance(tid, int) or tid < 0:
            continue
        if unk is not None and tid == unk:
            continue
        if tid == eos:
            continue
        if pad is not None and tid == pad:
            continue
        extra.append(int(tid))

    extra = list(dict.fromkeys(extra))
    return extra if extra else None


def inject_soft_think_in_user(prompt: str) -> str:
    p = prompt or ""
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = p.rfind(marker)
    if idx < 0:
        return p
    before = p[:idx]
    if THINK_SOFT in before[-400:]:
        return p
    return before.rstrip() + f"\n{THINK_SOFT}\n" + p[idx:]


def _call_load_jsonl_dataset(path: str, split_name: str):
    sig = inspect.signature(load_jsonl_dataset)
    if "split_name" in sig.parameters:
        return load_jsonl_dataset(path, split_name=split_name)
    return load_jsonl_dataset(path)


def _call_ensure_features(ds, num_proc: int):
    sig = inspect.signature(ensure_smartcity_features)
    kwargs: Dict[str, Any] = {"num_proc": num_proc}
    if "ensure_markers" in sig.parameters:
        kwargs["ensure_markers"] = False
    return ensure_smartcity_features(ds, **kwargs)


def trainable_named_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # Save only trainable params (avoids dumping full 8B state_dict)
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint_trainable(output_dir: str, step: int, model, tokenizer, accelerator: Accelerator, keep_last_n: int):
    if not accelerator.is_main_process:
        return

    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    m = accelerator.unwrap_model(model)
    torch.save(trainable_named_parameters(m), ckpt_dir / "adapter_model.bin")
    tokenizer.save_pretrained(str(ckpt_dir))

    with open(ckpt_dir / "grpo_step.txt", "w", encoding="utf-8") as f:
        f.write(str(step))

    # cleanup old checkpoints
    if keep_last_n and keep_last_n > 0:
        ckpts = []
        for d in Path(output_dir).glob("checkpoint-*"):
            mm = re.search(r"checkpoint-(\d+)$", d.name)
            if mm:
                ckpts.append((int(mm.group(1)), d))
        ckpts.sort(key=lambda x: x[0])
        if len(ckpts) > keep_last_n:
            for _, d in ckpts[:-keep_last_n]:
                try:
                    for child in d.rglob("*"):
                        if child.is_file():
                            child.unlink()
                    for child in sorted([x for x in d.rglob("*") if x.is_dir()], reverse=True):
                        try:
                            child.rmdir()
                        except Exception:
                            pass
                    d.rmdir()
                    print(f"[ckpt] removed old: {d}")
                except Exception:
                    pass

    print(f"[save] trainable checkpoint-{step} -> {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()

    # ✅ deepspeed launcher injects this
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))

    # data / ckpt
    ap.add_argument("--sft_dir", type=str, required=True)
    ap.add_argument("--train_files", type=str, nargs="+", required=True)
    ap.add_argument("--eval_files", type=str, nargs="+", default=None)
    ap.add_argument("--output_dir", type=str, default="outputs_grpo")
    ap.add_argument("--only_task", type=str, default="reason")

    # model/tokenizer
    ap.add_argument("--decoder_name_or_path", type=str, required=True)
    ap.add_argument("--encoder_name_or_path", type=str, required=True)
    ap.add_argument("--padding_token", type=str, default="<|pad|>")
    ap.add_argument("--local_files_only", type=str2bool, default=True)

    # structure (match StageC)
    ap.add_argument("--num_chunk_tokens", type=int, default=4)
    ap.add_argument("--adapter_heads", type=int, default=4)
    ap.add_argument("--encoder_max_length", type=int, default=512)

    ap.add_argument("--num_image_tokens", type=int, default=32)
    ap.add_argument("--imagebind_variant", type=str, default="google/siglip-so400m-patch14-384")
    ap.add_argument("--group_layers", type=int, default=2)
    ap.add_argument("--group_heads", type=int, default=8)
    ap.add_argument("--group_tau", type=float, default=1.0)

    # freeze controls
    ap.add_argument("--open_roberta", type=str2bool, default=False)
    ap.add_argument("--open_vpma", type=str2bool, default=False)
    ap.add_argument("--open_image_projector", type=str2bool, default=False)
    ap.add_argument("--open_imagebind", type=str2bool, default=False)
    ap.add_argument("--freeze_decoder", type=str2bool, default=True)

    # task weights
    ap.add_argument("--alpha_understand", type=float, default=0.2)
    ap.add_argument("--beta_prediction", type=float, default=1.0)
    ap.add_argument("--beta_reason", type=float, default=1.0)
    ap.add_argument("--lambda_rat", type=float, default=0.3)

    # MixLoRA
    ap.add_argument("--use_mixlora", type=str2bool, default=True)
    ap.add_argument("--mixlora_num_experts", type=int, default=20)
    ap.add_argument("--mixlora_r", type=int, default=8)
    ap.add_argument("--mixlora_alpha", type=float, default=16.0)
    ap.add_argument("--mixlora_share_router_for_wi", type=str2bool, default=True)
    ap.add_argument("--mixlora_enable_attention", type=str2bool, default=False)
    ap.add_argument("--mixlora_ensure_nonzero_gating", type=str2bool, default=False)

    ap.add_argument("--mixlora_num_universal", type=int, default=4)
    ap.add_argument("--mixlora_num_pred", type=int, default=8)
    ap.add_argument("--mixlora_num_reason", type=int, default=8)
    ap.add_argument("--mixlora_rank_universal_mul", type=float, default=2.0)

    # collator limits
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--max_images_per_sample", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--max_chunks_per_sample", type=int, default=None)

    # rollout / RL
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--clip_eps", type=float, default=0.1)
    ap.add_argument("--beta_kl", type=float, default=0.0)

    # training
    ap.add_argument("--batch_size", type=int, default=1)          # per-process
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=4e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--micro_batch_logps", type=int, default=2)   # micro batch for logps+backward
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--num_eval_batches", type=int, default=5)

    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    # wandb
    ap.add_argument("--use_wandb", type=str2bool, default=False)
    ap.add_argument("--wandb_project", type=str, default="smartcity_grpo")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="online")
    ap.add_argument("--log_text_steps", type=int, default=0)

    args = ap.parse_args()

    if args.local_rank >= 0:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    set_seed(int(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    # ------------------ wandb init ------------------
    wandb_run = None
    if args.use_wandb and is_main and args.wandb_mode != "disabled":
        import wandb
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        init_kwargs = dict(
            project=args.wandb_project,
            name=(args.wandb_run_name or None),
            config=vars(args),
            dir=args.output_dir,
            reinit=True,
        )
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        wandb_run = wandb.init(**init_kwargs)

    def wlog(d: Dict[str, Any], step: int):
        if wandb_run is not None and is_main:
            import wandb
            wandb.log(d, step=step)

    # ------------------ ckpt + tokenizer ------------------
    ckpt_dir = find_latest_ckpt_dir(args.sft_dir)
    if is_main:
        print(f"[info] using StageC ckpt_dir: {ckpt_dir}")

    tok = load_tokenizer_no_modify(
        ckpt_dir=ckpt_dir,
        fallback_name=args.decoder_name_or_path,
        padding_token=args.padding_token,
        local_files_only=bool(args.local_files_only),
    )

    enc_tok = AutoTokenizer.from_pretrained(
        args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    stop_ids = build_stop_token_ids(tok)
    if is_main:
        print(f"[stop] eos={tok.eos_token_id}, pad={tok.pad_token_id}, extra_stop_ids={stop_ids}")

    # ------------------ build model (SmartCityLLM) ------------------
    mix_cfg_overrides: Dict[str, Any] = dict(
        num_experts=int(args.mixlora_num_experts),
        lora_r=int(args.mixlora_r),
        lora_alpha=float(args.mixlora_alpha),
        enable_attention=bool(args.mixlora_enable_attention),
        enable_ffn=True,
        share_router_for_wi=bool(args.mixlora_share_router_for_wi),
        num_tasks=3,
        ensure_nonzero_gating=bool(args.mixlora_ensure_nonzero_gating),
        num_universal=int(args.mixlora_num_universal),
        num_pred=int(args.mixlora_num_pred),
        num_reason=int(args.mixlora_num_reason),
        rank_universal_mul=float(args.mixlora_rank_universal_mul),
        target_modules_override={
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [], "k": [], "v": [], "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        },
    )

    model = build_smartcity_llm(
        decoder_name=args.decoder_name_or_path,
        encoder_name=args.encoder_name_or_path,
        alpha_understand=args.alpha_understand,
        beta_prediction=args.beta_prediction,
        beta_reason=args.beta_reason,
        lambda_rat=args.lambda_rat,
        num_chunk_tokens=int(args.num_chunk_tokens),
        adapter_heads=int(args.adapter_heads),
        encoder_max_length=int(args.encoder_max_length),
        imagebind_variant=args.imagebind_variant,
        num_image_tokens=int(args.num_image_tokens),
        group_layers=int(args.group_layers),
        group_heads=int(args.group_heads),
        group_tau=float(args.group_tau),
        open_roberta=bool(args.open_roberta),
        open_vpma=bool(args.open_vpma),
        open_imagebind=bool(args.open_imagebind),
        open_image_projector=bool(args.open_image_projector),
        freeze_decoder=bool(args.freeze_decoder),
        use_mixlora=bool(args.use_mixlora),
        mixlora_cfg_overrides=mix_cfg_overrides,
    )

    # sync ids (no vocab change)
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
            model.decoder.config.use_cache = False
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
            model.config.use_cache = False
    except Exception:
        pass

    # load StageC full weights using your method
    if not hasattr(model, "load_full_state_dict"):
        raise AttributeError("SmartCityLLM model has no load_full_state_dict(). Make sure build_smartcity_llm returns SmartCityLLMForCausalLM.")
    if is_main:
        print(f"[load] model.load_full_state_dict({ckpt_dir})")
    model.load_full_state_dict(str(ckpt_dir), cast_to_target_dtype=True, verbose=is_main)

    _assert_vocab_safe_no_resize(model, tok)

    # gradient checkpointing
    if bool(args.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        try:
            if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
                model.decoder.config.use_cache = False
        except Exception:
            pass

    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # optional ref model (KL)
    ref_model = None
    if float(args.beta_kl) != 0.0:
        if is_main:
            print("[warn] beta_kl != 0 -> building ref_model (extra memory)")
        ref_model = build_smartcity_llm(
            decoder_name=args.decoder_name_or_path,
            encoder_name=args.encoder_name_or_path,
            alpha_understand=args.alpha_understand,
            beta_prediction=args.beta_prediction,
            beta_reason=args.beta_reason,
            lambda_rat=args.lambda_rat,
            num_chunk_tokens=int(args.num_chunk_tokens),
            adapter_heads=int(args.adapter_heads),
            encoder_max_length=int(args.encoder_max_length),
            imagebind_variant=args.imagebind_variant,
            num_image_tokens=int(args.num_image_tokens),
            group_layers=int(args.group_layers),
            group_heads=int(args.group_heads),
            group_tau=float(args.group_tau),
            open_roberta=False,
            open_vpma=False,
            open_imagebind=False,
            open_image_projector=False,
            freeze_decoder=True,
            use_mixlora=bool(args.use_mixlora),
            mixlora_cfg_overrides=mix_cfg_overrides,
        )
        ref_model.load_full_state_dict(str(ckpt_dir), cast_to_target_dtype=True, verbose=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        try:
            ref_model.to(accelerator.device)
        except Exception:
            pass

    # ------------------ dataset ------------------
    dss = [_call_load_jsonl_dataset(p, split_name="train") for p in args.train_files]
    ds = dss[0] if len(dss) == 1 else __import__("datasets").concatenate_datasets(dss)

    num_proc = int(os.environ.get("SMARTCITY_MAP_NUM_PROC", "16"))
    ds = _call_ensure_features(ds, num_proc=num_proc)

    if args.only_task:
        want = args.only_task.strip().lower()
        ds = ds.filter(lambda ex: (ex.get("task", "") or "").lower() == want)

    collator_kwargs: Dict[str, Any] = dict(
        max_length=int(args.max_length),
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=int(args.num_chunk_tokens),
        image_token_factor=int(args.num_image_tokens),
        max_images_per_sample=int(args.max_images_per_sample),
        image_size=int(args.image_size),
        max_chunks_per_sample=args.max_chunks_per_sample,
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
        if "add_eos_to_labels" in sig.parameters:
            collator_kwargs["add_eos_to_labels"] = True
        if "wrap_reason_with_tags" in sig.parameters:
            collator_kwargs["wrap_reason_with_tags"] = True
    except Exception:
        pass

    base_collator = DataCollatorSmartCity(tok, **collator_kwargs)

    def rl_collator(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        metas = [
            ExampleMeta(
                task=str(ex.get("task", "")),
                target=str(ex.get("target", "")),
                answer=str(ex.get("answer", "")),
            )
            for ex in examples
        ]
        patched = []
        for ex in examples:
            ex2 = dict(ex)
            ex2["answer"] = ""
            ex2["target"] = ""
            ex2["thinking"] = []
            if (ex2.get("task", "") or "").lower() == "reason":
                ex2["prompt"] = inject_soft_think_in_user(ex2.get("prompt", "") or "")
            patched.append(ex2)
        out = base_collator(patched)
        out["meta"] = metas
        return out

    dl = DataLoader(HFWrap(ds), batch_size=int(args.batch_size), shuffle=True, collate_fn=rl_collator)

    eval_dl = None
    if args.eval_files:
        eval_dss = [_call_load_jsonl_dataset(p, split_name="validation") for p in args.eval_files]
        eval_ds = eval_dss[0] if len(eval_dss) == 1 else __import__("datasets").concatenate_datasets(eval_dss)
        eval_ds = _call_ensure_features(eval_ds, num_proc=num_proc)
        if args.only_task:
            want = args.only_task.strip().lower()
            eval_ds = eval_ds.filter(lambda ex: (ex.get("task", "") or "").lower() == want)
        eval_dl = DataLoader(HFWrap(eval_ds), batch_size=int(args.batch_size), shuffle=False, collate_fn=rl_collator)

    # ------------------ GRPO config ------------------
    cfg = GRPOConfig(
        num_generations=int(args.num_generations),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        beta_kl=float(args.beta_kl),
        stop_token_ids=build_stop_token_ids(tok),
        clip_epsilon_low=float(args.clip_eps),
        clip_epsilon_high=float(args.clip_eps),
        num_policy_updates=1,
        loss_on_answer_only=True,
    )
    cfg.validate()

    if is_main:
        with open(Path(args.output_dir) / "grpo_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # ------------------ optimizer ------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    # prepare (DDP)
    model, optim, dl = accelerator.prepare(model, optim, dl)
    if eval_dl is not None:
        eval_dl = accelerator.prepare(eval_dl)

    # refresh trainable params after wrapping (safer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ------------------ steps ------------------
    num_update_steps_per_epoch = math.ceil(len(dl) / max(1, int(args.grad_accum)))
    max_train_steps = num_update_steps_per_epoch * int(args.num_epochs)
    if is_main:
        print(f"[info] num_update_steps_per_epoch={num_update_steps_per_epoch}, max_train_steps={max_train_steps}")

    progress_bar = tqdm(total=max_train_steps, desc="GRPO train", disable=not is_main)

    global_step = 0
    optim.zero_grad(set_to_none=True)

    need_grad_modalities = bool(args.open_roberta or args.open_vpma or args.open_image_projector or args.open_imagebind)

    def run_eval_once(step: int) -> Optional[float]:
        if eval_dl is None:
            return None
        model.eval()
        total = 0.0
        count = 0
        for i, batch in enumerate(eval_dl):
            roll = rollout_and_cache(
                accelerator=accelerator,
                model=model,
                tokenizer=tok,
                batch=batch,
                cfg=cfg,
                ref_model=ref_model,
                micro_batch_logps=(int(args.micro_batch_logps) if int(args.micro_batch_logps) > 0 else None),
            )
            rr = accelerator.gather_for_metrics(roll.rewards)
            total += float(rr.sum().item())
            count += int(rr.numel())
            if int(args.num_eval_batches) > 0 and (i + 1) >= int(args.num_eval_batches):
                break
        model.train()
        return (total / max(1, count)) if count > 0 else None

    # ------------------ train loop ------------------
    for epoch in range(int(args.num_epochs)):
        dl_len = len(dl)
        for batch_idx, batch in enumerate(dl, start=1):
            roll = rollout_and_cache(
                accelerator=accelerator,
                model=model,
                tokenizer=tok,
                batch=batch,
                cfg=cfg,
                ref_model=ref_model,
                micro_batch_logps=(int(args.micro_batch_logps) if int(args.micro_batch_logps) > 0 else None),
            )

            Btot = int(roll.completion_ids.size(0))
            mb = int(args.micro_batch_logps) if int(args.micro_batch_logps) > 0 else Btot

            window_start = batch_idx - ((batch_idx - 1) % int(args.grad_accum))
            effective_accum = min(int(args.grad_accum), dl_len - window_start + 1)
            do_step = (batch_idx % int(args.grad_accum) == 0) or (batch_idx == dl_len)

            loss_weighted_sum = 0.0

            prefix_ids_base = batch["prefix_ids"]
            B = int(prefix_ids_base.size(0))
            G = max(1, Btot // max(1, B))

            chunk_txts_base = batch.get("chunk_txts", None)
            pixel_values_base = batch.get("pixel_values", None)
            n_images_base = batch.get("n_images", None)

            chunk_txts_exp = None
            pixel_values_exp = None
            n_images_exp = None
            if need_grad_modalities:
                if isinstance(chunk_txts_base, list) and len(chunk_txts_base) == B:
                    chunk_txts_exp = []
                    for i in range(B):
                        for _ in range(G):
                            chunk_txts_exp.append(chunk_txts_base[i])
                if torch.is_tensor(pixel_values_base) and pixel_values_base.size(0) == B:
                    pixel_values_exp = pixel_values_base.repeat_interleave(G, dim=0)
                if torch.is_tensor(n_images_base) and n_images_base.size(0) == B:
                    n_images_exp = n_images_base.repeat_interleave(G, dim=0)

            task_ids_full = roll.rep.get("task_ids", None)

            for st in range(0, Btot, mb):
                ed = min(Btot, st + mb)
                sl = slice(st, ed)
                bs_mb = int(ed - st)

                if need_grad_modalities:
                    chunk_txts_mb = chunk_txts_exp[st:ed] if chunk_txts_exp is not None else None
                    pv_mb = pixel_values_exp[sl] if pixel_values_exp is not None else None
                    ni_mb = n_images_exp[sl] if n_images_exp is not None else None

                    chunk_embeds, chunk_mask, image_embeds, image_mask, chunk_emb_all, img_proj_hook = model.encode_modalities(
                        chunk_txts_all=chunk_txts_mb,
                        pixel_values=pv_mb,
                        n_images=ni_mb,
                        batch_size=bs_mb,
                        device=None,
                        dtype=None,
                    )
                else:
                    chunk_embeds = roll.chunk_embeds[sl]
                    chunk_mask = roll.chunk_mask[sl]
                    image_embeds = roll.image_embeds[sl]
                    image_mask = roll.image_mask[sl]
                    chunk_emb_all = None
                    img_proj_hook = None

                new_logps_mb = get_per_token_logps_smartcity(
                    model,
                    prefix_ids=roll.rep["prefix_ids"][sl],
                    prefix_mask=roll.rep["prefix_mask"][sl],
                    suffix_ids=roll.rep["suffix_ids"][sl],
                    suffix_mask=roll.rep["suffix_mask"][sl],
                    chunk_embeds=chunk_embeds,
                    chunk_mask=chunk_mask,
                    image_embeds=image_embeds,
                    image_mask=image_mask,
                    completion_ids=roll.completion_ids[sl],
                    completion_mask=roll.stop_mask[sl],
                    task_ids=(task_ids_full[sl] if task_ids_full is not None else None),
                    temperature=cfg.temperature,
                    micro_batch_size=None,
                )

                loss_mb = grpo_loss(
                    per_token_logps=new_logps_mb,
                    completion_mask=roll.loss_mask[sl],
                    advantages=roll.advantages[sl],
                    old_per_token_logps=(roll.old_per_token_logps[sl] if roll.old_per_token_logps is not None else None),
                    ref_per_token_logps=(roll.ref_per_token_logps[sl] if roll.ref_per_token_logps is not None else None),
                    beta_kl=cfg.beta_kl,
                    epsilon_low=cfg.clip_epsilon_low,
                    epsilon_high=cfg.clip_epsilon_high,
                )

                if need_grad_modalities:
                    if chunk_emb_all is not None:
                        loss_mb = loss_mb + (chunk_emb_all.sum().to(loss_mb.dtype) * 0.0)
                    if img_proj_hook is not None:
                        loss_mb = loss_mb + img_proj_hook.to(loss_mb.dtype)

                scale = (bs_mb / float(Btot)) / max(1, effective_accum)
                accelerator.backward(loss_mb * scale)

                loss_weighted_sum += float(loss_mb.detach().item()) * bs_mb
                del new_logps_mb, loss_mb

            if do_step:
                grad_norm = accelerator.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if is_main:
                    progress_bar.update(1)

                with torch.no_grad():
                    r_all = accelerator.gather_for_metrics(roll.rewards)
                    r_mean = float(r_all.mean().item())
                    r_std = float(r_all.std(unbiased=False).item())
                    r_min = float(r_all.min().item())
                    r_max = float(r_all.max().item())

                    loss_value = loss_weighted_sum / max(1.0, float(Btot))

                    len_stop = accelerator.gather_for_metrics(roll.stop_mask.sum(dim=1).float())
                    len_loss = accelerator.gather_for_metrics(roll.loss_mask.sum(dim=1).float())
                    len_stop_mean = float(len_stop.mean().item())
                    len_loss_mean = float(len_loss.mean().item())

                    lr = float(optim.param_groups[0]["lr"])
                    gnorm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                if is_main and (global_step % int(args.log_steps) == 0):
                    progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "r": f"{r_mean:.4f}"})

                log_dict = {
                    "train/loss": float(loss_value),
                    "train/reward_mean": r_mean,
                    "train/reward_std": r_std,
                    "train/reward_min": r_min,
                    "train/reward_max": r_max,
                    "train/len_stop_mean": len_stop_mean,
                    "train/len_loss_mean": len_loss_mean,
                    "train/grad_norm": gnorm,
                    "train/lr": lr,
                }
                if roll.reward_info:
                    for k, v in roll.reward_info.items():
                        try:
                            log_dict[f"reward/{k}"] = float(v)
                        except Exception:
                            pass
                wlog(log_dict, step=global_step)

                if args.log_text_steps and (global_step % int(args.log_text_steps) == 0) and roll.texts and is_main:
                    import wandb
                    wlog({"sample/text": wandb.Html(f"<pre>{roll.texts[0]}</pre>")}, step=global_step)

                if int(args.save_steps) > 0 and (global_step % int(args.save_steps) == 0):
                    save_checkpoint_trainable(args.output_dir, global_step, model, tok, accelerator, int(args.save_total_limit))

                if int(args.eval_steps) > 0 and eval_dl is not None and (global_step % int(args.eval_steps) == 0):
                    er = run_eval_once(global_step)
                    if is_main and er is not None:
                        print(f"[eval] step={global_step} mean_reward={er:.4f}")

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    if is_main:
        progress_bar.close()

    save_checkpoint_trainable(args.output_dir, global_step, model, tok, accelerator, int(args.save_total_limit))

    if is_main:
        print(f"[done] global_step={global_step}")

    if wandb_run is not None and is_main:
        wandb_run.finish()


if __name__ == "__main__":
    main()
