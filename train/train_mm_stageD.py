# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

"""
GRPO training for SmartCityLLM (Qwen3-aligned)

Key rules for Qwen3 (match your SFT/eval Route-A success):
- DO NOT add special tokens; DO NOT resize embeddings.
- Inject '/think' into USER message for reason tasks.
- Rewards score answer AFTER the last </think> (do NOT require <answer> tags).
- Stop tokens: rely on eos + optional <|im_end|> (exclude pad token id from stop ids).
- RL loss masking: keep conditioning full (stop_mask), but optimize answer-only (loss_mask after </think>).

W&B logging:
- train/loss
- train/reward_mean/std/min/max
- train/format_mean, train/rougeL_mean, train/cosine_mean (reward components)
- train/clip_frac, train/ratio_mean
- train/kl_mean (if beta_kl > 0)
- train/len_stop_mean, train/len_loss_mean
- train/think_close_rate
- train/grad_norm, train/lr, train/gpu_mem_gb
"""

import os
import json
import math
import re
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from tqdm.auto import tqdm
from accelerate import Accelerator

from dataprocessing_lora import load_jsonl_dataset, DataCollatorTraffic
from model.language_model.smartcity_llm_lora import build_traffic_llm

from model.reinforcement_learning.config import GRPOConfig
from model.reinforcement_learning.logprobs import get_per_token_logps_smartcity
from model.reinforcement_learning.loss import grpo_loss
from model.reinforcement_learning.rewards import ExampleMeta
from model.reinforcement_learning.rollout import rollout_and_cache


THINK_SOFT = "/think"
THINK_CLOSE = "</think>"


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


def resolve_dtype(s: str) -> Optional[torch.dtype]:
    if s == "auto":
        return None
    return getattr(torch, s, None)


def build_bnb_config(load_in_bits: int) -> Optional[BitsAndBytesConfig]:
    if load_in_bits in (4, 8):
        return BitsAndBytesConfig(
            load_in_4bit=(load_in_bits == 4),
            load_in_8bit=(load_in_bits == 8),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


# ---------------------------
# checkpoint utils
# ---------------------------
def find_latest_ckpt_dir(root_dir: str) -> str:
    p = Path(root_dir)
    if not p.exists():
        raise FileNotFoundError(root_dir)

    for fn in ["model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin"]:
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


def load_state_dict_any(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    ckpt = Path(ckpt_dir)

    sf2 = ckpt / "adapter_model.safetensors"
    if sf2.exists():
        from safetensors.torch import load_file
        return load_file(str(sf2))

    pb2 = ckpt / "adapter_model.bin"
    if pb2.exists():
        return torch.load(str(pb2), map_location="cpu")

    sf = ckpt / "model.safetensors"
    if sf.exists():
        from safetensors.torch import load_file
        return load_file(str(sf))

    pb = ckpt / "pytorch_model.bin"
    if pb.exists():
        return torch.load(str(pb), map_location="cpu")

    raise FileNotFoundError(f"Cannot find weights in: {ckpt_dir}")


def trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    name_set = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if k in name_set}


def save_checkpoint(output_dir: str, step: int, model, tokenizer, accelerator: Accelerator):
    if not accelerator.is_main_process:
        return
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    m = accelerator.unwrap_model(model)
    torch.save(trainable_state_dict(m), ckpt_dir / "adapter_model.bin")
    tokenizer.save_pretrained(str(ckpt_dir))
    with open(ckpt_dir / "grpo_step.txt", "w", encoding="utf-8") as f:
        f.write(str(step))
    print(f"[save] checkpoint-{step} -> {ckpt_dir}")


# ---------------------------
# tokenizer + prompt patch
# ---------------------------
def load_tokenizer_no_modify(ckpt_dir: str, fallback_name: str):
    """
    Qwen3 rule: DO NOT add special tokens here.
    Keep token-id alignment with SFT.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )
        print(f"[tok] loaded from ckpt_dir: {ckpt_dir}")
    except Exception as e:
        print(f"[tok][warn] failed to load from ckpt_dir ({ckpt_dir}): {e}")
        tok = AutoTokenizer.from_pretrained(
            fallback_name,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )
        print(f"[tok] loaded from base: {fallback_name}")

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    # Set pad to an EXISTING token (no new tokens / no resize)
    if tok.pad_token_id is None:
        if "<|pad|>" in tok.get_vocab():
            tid = tok.convert_tokens_to_ids("<|pad|>")
            if isinstance(tid, int) and tid >= 0 and tid != tok.eos_token_id:
                tok.pad_token = "<|pad|>"
            else:
                tok.pad_token = tok.eos_token
        else:
            tok.pad_token = tok.eos_token

        print(f"[tok] pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id}")

    if tok.pad_token_id == tok.eos_token_id:
        print(
            f"[tok][warn] pad_token_id == eos_token_id == {tok.eos_token_id}. "
            f"This is allowed in no-vocab-expansion mode, but ensure your collator uses attention_mask correctly."
        )

    for t in ["<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|pad|>"]:
        tid = tok.convert_tokens_to_ids(t)
        print(f"[tok] {t} -> {tid}")

    return tok


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


def build_stop_token_ids(tok) -> Optional[List[int]]:
    eos = tok.eos_token_id
    pad = tok.pad_token_id

    extra: List[int] = []
    for s in ["<|im_end|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            if tid == eos:
                continue
            if pad is not None and tid == pad:
                continue
            extra.append(int(tid))

    extra = list(dict.fromkeys(extra))
    return extra if extra else None


# ---------------------------
# training
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sft_dir", type=str, required=True)
    ap.add_argument("--train_files", type=str, nargs="+", required=True)
    ap.add_argument("--eval_files", type=str, nargs="+", default=None)
    ap.add_argument("--output_dir", type=str, default="outputs_grpo")
    ap.add_argument("--only_task", type=str, default="reason")

    ap.add_argument("--decoder_name_or_path", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--encoder_name_or_path", type=str, default="FacebookAI/roberta-large")
    ap.add_argument("--num_chunk_tokens", type=int, default=2)

    ap.add_argument("--enc_lora_r", type=int, default=4)
    ap.add_argument("--enc_lora_alpha", type=int, default=16)
    ap.add_argument("--enc_lora_dropout", type=float, default=0.05)
    ap.add_argument("--enc_target_modules", type=str, default="query,key,value,dense")

    ap.add_argument("--dec_lora_r", type=int, default=8)
    ap.add_argument("--dec_lora_alpha", type=int, default=32)
    ap.add_argument("--dec_lora_dropout", type=float, default=0.05)
    ap.add_argument("--dec_target_modules", type=str, default="q_proj,k_proj,v_proj")

    ap.add_argument("--alpha_understand", type=float, default=0.2)
    ap.add_argument("--beta_prediction", type=float, default=1.0)
    ap.add_argument("--beta_reason", type=float, default=1.0)
    ap.add_argument("--lambda_rat", type=float, default=0.25)

    ap.add_argument("--load_in_bits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--beta_kl", type=float, default=0.0)
    ap.add_argument("--clip_eps", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--micro_batch_logps", type=int, default=0)

    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--num_eval_batches", type=int, default=5)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)

    ap.add_argument("--use_wandb", type=str2bool, default=False)
    ap.add_argument("--wandb_project", type=str, default="smartcity_grpo")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default="online")
    ap.add_argument("--log_text_steps", type=int, default=0)
    ap.add_argument("--adv_std_mode", type=str, default="group", choices=["group", "batch", "none"])
    ap.add_argument("--use_old_logps", type=str2bool, default=False)
    ap.add_argument("--open_roberta", type=str2bool, default=False)
    ap.add_argument("--open_vpma", type=str2bool, default=False)
    ap.add_argument("--freeze_decoder", type=str2bool, default=True)
    ap.add_argument("--disable_tqdm", type=str2bool, default=False)
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Transformer attention backend. Use 'auto' to not force anything.",
    )
    ap.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator()

    wandb_run = None
    if args.use_wandb and accelerator.is_main_process and args.wandb_mode != "disabled":
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

    def wlog(d: Dict, step: int):
        if wandb_run is not None and accelerator.is_main_process:
            import wandb
            wandb.log(d, step=step)

    ckpt_dir = find_latest_ckpt_dir(args.sft_dir)
    if accelerator.is_main_process:
        print(f"[info] using ckpt_dir: {ckpt_dir}")

    tok = load_tokenizer_no_modify(ckpt_dir, args.decoder_name_or_path)

    enc_tok = AutoTokenizer.from_pretrained(
        args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    stop_ids = build_stop_token_ids(tok)
    if accelerator.is_main_process:
        print(f"[stop] eos={tok.eos_token_id}, pad={tok.pad_token_id}, extra_stop_ids={stop_ids}")

    bnb_config = build_bnb_config(args.load_in_bits)
    attn_impl = None if (args.attn_implementation in ("auto", "", "none")) else args.attn_implementation

    model = build_traffic_llm(
        encoder_name=args.encoder_name_or_path,
        decoder_name=args.decoder_name_or_path,
        num_chunk_tokens=int(args.num_chunk_tokens),
        encoder_lora_cfg=dict(
            r=args.enc_lora_r,
            alpha=args.enc_lora_alpha,
            lora_dropout=args.enc_lora_dropout,
            target_modules=[x.strip() for x in args.enc_target_modules.split(",") if x.strip()],
            bias="none",
        ),
        decoder_lora_cfg=dict(
            r=args.dec_lora_r,
            lora_alpha=args.dec_lora_alpha,
            lora_dropout=args.dec_lora_dropout,
            target_modules=[x.strip() for x in args.dec_target_modules.split(",") if x.strip()],
            bias="none",
        ),
        quantization_config=bnb_config,
        torch_dtype=resolve_dtype(args.torch_dtype),
        alpha_understand=args.alpha_understand,
        beta_prediction=args.beta_prediction,
        beta_reason=args.beta_reason,
        lambda_rat=args.lambda_rat,
        open_roberta=bool(args.open_roberta),
        open_vpma=bool(args.open_vpma),
        freeze_decoder=bool(args.freeze_decoder),
        attn_implementation=attn_impl,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    sd = load_state_dict_any(ckpt_dir)
    model.load_state_dict(sd, strict=False)

    try:
        model.to(accelerator.device)
    except Exception:
        pass

    ref_model = None
    if float(args.beta_kl) != 0.0:
        ref_model = build_traffic_llm(
            encoder_name=args.encoder_name_or_path,
            decoder_name=args.decoder_name_or_path,
            num_chunk_tokens=int(args.num_chunk_tokens),
            encoder_lora_cfg=dict(
                r=args.enc_lora_r,
                alpha=args.enc_lora_alpha,
                lora_dropout=args.enc_lora_dropout,
                target_modules=[x.strip() for x in args.enc_target_modules.split(",") if x.strip()],
                bias="none",
            ),
            decoder_lora_cfg=dict(
                r=args.dec_lora_r,
                lora_alpha=args.dec_lora_alpha,
                lora_dropout=args.dec_lora_dropout,
                target_modules=[x.strip() for x in args.dec_target_modules.split(",") if x.strip()],
                bias="none",
            ),
            quantization_config=bnb_config,
            torch_dtype=resolve_dtype(args.torch_dtype),
            alpha_understand=args.alpha_understand,
            beta_prediction=args.beta_prediction,
            beta_reason=args.beta_reason,
            lambda_rat=args.lambda_rat,
            attn_implementation=attn_impl,
            gradient_checkpointing=False,
        )
        ref_model.load_state_dict(sd, strict=False)
        try:
            ref_model.to(accelerator.device)
        except Exception:
            pass
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    dss = [load_jsonl_dataset(p, split_name="train") for p in args.train_files]
    ds = dss[0] if len(dss) == 1 else __import__("datasets").concatenate_datasets(dss)

    if args.only_task:
        want = args.only_task.strip().lower()
        ds = ds.filter(lambda ex: (ex.get("task", "") or "").lower() == want)

    base_collator = DataCollatorTraffic(
        tok,
        chunk_tokenizer=enc_tok,
        chunk_size_tokens=510,
        chunk_overlap=0,
        join_fields_before_chunk=True,
        add_field_header=True,
        chunk_token_factor=int(args.num_chunk_tokens),
        max_length=args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        add_eos_to_labels=True,
        wrap_reason_with_tags=False,
    )

    def rl_collator(examples):
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

    dl = DataLoader(HFWrap(ds), batch_size=args.batch_size, shuffle=True, collate_fn=rl_collator)

    eval_dl = None
    if args.eval_files:
        eval_dss = [load_jsonl_dataset(p, split_name="validation") for p in args.eval_files]
        eval_ds = eval_dss[0] if len(eval_dss) == 1 else __import__("datasets").concatenate_datasets(eval_dss)
        if args.only_task:
            want = args.only_task.strip().lower()
            eval_ds = eval_ds.filter(lambda ex: (ex.get("task", "") or "").lower() == want)
        eval_dl = DataLoader(HFWrap(eval_ds), batch_size=args.batch_size, shuffle=False, collate_fn=rl_collator)

    cfg = GRPOConfig(
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta_kl=args.beta_kl,
        stop_token_ids=stop_ids,
        clip_epsilon_low=args.clip_eps,
        clip_epsilon_high=args.clip_eps,
        num_policy_updates=1,
        loss_on_answer_only=True,
        adv_std_mode=args.adv_std_mode,
        use_old_logps=bool(args.use_old_logps),
    )

    cfg.validate()

    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "grpo_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(dl) / max(1, args.grad_accum))
    max_train_steps = num_update_steps_per_epoch * args.num_epochs
    if accelerator.is_main_process:
        print(f"[info] num_update_steps_per_epoch={num_update_steps_per_epoch}, max_train_steps={max_train_steps}")

    global_step = 0
    optim.zero_grad(set_to_none=True)
    disable_bar = bool(args.disable_tqdm) or (not accelerator.is_local_main_process)
    progress_bar = tqdm(
        total=max_train_steps,
        desc="GRPO train",
        position=0,
        disable=disable_bar,
    )

    def run_eval_once(step: int) -> Optional[float]:
        if eval_dl is None:
            return None
        model.eval()
        total_reward = 0.0
        n_batches = 0
        for batch in eval_dl:
            roll = rollout_and_cache(
                accelerator=accelerator,
                model=model,
                tokenizer=tok,
                batch=batch,
                cfg=cfg,
                ref_model=ref_model,
                micro_batch_logps=args.micro_batch_logps or None,
            )
            total_reward += roll.rewards.mean().item()
            n_batches += 1
            if args.num_eval_batches > 0 and n_batches >= args.num_eval_batches:
                break
        er = (total_reward / n_batches) if n_batches else None
        if er is not None:
            wlog({"eval/mean_reward": float(er)}, step=step)
        return er

    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(dl, start=1):
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            roll = rollout_and_cache(
                accelerator=accelerator,
                model=model,
                tokenizer=tok,
                batch=batch,
                cfg=cfg,
                ref_model=ref_model,
                micro_batch_logps=args.micro_batch_logps or None,
            )

            model.train()
            m = accelerator.unwrap_model(model)

            try:
                if hasattr(m, "decoder") and hasattr(m.decoder, "config") and m.decoder.config is not None:
                    m.decoder.config.use_cache = False
            except Exception:
                pass

            if accelerator.is_main_process and global_step == 0 and batch_idx == 1:
                prompt_len = (
                    roll.rep["prefix_mask"].sum(dim=1).max().item()
                    + roll.chunk_mask.sum(dim=1).max().item()
                    + roll.image_mask.sum(dim=1).max().item()
                    + roll.rep["suffix_mask"].sum(dim=1).max().item()
                )
                comp_len = roll.stop_mask.sum(dim=1).max().item()
                print(f"[debug] B_total={roll.completion_ids.size(0)} prompt_len={prompt_len} comp_len={comp_len} total={prompt_len+comp_len}")

            Btot = int(roll.completion_ids.size(0))
            mb = int(args.micro_batch_logps) if (args.micro_batch_logps and args.micro_batch_logps > 0) else Btot

            loss_weighted_sum = 0.0

            clip_num_sum = 0.0
            ratio_sum = 0.0
            denom_ratio_sum = 0.0

            kl_sum = 0.0
            denom_kl_sum = 0.0

            for st in range(0, Btot, mb):
                ed = min(Btot, st + mb)
                sl = slice(st, ed)
                bs_mb = int(ed - st)

                new_logps_mb = get_per_token_logps_smartcity(
                    m,
                    prefix_ids=roll.rep["prefix_ids"][sl],
                    prefix_mask=roll.rep["prefix_mask"][sl],
                    suffix_ids=roll.rep["suffix_ids"][sl],
                    suffix_mask=roll.rep["suffix_mask"][sl],
                    chunk_embeds=roll.chunk_embeds[sl],
                    chunk_mask=roll.chunk_mask[sl],
                    image_embeds=roll.image_embeds[sl],
                    image_mask=roll.image_mask[sl],
                    completion_ids=roll.completion_ids[sl],
                    completion_mask=roll.stop_mask[sl],
                    task_ids=(roll.rep.get("task_ids", None)[sl] if roll.rep.get("task_ids", None) is not None else None),
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

                scale = (bs_mb / float(Btot)) / max(1, args.grad_accum)
                accelerator.backward(loss_mb * scale)

                loss_weighted_sum += float(loss_mb.detach().item()) * bs_mb

                with torch.no_grad():
                    if roll.old_per_token_logps is not None:
                        old_mb = roll.old_per_token_logps[sl].to(new_logps_mb.dtype)
                        ratio = torch.exp(new_logps_mb.detach() - old_mb)

                        msk = roll.loss_mask[sl].to(ratio.dtype)
                        denom = msk.sum().clamp(min=1.0)

                        lo = 1.0 - float(cfg.clip_epsilon_low)
                        hi = 1.0 + float(cfg.clip_epsilon_high)
                        clipped = ((ratio < lo) | (ratio > hi)).to(msk.dtype)

                        clip_num_sum += float((clipped * msk).sum().item())
                        ratio_sum += float((ratio * msk).sum().item())
                        denom_ratio_sum += float(denom.item())

                    if cfg.beta_kl != 0.0 and roll.ref_per_token_logps is not None:
                        ref_mb = roll.ref_per_token_logps[sl].to(new_logps_mb.dtype)
                        delta = ref_mb - new_logps_mb.detach()
                        per_tok_kl = torch.exp(delta) - delta - 1.0

                        msk = roll.loss_mask[sl].to(per_tok_kl.dtype)
                        denom = msk.sum().clamp(min=1.0)

                        kl_sum += float((per_tok_kl * msk).sum().item())
                        denom_kl_sum += float(denom.item())

                # Release graph tensors as early as possible to reduce memory pressure
                del new_logps_mb, loss_mb

            loss_value = loss_weighted_sum / max(1.0, float(Btot))
            clip_frac = (clip_num_sum / denom_ratio_sum) if denom_ratio_sum > 0 else None
            ratio_mean = (ratio_sum / denom_ratio_sum) if denom_ratio_sum > 0 else None
            kl_mean = (kl_sum / denom_kl_sum) if denom_kl_sum > 0 else None

            if batch_idx % max(1, args.grad_accum) == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if accelerator.is_main_process:
                    progress_bar.update(1)

                with torch.no_grad():
                    r = roll.rewards
                    r_mean = float(r.mean().item())
                    r_std = float(r.std(unbiased=False).item())
                    r_min = float(r.min().item())
                    r_max = float(r.max().item())

                    len_stop = roll.stop_mask.sum(dim=1).float()
                    len_loss = roll.loss_mask.sum(dim=1).float()
                    len_stop_mean = float(len_stop.mean().item())
                    len_loss_mean = float(len_loss.mean().item())

                    think_close_rate = 0.0
                    if roll.reward_info and "think_close_rate" in roll.reward_info:
                        think_close_rate = float(roll.reward_info["think_close_rate"])

                    lr = float(optim.param_groups[0]["lr"])
                    gnorm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                    gpu_mem_gb = None
                    if torch.cuda.is_available():
                        try:
                            gpu_mem_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                        except Exception:
                            gpu_mem_gb = None

                if accelerator.is_main_process and (global_step % args.log_steps == 0):
                    post = {"loss": float(loss_value), "r": r_mean}
                    progress_bar.set_postfix(post)

                log_dict = {
                    "train/loss": float(loss_value),
                    "train/reward_mean": r_mean,
                    "train/reward_std": r_std,
                    "train/reward_min": r_min,
                    "train/reward_max": r_max,
                    "train/len_stop_mean": len_stop_mean,
                    "train/len_loss_mean": len_loss_mean,
                    "train/think_close_rate": float(think_close_rate),
                    "train/grad_norm": gnorm,
                    "train/lr": lr,
                }
                if gpu_mem_gb is not None:
                    log_dict["train/gpu_mem_gb"] = gpu_mem_gb
                if clip_frac is not None:
                    log_dict["train/clip_frac"] = float(clip_frac)
                if ratio_mean is not None:
                    log_dict["train/ratio_mean"] = float(ratio_mean)
                if kl_mean is not None:
                    log_dict["train/kl_mean"] = float(kl_mean)

                if roll.reward_info:
                    for k, v in roll.reward_info.items():
                        log_dict[f"reward/{k}"] = float(v)

                wlog(log_dict, step=global_step)

                if args.log_text_steps and (global_step % int(args.log_text_steps) == 0) and roll.texts and accelerator.is_main_process:
                    import wandb
                    sample_txt = roll.texts[0]
                    wlog({"sample/text": wandb.Html(f"<pre>{sample_txt}</pre>")}, step=global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args.output_dir, global_step, model, tok, accelerator)

                if args.eval_steps > 0 and eval_dl is not None and global_step % args.eval_steps == 0:
                    er = run_eval_once(global_step)
                    if accelerator.is_main_process and er is not None:
                        print(f"[eval] step={global_step} mean_reward={er:.4f}")

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    if accelerator.is_main_process:
        progress_bar.close()

    save_checkpoint(args.output_dir, global_step, model, tok, accelerator)

    if accelerator.is_main_process:
        print(f"[done] global_step={global_step}, num_epochs={args.num_epochs}")

    if wandb_run is not None and accelerator.is_main_process:
        wandb_run.finish()


if __name__ == "__main__":
    main()
