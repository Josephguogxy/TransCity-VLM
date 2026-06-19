#!/usr/bin/env python
# coding: utf-8
"""
GRPO training for SmartCityLLM (StageC -> GRPO), deepspeed-launch friendly.

- Accept --local_rank injected by `deepspeed` launcher.
- Reuse your SmartCityLLM framework + MixLoRA architecture.
_smartcity_load_sft_checkpoint(model, )` (NO tokenizer resize, verbose=True)
- Freezing is controlled ONLY by:
    --open_roberta / --open_vpma / --open_image_projector / --open_imagebind / --freeze_decoder
- For Qwen-prefix StageD GRPO, DataCollatorSmartCity pre-fills assistant context; legacy /think injection is disabled when SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX=1.
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import inspect
import time
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
try:
    from accelerate.utils import DeepSpeedPlugin
except Exception:  # pragma: no cover - handled at runtime if --deepspeed is used
    DeepSpeedPlugin = None

from dataprocessing import load_jsonl_dataset, ensure_smartcity_features, DataCollatorSmartCity
from model.builder import build_smartcity_llm

from model.reinforcement_learning.config import GRPOConfig
from model.reinforcement_learning.logprobs import get_per_token_logps_smartcity
from model.reinforcement_learning.loss import grpo_loss
from model.reinforcement_learning.rewards import ExampleMeta
from model.reinforcement_learning.rollout import rollout_and_cache


THINK_SOFT = "/think"
NO_THINK_SOFT = "/no_think"


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


def safe_gather_for_metrics(accelerator, tensor, name: str = "metric"):
    """
    During smoke tests, training has already happened before metrics are gathered.
    Some Accelerate + DeepSpeed launcher + Gloo combinations fail in gather_for_metrics
    with world-size mismatch. Use local metrics by default.

    Set SMARTCITY_LOCAL_METRICS=0 to try distributed metric gather again.
    """
    import os
    import torch

    if tensor is None:
        return tensor

    local_only = os.environ.get("SMARTCITY_LOCAL_METRICS", "1").strip().lower()
    if local_only in {"1", "true", "yes", "on"}:
        return tensor.detach() if torch.is_tensor(tensor) else tensor

    try:
        return accelerator.gather_for_metrics(tensor)
    except Exception as e:
        try:
            if accelerator.is_main_process:
                print(f"[warn] gather_for_metrics failed for {name}: {repr(e)}; using local metric only")
        except Exception:
            pass
        return tensor.detach() if torch.is_tensor(tensor) else tensor

def _stageD_train_dbg(accelerator, msg: str, t0=None) -> None:
    if os.environ.get("SMARTCITY_TIMING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    try:
        rank = int(getattr(accelerator, "process_index", 0))
    except Exception:
        rank = 0

    all_ranks = os.environ.get("SMARTCITY_TIMING_ALL_RANKS", "0").strip().lower() in {"1", "true", "yes", "on"}
    show_rank = int(os.environ.get("SMARTCITY_TIMING_RANK", "0"))

    if (not all_ranks) and rank != show_rank:
        return

    dt = ""
    if t0 is not None:
        dt = f" | dt={time.perf_counter() - t0:.2f}s"

    print(f"[stageD-timing][rank{rank}] {msg}{dt}", flush=True)

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


def inject_soft_no_think_in_user(prompt: str) -> str:
    p = prompt or ""
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = p.rfind(marker)
    if idx < 0:
        if NO_THINK_SOFT in p[-400:]:
            return p
        return p.rstrip() + f"\n{NO_THINK_SOFT}\n"
    before = p[:idx]
    if NO_THINK_SOFT in before[-400:]:
        return p
    return before.rstrip() + f"\n{NO_THINK_SOFT}\n" + p[idx:]


def qwen_reason_template_prefix_enabled() -> bool:
    return os.environ.get("SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX", "0").strip().lower() in {"1", "true", "yes", "on", "y"}


def legacy_soft_think_disabled() -> bool:
    # New route:
    #   think    context: <think>\n
    #   no_think context: <think>\n\n</think>\n\n
    # Therefore GRPO must not also inject legacy /think or /no_think into the user prompt.
    forced = os.environ.get("SMARTCITY_GRPO_DISABLE_LEGACY_SOFT_THINK", "0").strip().lower() in {"1", "true", "yes", "on", "y"}
    return forced or qwen_reason_template_prefix_enabled()


def infer_reasoning_mode_from_example(ex: Dict[str, Any], default_mode: str = "auto") -> str:
    mode = str(ex.get("reasoning_mode", "") or "").strip().lower().replace("-", "_")
    if mode in ("think", "thinking", "cot"):
        return "think"
    if mode in ("no_think", "nothink", "no", "none", "instant"):
        return "no_think"

    if default_mode and default_mode.lower() not in ("auto", ""):
        d = default_mode.strip().lower().replace("-", "_")
        if d in ("think", "thinking", "cot"):
            return "think"
        if d in ("no_think", "nothink", "no", "none", "instant"):
            return "no_think"

    thinking = ex.get("thinking", []) or []
    if isinstance(thinking, list) and any(str(x).strip() for x in thinking):
        return "think"

    ans = str(ex.get("answer", "") or "")
    if "<think>" in ans.lower() or "</think>" in ans.lower():
        return "think"

    return "no_think"


def _mixlora_upr_indices(num_universal: int, num_pred: int, num_reason: int):
    u = int(num_universal)
    p = int(num_pred)
    r = int(num_reason)
    U = list(range(0, u))
    P = list(range(u, u + p))
    R = list(range(u + p, u + p + r))
    return U, P, R


def apply_reason_only_mixlora_gradient_mask(
    model: torch.nn.Module,
    *,
    num_universal: int,
    num_pred: int,
    num_reason: int,
    freeze_router_hidden: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Stage IV reasoning-only RL:
    - U experts: gradient zero
    - P experts: gradient zero
    - R experts: trainable
    - Router: only output rows corresponding to R experts are trainable.

    注意：
    MixLoRAExperts 的 A/B 是一个大 Parameter，无法对 expert slice 单独 requires_grad=False。
    因此这里用 gradient hook 对 U/P slices 置零。
    """
    U, P, R = _mixlora_upr_indices(num_universal, num_pred, num_reason)
    allowed = set(R)
    total_experts = int(num_universal) + int(num_pred) + int(num_reason)

    stats = {
        "mixlora_expert_modules": 0,
        "router_modules": 0,
        "router_hidden_linears_frozen": 0,
        "hooks_registered": 0,
    }

    def make_hook(mask: torch.Tensor):
        def _hook(grad: torch.Tensor):
            return grad * mask.to(device=grad.device, dtype=grad.dtype)
        return _hook

    for mod in model.modules():
        cls_name = mod.__class__.__name__

        if cls_name == "MixLoRAExperts" and hasattr(mod, "A") and hasattr(mod, "B"):
            stats["mixlora_expert_modules"] += 1
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)

            if getattr(mod, "_variable_r", False) and hasattr(mod, "_offsets"):
                offsets = list(getattr(mod, "_offsets"))
                a_mask = torch.zeros_like(mod.A.data, dtype=torch.float32)
                b_mask = torch.zeros_like(mod.B.data, dtype=torch.float32)
                for eid in allowed:
                    if 0 <= eid < len(offsets) - 1:
                        st, ed = offsets[eid], offsets[eid + 1]
                        a_mask[st:ed, :] = 1.0
                        b_mask[:, st:ed] = 1.0
            else:
                rank = int(getattr(mod, "rank", 0))
                if rank <= 0:
                    continue
                a_mask = torch.zeros_like(mod.A.data, dtype=torch.float32)
                b_mask = torch.zeros_like(mod.B.data, dtype=torch.float32)
                for eid in allowed:
                    if 0 <= eid < total_experts:
                        st, ed = eid * rank, (eid + 1) * rank
                        a_mask[st:ed, :] = 1.0
                        b_mask[:, st:ed] = 1.0

            mod.A.register_hook(make_hook(a_mask))
            mod.B.register_hook(make_hook(b_mask))
            setattr(mod, "_reason_only_train_mask_A", a_mask.detach().cpu())
            setattr(mod, "_reason_only_train_mask_B", b_mask.detach().cpu())
            setattr(mod, "_reason_only_frozen_A", mod.A.detach().cpu().clone())
            setattr(mod, "_reason_only_frozen_B", mod.B.detach().cpu().clone())
            stats["hooks_registered"] += 2

        elif cls_name == "MaskedTokenRouter" and hasattr(mod, "mlp"):
            stats["router_modules"] += 1
            linears = [m for m in mod.mlp.modules() if isinstance(m, torch.nn.Linear)]

            if not linears:
                continue

            if freeze_router_hidden and len(linears) > 1:
                for lin in linears[:-1]:
                    for p in lin.parameters():
                        p.requires_grad_(False)
                stats["router_hidden_linears_frozen"] += len(linears) - 1

            last = linears[-1]
            last.weight.requires_grad_(True)
            row_mask = torch.zeros_like(last.weight.data, dtype=torch.float32)
            for eid in allowed:
                if 0 <= eid < row_mask.size(0):
                    row_mask[eid, :] = 1.0
            last.weight.register_hook(make_hook(row_mask))
            setattr(last, "_reason_only_train_mask_weight", row_mask.detach().cpu())
            setattr(last, "_reason_only_frozen_weight", last.weight.detach().cpu().clone())
            stats["hooks_registered"] += 1

            if last.bias is not None:
                last.bias.requires_grad_(True)
                bias_mask = torch.zeros_like(last.bias.data, dtype=torch.float32)
                for eid in allowed:
                    if 0 <= eid < bias_mask.size(0):
                        bias_mask[eid] = 1.0
                last.bias.register_hook(make_hook(bias_mask))
                setattr(last, "_reason_only_train_mask_bias", bias_mask.detach().cpu())
                setattr(last, "_reason_only_frozen_bias", last.bias.detach().cpu().clone())
                stats["hooks_registered"] += 1

    if verbose:
        print(
            "[reason-only-freeze] "
            f"U={U}, P={P}, R={R}, "
            f"mixlora_modules={stats['mixlora_expert_modules']}, "
            f"routers={stats['router_modules']}, "
            f"hooks={stats['hooks_registered']}"
        )

    return stats


def restore_reason_only_frozen_slices(model: torch.nn.Module) -> None:
    """
    Gradient hooks stop gradients for frozen U/P slices, but AdamW weight decay
    and optimizer state can still mutate zero-gradient slices. Restore masked
    slices after every optimizer step.
    """
    with torch.no_grad():
        for mod in model.modules():
            if mod.__class__.__name__ == "MixLoRAExperts" and hasattr(mod, "A") and hasattr(mod, "B"):
                for pname in ("A", "B"):
                    p = getattr(mod, pname, None)
                    mask = getattr(mod, f"_reason_only_train_mask_{pname}", None)
                    frozen = getattr(mod, f"_reason_only_frozen_{pname}", None)
                    if p is None or mask is None or frozen is None:
                        continue
                    if tuple(mask.shape) != tuple(p.shape) or tuple(frozen.shape) != tuple(p.shape):
                        continue
                    mask_t = mask.to(device=p.device, dtype=p.dtype)
                    frozen_t = frozen.to(device=p.device, dtype=p.dtype)
                    p.data.mul_(mask_t).add_(frozen_t * (1.0 - mask_t))

            if mod.__class__.__name__ == "Linear":
                for pname in ("weight", "bias"):
                    p = getattr(mod, pname, None)
                    mask = getattr(mod, f"_reason_only_train_mask_{pname}", None)
                    frozen = getattr(mod, f"_reason_only_frozen_{pname}", None)
                    if p is None or mask is None or frozen is None:
                        continue
                    if tuple(mask.shape) != tuple(p.shape) or tuple(frozen.shape) != tuple(p.shape):
                        continue
                    mask_t = mask.to(device=p.device, dtype=p.dtype)
                    frozen_t = frozen.to(device=p.device, dtype=p.dtype)
                    p.data.mul_(mask_t).add_(frozen_t * (1.0 - mask_t))


def _mixlora_group_masks(
    shape: torch.Size,
    *,
    module: torch.nn.Module,
    param_kind: str,
    U: List[int],
    P: List[int],
    R: List[int],
    total_experts: int,
) -> Dict[str, torch.Tensor]:
    out = {k: torch.zeros(shape, dtype=torch.float32) for k in ("U", "P", "R")}
    groups = {"U": U, "P": P, "R": R}

    def _fill_mixlora_mask_slice(m: torch.Tensor, st: int, ed: int, kind: str) -> None:
        # Some ZeRO-3 diagnostic tensors may appear flattened / 1D.
        # This mask is only for diagnostics, so never crash here.
        if m.ndim == 0:
            return

        st = int(st)
        ed = int(ed)

        if m.ndim == 1:
            st = max(0, min(st, m.shape[0]))
            ed = max(0, min(ed, m.shape[0]))
            if ed > st:
                m[st:ed] = 1.0
            return

        if kind == "A":
            st = max(0, min(st, m.shape[0]))
            ed = max(0, min(ed, m.shape[0]))
            if ed > st:
                m[st:ed, ...] = 1.0
        else:
            last = m.ndim - 1
            st = max(0, min(st, m.shape[last]))
            ed = max(0, min(ed, m.shape[last]))
            if ed > st:
                idx = [slice(None)] * m.ndim
                idx[last] = slice(st, ed)
                m[tuple(idx)] = 1.0

    if getattr(module, "_variable_r", False) and hasattr(module, "_offsets"):
        offsets = list(getattr(module, "_offsets"))
        for gname, ids in groups.items():
            m = out[gname]
            for eid in ids:
                if 0 <= eid < len(offsets) - 1:
                    st, ed = offsets[eid], offsets[eid + 1]
                    _fill_mixlora_mask_slice(m, st, ed, param_kind)
        return out

    rank = int(getattr(module, "rank", 0))
    if rank <= 0:
        return out

    for gname, ids in groups.items():
        m = out[gname]
        for eid in ids:
            if 0 <= eid < total_experts:
                st, ed = eid * rank, (eid + 1) * rank
                _fill_mixlora_mask_slice(m, st, ed, param_kind)
    return out

def _router_group_masks(shape: torch.Size, U: List[int], P: List[int], R: List[int]) -> Dict[str, torch.Tensor]:
    out = {k: torch.zeros(shape, dtype=torch.float32) for k in ("U", "P", "R")}
    for gname, ids in {"U": U, "P": P, "R": R}.items():
        m = out[gname]
        for eid in ids:
            if 0 <= eid < shape[0]:
                if len(shape) == 1:
                    m[eid] = 1.0
                else:
                    m[eid, :] = 1.0
    return out


def capture_reason_only_values(
    model: torch.nn.Module,
    *,
    num_universal: int,
    num_pred: int,
    num_reason: int,
) -> List[Tuple[torch.nn.Parameter, torch.Tensor, Dict[str, torch.Tensor]]]:
    U, P, R = _mixlora_upr_indices(num_universal, num_pred, num_reason)
    total = int(num_universal) + int(num_pred) + int(num_reason)
    rows: List[Tuple[torch.nn.Parameter, torch.Tensor, Dict[str, torch.Tensor]]] = []
    for mod in model.modules():
        if mod.__class__.__name__ == "MixLoRAExperts" and hasattr(mod, "A") and hasattr(mod, "B"):
            for pname in ("A", "B"):
                p = getattr(mod, pname)
                masks = _mixlora_group_masks(
                    p.shape, module=mod, param_kind=pname, U=U, P=P, R=R, total_experts=total
                )
                rows.append((p, p.detach().float().cpu().clone(), masks))
        elif mod.__class__.__name__ == "MaskedTokenRouter" and hasattr(mod, "mlp"):
            linears = [m for m in mod.mlp.modules() if isinstance(m, torch.nn.Linear)]
            if not linears:
                continue
            last = linears[-1]
            rows.append((last.weight, last.weight.detach().float().cpu().clone(), _router_group_masks(last.weight.shape, U, P, R)))
            if last.bias is not None:
                rows.append((last.bias, last.bias.detach().float().cpu().clone(), _router_group_masks(last.bias.shape, U, P, R)))
    return rows


def reason_only_delta_norms(snapshot: List[Tuple[torch.nn.Parameter, torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, float]:
    ss = {"U": 0.0, "P": 0.0, "R": 0.0}
    for p, before, masks in snapshot:
        cur = p.detach().float().cpu()
        delta = cur - before
        for g in ss:
            m = masks[g].to(dtype=delta.dtype)
            ss[g] += float(((delta * m) ** 2).sum().item())
    return {f"delta_norm_{g}": float(math.sqrt(v)) for g, v in ss.items()}


def reason_only_grad_norms(
    model: torch.nn.Module,
    *,
    num_universal: int,
    num_pred: int,
    num_reason: int,
) -> Dict[str, float]:
    U, P, R = _mixlora_upr_indices(num_universal, num_pred, num_reason)
    total = int(num_universal) + int(num_pred) + int(num_reason)
    ss = {"U": 0.0, "P": 0.0, "R": 0.0}
    for mod in model.modules():
        if mod.__class__.__name__ == "MixLoRAExperts" and hasattr(mod, "A") and hasattr(mod, "B"):
            for pname in ("A", "B"):
                p = getattr(mod, pname)
                if p.grad is None:
                    continue
                masks = _mixlora_group_masks(
                    p.shape, module=mod, param_kind=pname, U=U, P=P, R=R, total_experts=total
                )
                grad = p.grad.detach().float().cpu()
                for g in ss:
                    ss[g] += float(((grad * masks[g]) ** 2).sum().item())
        elif mod.__class__.__name__ == "MaskedTokenRouter" and hasattr(mod, "mlp"):
            linears = [m for m in mod.mlp.modules() if isinstance(m, torch.nn.Linear)]
            if not linears:
                continue
            last = linears[-1]
            for p in (last.weight, last.bias):
                if p is None or p.grad is None:
                    continue
                masks = _router_group_masks(p.shape, U, P, R)
                grad = p.grad.detach().float().cpu()
                for g in ss:
                    ss[g] += float(((grad * masks[g]) ** 2).sum().item())
    return {f"grad_norm_{g}": float(math.sqrt(v)) for g, v in ss.items()}


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

def _parse_size_to_bytes(x: str | int | float) -> int:
    """
    Parse strings like '2GB', '500MB', '1024' into bytes.
    """
    if isinstance(x, (int, float)):
        return int(x)

    s = str(x).strip().lower().replace(" ", "")
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)(b|kb|kib|mb|mib|gb|gib|tb|tib)?$", s)
    if not m:
        raise ValueError(f"Invalid size string: {x!r}")

    val = float(m.group(1))
    unit = m.group(2) or "b"

    mult = {
        "b": 1,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "tb": 1000 ** 4,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
        "tib": 1024 ** 4,
    }[unit]

    return int(val * mult)


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _save_sharded_pytorch_state_dict(
    state_dict: Dict[str, torch.Tensor],
    save_dir: str | Path,
    *,
    max_shard_size: str | int = "2GB",
) -> None:
    """
    Save a full model state_dict in HF-style sharded PyTorch format:

      pytorch_model.bin.index.json
      pytorch_model-00001-of-000xx.bin
      ...

    This is intentionally compatible with SmartCityLLM.load_full_state_dict().
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = _parse_size_to_bytes(max_shard_size)

    # Remove old full-model shards if re-saving into the same checkpoint dir.
    for pat in (
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ):
        for p in save_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass

    keys = list(state_dict.keys())
    shards: List[List[str]] = []
    cur: List[str] = []
    cur_size = 0
    total_size = 0

    for k in keys:
        v = state_dict[k]
        if not torch.is_tensor(v):
            continue
        sz = _tensor_nbytes(v)
        total_size += sz

        if cur and (cur_size + sz > max_bytes):
            shards.append(cur)
            cur = []
            cur_size = 0

        cur.append(k)
        cur_size += sz

    if cur:
        shards.append(cur)

    if not shards:
        raise RuntimeError("No tensor found in state_dict; cannot save full checkpoint.")

    n_shards = len(shards)
    weight_map: Dict[str, str] = {}

    for idx, shard_keys in enumerate(shards, start=1):
        shard_name = f"pytorch_model-{idx:05d}-of-{n_shards:05d}.bin"
        shard_path = save_dir / shard_name

        shard_state = {}
        for k in shard_keys:
            v = state_dict[k]
            if torch.is_tensor(v):
                shard_state[k] = v.detach().cpu()
                weight_map[k] = shard_name

        torch.save(shard_state, shard_path)

        # Free shard dict ASAP.
        del shard_state

    index = {
        "metadata": {
            "total_size": int(total_size),
            "format": "pt",
            "note": "Full SmartCityLLM checkpoint saved after GRPO; compatible with load_full_state_dict().",
        },
        "weight_map": weight_map,
    }

    with open(save_dir / "pytorch_model.bin.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(
        f"[save-full] wrote {n_shards} shards, "
        f"total_size={total_size / (1024 ** 3):.2f} GiB -> {save_dir}"
    )


def save_checkpoint_full(
    output_dir: str,
    step: int,
    model,
    tokenizer,
    accelerator: Accelerator,
    *,
    max_shard_size: str = "2GB",
) -> None:
    """
    Save the *full* model checkpoint after GRPO.

    Important for ZeRO-3:
    - accelerator.get_state_dict(model) must be called by all ranks.
    - Only main process writes files.
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"[save-full] gathering full state_dict for checkpoint-{step} ...")

    # This is the key call for DeepSpeed ZeRO-3.
    # It gathers the full fp16/bf16 model weights according to Accelerate/DeepSpeed.
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        _save_sharded_pytorch_state_dict(
            state_dict,
            ckpt_dir,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(str(ckpt_dir))
        with open(ckpt_dir / "grpo_full_checkpoint.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Full model checkpoint saved after GRPO.\n"
                f"step={step}\n"
                f"max_shard_size={max_shard_size}\n"
            )

    del state_dict
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    accelerator.wait_for_everyone()



def save_checkpoint_trainable(output_dir: str, step: int, model, tokenizer, accelerator: Accelerator, keep_last_n: int):
    """
    Save trainable parameters safely under DeepSpeed ZeRO-3.

    Important:
      Under ZeRO-3, trainable parameters are partitioned. Directly saving
      p.detach().cpu() on rank0 may save only shards or invalid tensors.
      Therefore all ranks enter GatheredParameters, and only rank0 writes.
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    m = accelerator.unwrap_model(model)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    state = {}
    used_zero3_gather = False

    try:
        import deepspeed

        params = [p for _, p in named]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            used_zero3_gather = True
            if accelerator.is_main_process:
                bad = []
                for n, p in named:
                    t = p.detach().cpu().clone()
                    if t.numel() == 0:
                        bad.append(n)
                    state[n] = t
                if bad:
                    print(f"[save][warn] zero-sized gathered trainable tensors: {bad[:10]} count={len(bad)}", flush=True)

    except Exception as e:
        if accelerator.is_main_process:
            print(f"[save][warn] ZeRO-3 GatheredParameters failed for trainable checkpoint: {repr(e)}", flush=True)
            print("[save][warn] Falling back to direct trainable_named_parameters(); only safe outside ZeRO-3.", flush=True)
            state = trainable_named_parameters(m)

    if accelerator.is_main_process:
        torch.save(state, ckpt_dir / "adapter_model.bin")
        tokenizer.save_pretrained(str(ckpt_dir))

        with open(ckpt_dir / "grpo_step.txt", "w", encoding="utf-8") as f:
            f.write(str(step))

        with open(ckpt_dir / "adapter_checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": int(step),
                    "num_tensors": int(len(state)),
                    "numel": int(sum(v.numel() for v in state.values() if torch.is_tensor(v))),
                    "zero3_gathered": bool(used_zero3_gather),
                    "note": "StageD trainable adapter checkpoint. Load with StageC full checkpoint first, then adapter_model.bin.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

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
                        print(f"[ckpt] removed old: {d}", flush=True)
                    except Exception:
                        pass

        print(
            f"[save] trainable checkpoint-{step} -> {ckpt_dir} "
            f"| tensors={len(state)} "
            f"| numel={sum(v.numel() for v in state.values() if torch.is_tensor(v))} "
            f"| zero3_gathered={used_zero3_gather}",
            flush=True,
        )

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    accelerator.wait_for_everyone()



# ===== SmartCity StageD helper restore v3 =====
# These helpers are required by the patched DeepSpeed/ZeRO-3 StageD path.

def _smartcity_world_size_from_env() -> int:
    """Best-effort world-size detection before Accelerator.prepare()."""
    import os

    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(name, "")
        try:
            iv = int(v)
            if iv > 0:
                return iv
        except Exception:
            pass

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd.lower() not in {"none", "nodevfiles"}:
        parts = [x for x in cvd.split(",") if x.strip()]
        if parts:
            return len(parts)

    try:
        n = int(os.environ.get("NUM_GPUS", "0") or "0")
        if n > 0:
            return n
    except Exception:
        pass

    return 1



def _smartcity_prepare_deepspeed_config(ds_config_path, output_dir: str, per_device_batch: int):
    """
    Write a runtime DeepSpeed config per rank.

    StageD note:
      - For ZeRO-2, it is safe to write train_batch_size explicitly.
      - For ZeRO-3, transformers.from_pretrained() enters deepspeed.zero.Init()
        before DeepSpeed sees the final distributed world size. In that phase,
        DeepSpeed may treat world_size as 1, so writing train_batch_size=64 for
        micro=8/world=8 causes:
            64 != 8 * 1 * 1
        Therefore, for ZeRO-3 we omit train_batch_size and let DeepSpeed infer it.
    """
    import os
    import json
    from pathlib import Path

    ds_config_path = Path(ds_config_path)
    cfg = json.loads(ds_config_path.read_text(encoding="utf-8"))

    # Keep the Python-side torch.optim.AdamW.
    cfg.pop("optimizer", None)
    cfg.pop("scheduler", None)

    micro = max(1, int(per_device_batch))
    world = max(1, _smartcity_world_size_from_env())
    ds_grad_acc = max(1, int(os.environ.get("SMARTCITY_DS_GRAD_ACC", "1") or "1"))

    zero_opt = cfg.get("zero_optimization") or {}
    try:
        zero_stage = int(zero_opt.get("stage", 0) or 0)
    except Exception:
        zero_stage = 0

    cfg["train_micro_batch_size_per_gpu"] = micro
    cfg["gradient_accumulation_steps"] = ds_grad_acc

    # Critical ZeRO-3 fix:
    # Do NOT set train_batch_size for ZeRO-3. Let DeepSpeed infer it because
    # zero.Init may run while world_size is still observed as 1.
    omit_train_batch = (
        zero_stage == 3
        and os.environ.get("SMARTCITY_ZERO3_OMIT_TRAIN_BATCH_SIZE", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )

    if omit_train_batch:
        cfg.pop("train_batch_size", None)
        batch_msg = (
            f"train_batch_size=inferred by DeepSpeed "
            f"(zero_stage=3, micro={micro}, ds_grad_acc={ds_grad_acc}, "
            f"expected_global_after_dist={micro * ds_grad_acc * world})"
        )
    else:
        cfg["train_batch_size"] = micro * ds_grad_acc * world
        batch_msg = (
            f"train_batch_size={cfg['train_batch_size']} = "
            f"micro={micro} * ds_grad_acc={ds_grad_acc} * world={world}"
        )

    runtime_dir = Path(output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or "0")
    runtime_path = runtime_dir / f"{ds_config_path.stem}.runtime.rank{rank}.json"
    runtime_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if str(rank) == "0":
        print(
            "[ds-config] "
            f"runtime={runtime_path} | zero_stage={zero_stage} | "
            f"{batch_msg}; optimizer/scheduler removed from DS config",
            flush=True,
        )

    return runtime_path

def configure_zero3_leaf_modules_for_mixlora(model, is_main: bool = False) -> None:
    """
    Optional DeepSpeed ZeRO-3 leaf-module setup for dynamic MixLoRA routers.

    In this model the printed module path ends with '.mixlora', e.g.
      gate_proj.mixlora: MixLoRAExperts

    So the useful suffix is usually 'mixlora', not 'experts'.
    Disable with:
      SMARTCITY_Z3_LEAF_SUFFIXES=
    """
    import os

    actual_zero = os.environ.get("SMARTCITY_ACTUAL_ZERO_STAGE", "").strip()
    if actual_zero and actual_zero != "3":
        if is_main:
            print(f"[stageD] skip ZeRO-3 leaf modules because actual_ds_zero_stage={actual_zero}", flush=True)
        return

    suffixes = os.environ.get("SMARTCITY_Z3_LEAF_SUFFIXES", "").strip()
    if not suffixes:
        return

    suffix_list = [x.strip() for x in suffixes.split(",") if x.strip()]
    if not suffix_list:
        return

    try:
        from deepspeed.utils import set_z3_leaf_modules_by_suffix

        set_z3_leaf_modules_by_suffix(model, suffix_list)
        if is_main:
            print(f"[stageD] ZeRO-3 leaf modules by suffix enabled: {suffix_list}", flush=True)

    except Exception as e:
        if is_main:
            print(f"[stageD][warn] failed to set ZeRO-3 leaf modules by suffix {suffix_list}: {repr(e)}", flush=True)


def safe_gather_for_metrics(accelerator, tensor, name: str = "metric"):
    """Metrics must never block or crash Stage D; default to local per-rank metrics."""
    import os
    import torch

    if tensor is None:
        return tensor

    local_only = os.environ.get("SMARTCITY_LOCAL_METRICS", "1").strip().lower()
    if local_only in {"1", "true", "yes", "y", "on"}:
        return tensor.detach() if torch.is_tensor(tensor) else tensor

    try:
        return accelerator.gather_for_metrics(tensor)
    except Exception as e:
        try:
            main = bool(accelerator.is_main_process)
        except Exception:
            main = True
        if main:
            print(f"[warn] gather_for_metrics failed for {name}: {repr(e)}; using local metric only", flush=True)
        return tensor.detach() if torch.is_tensor(tensor) else tensor

# ===== End SmartCity StageD helper restore v3 =====




# ===== SmartCity StageD strict tag prompt helper =====
def stageD_strict_tag_instruction(prompt: str, mode: str) -> str:
    """
    Add mode-specific hard format instructions without pre-filling assistant tokens.
    The model must still generate <think> / </think> itself.
    """
    import os

    if os.environ.get("SMARTCITY_GRPO_STRICT_TAG_PROMPT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return prompt

    p = prompt or ""
    mode = (mode or "auto").strip().lower().replace("-", "_")

    if mode in {"think", "thinking", "cot"}:
        inst = """
STRICT RESPONSE FORMAT FOR THIS SAMPLE:
- Your assistant response must start with the exact characters: <think>
- Do not put any text, punctuation, whitespace, or replacement character before <think>.
- Inside <think>...</think>, write brief evidence planning only.
- You must then output the exact closing tag: </think>
- After </think>, write the public final answer in natural language with valid [E_*] citations.
- Do not put the final answer inside the think block.
""".strip()
    elif mode in {"no_think", "nothink", "no", "none", "instant"}:
        inst = """
STRICT RESPONSE FORMAT FOR THIS SAMPLE:
- Your assistant response must start with the exact empty block: <think></think>
- Do not put any text, punctuation, whitespace, or replacement character before <think></think>.
- Do not write any reasoning inside the think block.
- After <think></think>, write only the public final answer in natural language with valid [E_*] citations.
- Do not generate another <think> or </think> later.
""".strip()
    else:
        inst = """
STRICT RESPONSE FORMAT FOR THIS SAMPLE:
- Your assistant response must start with <think>.
- Use a closed <think>...</think> block, then write the public final answer with valid [E_*] citations.
""".strip()

    if inst in p:
        return p

    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = p.rfind(marker)
    if idx >= 0:
        return p[:idx].rstrip() + "\n\n" + inst + "\n" + p[idx:]
    return p.rstrip() + "\n\n" + inst + "\n"
# ===== End SmartCity StageD strict tag prompt helper =====

# ===== SmartCity StageD freeform evidence prompt v1 =====
_FREEFORM_EVIDENCE_OUTPUT_OVERRIDE = """
IMPORTANT OUTPUT OVERRIDE FOR THIS REASONING TASK:
Ignore any earlier instruction that requires a JSON object.
Use Qwen3 style: keep brief private reasoning inside <think>...</think> if thinking is enabled; after </think>, write the public answer in natural language.
Do not output JSON, markdown, or a confidence field.
Cite evidence ids exactly in square brackets when using evidence, for example [E_HistoricalSeries] or [E_Weather].
Only cite present/support evidence IDs listed in Evidence handles. Never cite missing or do_not_use_as_support evidence as support.
Mention missing evidence only as a limitation, for example: "No accident evidence was provided."
State the decision clearly in words, such as "no significant change / low risk" or "significant change / high risk", and explain why.
""".strip()


def rewrite_reason_prompt_for_freeform_evidence(prompt: str) -> str:
    """Runtime-only prompt override; keeps the dataset JSON gold answer for reward metadata."""
    p = prompt or ""
    if _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE in p:
        return p
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = p.rfind(marker)
    if idx >= 0:
        return p[:idx].rstrip() + "\n\n" + _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE + "\n" + p[idx:]
    return p.rstrip() + "\n\n" + _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE + "\n"
# ===== End SmartCity StageD freeform evidence prompt v1 =====

# ===== SmartCity StageD Qwen freeform evidence prompt v1 =====
_FREEFORM_EVIDENCE_OUTPUT_OVERRIDE = """
Output format id: qwen3_freeform_evidence_v1

IMPORTANT OUTPUT FORMAT FOR THIS REASONING TASK:
Use Qwen3 style. If thinking is enabled, keep brief private reasoning inside <think>...</think>; after </think>, write the public final answer in natural language.
Do not output JSON. Do not output markdown. Do not output a confidence field.
In the final answer, cite evidence handles exactly in square brackets when you rely on them, for example [E_HistoricalSeries] or [E_Weather].
Only cite present/support evidence IDs listed in Evidence handles. Never cite missing or do_not_use_as_support evidence as support.
Mention missing evidence only as a limitation, for example: "No accident evidence was provided."
State the decision clearly in words, such as "no significant change / low risk" or "significant change / high risk", and explain why.
""".strip()


def rewrite_reason_prompt_for_freeform_evidence(prompt: str) -> str:
    """Runtime-only Qwen freeform prompt override for StageD GRPO."""
    p = prompt or ""

    # Avoid duplicate instructions if dataset_stageD/version_1 already contains them.
    if "qwen3_freeform_evidence_v1" in p:
        return p

    if _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE in p:
        return p

    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = p.rfind(marker)
    if idx >= 0:
        return p[:idx].rstrip() + "\n\n" + _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE + "\n" + p[idx:]

    return p.rstrip() + "\n\n" + _FREEFORM_EVIDENCE_OUTPUT_OVERRIDE + "\n"
# ===== End SmartCity StageD Qwen freeform evidence prompt v1 =====





def _smartcity_load_sft_checkpoint(model, ckpt_dir, verbose=True):
    """
    Flexible SFT/base checkpoint loader for StageD GRPO.

    Order:
      1) SmartCityLLM.load_full_state_dict(...) for sharded StageC/ZeRO-style checkpoints.
      2) single-file model.safetensors from HF Trainer / StageD SFT warmup.
      3) single-file pytorch_model.bin fallback.

    This function intentionally does not resize tokenizer/model embeddings.
    """
    from pathlib import Path
    import gc
    import torch

    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"SFT checkpoint directory/file does not exist: {ckpt}")

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    errors = []

    # 1) Native SmartCity sharded loader.
    # This supports pytorch_model.bin.index.json and model.safetensors.index.json.
    if hasattr(model, "load_full_state_dict"):
        try:
            try:
                model.load_full_state_dict(str(ckpt), cast_to_target_dtype=True, verbose=verbose)
            except TypeError:
                model.load_full_state_dict(str(ckpt), verbose=verbose)
            _log(f"[load-flex] loaded by model.load_full_state_dict: {ckpt}")
            return {"method": "load_full_state_dict", "path": str(ckpt)}
        except Exception as e:
            errors.append(("load_full_state_dict", repr(e)))
            _log(f"[load-flex][warn] load_full_state_dict failed for {ckpt}: {repr(e)}")
    else:
        errors.append(("load_full_state_dict", "model has no load_full_state_dict"))
        _log("[load-flex][warn] model has no load_full_state_dict; trying single-file checkpoints")

    def _load_state_dict_flexible(sd, label):
        # Try as-is first. If every key has module. prefix, strip it and retry.
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            return missing, unexpected, "as_is"
        except RuntimeError as e:
            first_error = repr(e)
            if sd and all(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
                sd2 = {k[len("module."):]: v for k, v in sd.items()}
                missing, unexpected = model.load_state_dict(sd2, strict=False)
                sd.clear()
                sd.update(sd2)
                return missing, unexpected, "strip_module_prefix"
            raise RuntimeError(f"load_state_dict failed for {label}: {first_error}") from e

    # 2) HF Trainer single safetensors checkpoint.
    sf = ckpt / "model.safetensors"
    if sf.is_file():
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("model.safetensors exists but safetensors is not importable") from e

        sd = load_file(str(sf), device="cpu")
        missing, unexpected, mode = _load_state_dict_flexible(sd, str(sf))
        _log(f"[load-flex] loaded model.safetensors: {sf}")
        _log(f"[load-flex] tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)} mode={mode}")
        if missing[:10]:
            _log(f"[load-flex][missing:first10] {missing[:10]}")
        if unexpected[:10]:
            _log(f"[load-flex][unexpected:first10] {unexpected[:10]}")
        del sd
        gc.collect()
        return {
            "method": "model.safetensors",
            "path": str(sf),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    # 3) Plain PyTorch single-file checkpoint.
    pt = ckpt / "pytorch_model.bin"
    if pt.is_file():
        sd = torch.load(str(pt), map_location="cpu")
        if isinstance(sd, dict):
            for key in ("state_dict", "model", "module"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        if not isinstance(sd, dict):
            raise RuntimeError(f"pytorch_model.bin did not contain a state dict: {pt}")

        missing, unexpected, mode = _load_state_dict_flexible(sd, str(pt))
        _log(f"[load-flex] loaded pytorch_model.bin: {pt}")
        _log(f"[load-flex] tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)} mode={mode}")
        if missing[:10]:
            _log(f"[load-flex][missing:first10] {missing[:10]}")
        if unexpected[:10]:
            _log(f"[load-flex][unexpected:first10] {unexpected[:10]}")
        del sd
        gc.collect()
        return {
            "method": "pytorch_model.bin",
            "path": str(pt),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    msg = (
        f"Cannot load SFT checkpoint from {ckpt}. "
        f"Tried native loader, model.safetensors, pytorch_model.bin. errors={errors}"
    )
    raise FileNotFoundError(msg)



def _smartcity_maybe_load_trainable_adapter(model, verbose=True):
    """
    Optional loader for StageD GRPO trainable-only checkpoints.

    Use:
      SMARTCITY_LOAD_GRPO_ADAPTER=/path/to/checkpoint-2000
    or:
      SMARTCITY_LOAD_GRPO_ADAPTER=/path/to/checkpoint-2000/adapter_model.bin

    This loads adapter_model.bin after the full stageC_final checkpoint has been loaded.
    """
    import os
    import gc
    import torch
    from pathlib import Path

    adapter = (
        os.environ.get("SMARTCITY_LOAD_GRPO_ADAPTER", "")
        or os.environ.get("SMARTCITY_LOAD_TRAINABLE_ADAPTER", "")
    ).strip()

    if not adapter:
        return None

    ap = Path(adapter)
    if ap.is_dir():
        ap = ap / "adapter_model.bin"

    if not ap.is_file():
        raise FileNotFoundError(
            f"SMARTCITY_LOAD_GRPO_ADAPTER is set but adapter_model.bin was not found: {ap}"
        )

    if verbose:
        print(f"[load-adapter] loading trainable adapter: {ap}", flush=True)

    sd = torch.load(str(ap), map_location="cpu")

    if isinstance(sd, dict):
        for key in ("state_dict", "model", "module"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break

    if not isinstance(sd, dict):
        raise RuntimeError(f"adapter_model.bin did not contain a state dict: {ap}")

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        mode = "as_is"
    except RuntimeError:
        if sd and all(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
            sd2 = {k[len("module."):]: v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd2, strict=False)
            mode = "strip_module_prefix"
            sd.clear()
            sd.update(sd2)
        else:
            raise

    if verbose:
        print(
            f"[load-adapter] loaded trainable adapter: tensors={len(sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)} mode={mode}",
            flush=True,
        )
        if unexpected[:20]:
            print(f"[load-adapter][unexpected:first20] {unexpected[:20]}", flush=True)
        if missing[:20]:
            print(f"[load-adapter][missing:first20] {missing[:20]}", flush=True)

    del sd
    gc.collect()
    return {
        "method": "adapter_model.bin",
        "path": str(ap),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def main():
    ap = argparse.ArgumentParser()

    # ✅ deepspeed launcher injects this
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))
    ap.add_argument("--deepspeed", type=str, default="", help="Optional DeepSpeed config path for ZeRO training.")

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

    ap.add_argument("--freeze_mixlora_non_reason", type=str2bool, default=True)
    ap.add_argument("--default_reasoning_mode", type=str, default="auto")
    ap.add_argument(
        "--inject_no_think_token",
        type=str2bool,
        default=False,
        help="If true, inject /no_think for no-think reason samples. Default false preserves old prompts.",
    )

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
    ap.add_argument("--max_steps", type=int, default=0, help="If > 0, stop after this many optimizer steps.")
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--num_eval_batches", type=int, default=5)

    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_final", type=str2bool, default=True)
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
    ap.add_argument(
        "--save_trainable",
        type=str2bool,
        default=True,
        help="Save small trainable-parameter checkpoint adapter_model.bin.",
    )
    ap.add_argument(
        "--save_full_model",
        type=str2bool,
        default=False,
        help="Also save a full model checkpoint compatible with load_full_state_dict().",
    )
    ap.add_argument(
        "--save_full_every_checkpoint",
        type=str2bool,
        default=False,
        help="If true, save full model at every save_steps. Usually keep false because full checkpoints are large.",
    )
    ap.add_argument(
        "--full_save_max_shard_size",
        type=str,
        default="2GB",
        help="Max shard size for full-model PyTorch checkpoint, e.g. 2GB or 5GB.",
    )

    args = ap.parse_args()

    if args.local_rank >= 0:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    set_seed(int(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)
    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":
        print(
            "[GRPO qwen-prefix env] "
            f"SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX={os.environ.get('SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX', '')} "
            f"SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE={os.environ.get('SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE', '')} "
            f"SMARTCITY_GRPO_DISABLE_LEGACY_SOFT_THINK={os.environ.get('SMARTCITY_GRPO_DISABLE_LEGACY_SOFT_THINK', '')} "
            f"legacy_soft_think_disabled={legacy_soft_think_disabled()} "
            f"SMARTCITY_REWARD_STRICT_ALWAYS_THINK_TAGS={os.environ.get('SMARTCITY_REWARD_STRICT_ALWAYS_THINK_TAGS', '')}",
            flush=True,
        )

    ds_plugin = None
    if args.deepspeed:
        if DeepSpeedPlugin is None:
            raise RuntimeError("accelerate.utils.DeepSpeedPlugin is unavailable; install/upgrade accelerate or omit --deepspeed.")
        ds_config_path = Path(args.deepspeed)
        if not ds_config_path.is_file():
            raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")
        ds_config_path = _smartcity_prepare_deepspeed_config(
            ds_config_path=ds_config_path,
            output_dir=args.output_dir,
            per_device_batch=int(args.batch_size),
        )
        # StageD hardfix: expose actual DeepSpeed ZeRO stage to rollout.py.
        # In ZeRO-2 runs this force-disables ZeRO-3 generation gather and ZeRO-3 leaf setup.
        try:
            _sc_ds_cfg = json.loads(Path(ds_config_path).read_text(encoding="utf-8"))
            _sc_zero_stage = int(((_sc_ds_cfg.get("zero_optimization") or {}).get("stage", 0)) or 0)
        except Exception:
            _sc_zero_stage = -1
        os.environ["SMARTCITY_ACTUAL_ZERO_STAGE"] = str(_sc_zero_stage)
        if _sc_zero_stage != 3:
            os.environ["SMARTCITY_DS3_GATHER_FOR_GENERATION"] = "0"
            os.environ["SMARTCITY_FORCE_ZERO3_GATHER"] = "0"
            os.environ["SMARTCITY_DS3_USE_OFFICIAL_HELPER"] = "0"
            os.environ["SMARTCITY_Z3_LEAF_SUFFIXES"] = ""
        if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":
            print(
                f"[stageD-hardfix] actual_ds_zero_stage={_sc_zero_stage}; "
                f"SMARTCITY_DS3_GATHER_FOR_GENERATION={os.environ.get('SMARTCITY_DS3_GATHER_FOR_GENERATION')}; "
                f"SMARTCITY_FORCE_ZERO3_GATHER={os.environ.get('SMARTCITY_FORCE_ZERO3_GATHER')}; "
                f"SMARTCITY_Z3_LEAF_SUFFIXES={os.environ.get('SMARTCITY_Z3_LEAF_SUFFIXES')!r}",
                flush=True,
            )
        ds_plugin = DeepSpeedPlugin(hf_ds_config=str(ds_config_path))

    accelerator = Accelerator(deepspeed_plugin=ds_plugin)
    is_main = accelerator.is_main_process
    if is_main and args.deepspeed:
        print(f"[info] DeepSpeed config: {Path(args.deepspeed).resolve()}")

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
        print(f"[info] using SFT/warmup ckpt_dir: {ckpt_dir}")

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




    # Load SFT/warmup checkpoint. For output_stageC_warmup/full_2epoch this
    # should directly load model.safetensors and should not try a ZeRO-3 index.
    if is_main:
        print(f"[load-flex] loading SFT checkpoint: {ckpt_dir}", flush=True)
    _smartcity_load_sft_checkpoint(model, str(ckpt_dir), verbose=is_main)
    _smartcity_maybe_load_trainable_adapter(model, verbose=is_main)

    # StageD speed fix:
    # SmartCityLLM constructor may enable decoder gradient checkpointing by default.
    # For GRPO rollout generation we need KV cache to work, so when CLI says
    # --gradient_checkpointing false, explicitly disable it here.
    if not bool(args.gradient_checkpointing):
        try:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model, "decoder") and hasattr(model.decoder, "gradient_checkpointing_disable"):
                model.decoder.gradient_checkpointing_disable()
            if hasattr(model, "encoder") and hasattr(getattr(model.encoder, "model", None), "gradient_checkpointing_disable"):
                model.encoder.model.gradient_checkpointing_disable()
        except Exception as e:
            if is_main:
                print(f"[stageD-speed][warn] gradient_checkpointing_disable failed: {repr(e)}", flush=True)
        try:
            if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
                model.decoder.config.use_cache = True
            if hasattr(model, "config"):
                model.config.use_cache = True
        except Exception:
            pass
        if is_main:
            try:
                dec_gc = getattr(model.decoder, "is_gradient_checkpointing", None)
            except Exception:
                dec_gc = None
            try:
                use_cache = getattr(model.decoder.config, "use_cache", None)
            except Exception:
                use_cache = None
            print(
                f"[stageD-speed] gradient_checkpointing=false -> disabled; "
                f"decoder.is_gradient_checkpointing={dec_gc}; decoder.config.use_cache={use_cache}",
                flush=True,
            )

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

    reason_only_freeze_active = bool(args.use_mixlora and args.freeze_mixlora_non_reason)
    if reason_only_freeze_active:
        freeze_stats = apply_reason_only_mixlora_gradient_mask(
            model,
            num_universal=int(args.mixlora_num_universal),
            num_pred=int(args.mixlora_num_pred),
            num_reason=int(args.mixlora_num_reason),
            freeze_router_hidden=True,
            verbose=is_main,
        )
        if is_main:
            print(f"[reason-only-freeze] post-step restore enabled; weight_decay={float(args.weight_decay)}")
            print(f"[reason-only-freeze] stats={freeze_stats}")
    else:
        if is_main:
            print("[reason-only-freeze] disabled")

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
        _smartcity_load_sft_checkpoint(ref_model, str(ckpt_dir), verbose=False)
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
        metas = [ExampleMeta.from_any(ex) for ex in examples]

        patched = []
        for ex in examples:
            ex2 = dict(ex)
            ex2["answer"] = ""
            ex2["target"] = ""
            ex2["thinking"] = []

            task_name = (ex2.get("task", "") or "").lower()
            mode = infer_reasoning_mode_from_example(ex, default_mode=args.default_reasoning_mode)
            ex2["reasoning_mode"] = mode

            # Freeform evidence mode: keep gold JSON only as reward metadata,
            # but ask the policy to produce Qwen3-style natural-language answers with [E_*] citations.
            if task_name == "reason" and os.environ.get("SMARTCITY_REWARD_FREEFORM_EVIDENCE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                ex2["prompt"] = rewrite_reason_prompt_for_freeform_evidence(ex2.get("prompt", "") or "")

            # StageD freeform mode:
            # The dataset may still contain old JSON-output instructions.
            # In freeform mode, override them at runtime and ask for Qwen3-style
            # natural-language final answers with [E_*] evidence citations.
            if task_name == "reason" and os.environ.get("SMARTCITY_REWARD_FREEFORM_EVIDENCE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                ex2["prompt"] = rewrite_reason_prompt_for_freeform_evidence(ex2.get("prompt", "") or "")

            # Strict tag prompt: this is only an instruction, not assistant prefill.
            # The model still has to generate <think> / </think> itself.
            if task_name == "reason":
                ex2["prompt"] = stageD_strict_tag_instruction(ex2.get("prompt", "") or "", mode)

            # Qwen-prefix continuation route:
            # When SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX=1, DataCollatorSmartCity
            # pre-fills assistant context directly:
            #   think    -> <think>\n
            #   no_think -> <think>\n\n</think>\n\n
            # In that route, do NOT inject legacy /think or /no_think into the user prompt.
            if task_name == "reason" and (not legacy_soft_think_disabled()):
                if mode == "think":
                    ex2["prompt"] = inject_soft_think_in_user(ex2.get("prompt", "") or "")
                elif mode == "no_think" and bool(args.inject_no_think_token):
                    ex2["prompt"] = inject_soft_no_think_in_user(ex2.get("prompt", "") or "")

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

    # DeepSpeed ZeRO-3 dynamic-router safety: set MixLoRA expert containers as leaf modules before prepare().
    if args.deepspeed:
        configure_zero3_leaf_modules_for_mixlora(model, is_main=is_main)

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
    requested_max_steps = int(args.max_steps)
    if requested_max_steps > 0:
        max_train_steps = requested_max_steps
        num_epochs_to_run = max(1, math.ceil(max_train_steps / max(1, num_update_steps_per_epoch)))
    else:
        num_epochs_to_run = int(args.num_epochs)
        max_train_steps = num_update_steps_per_epoch * num_epochs_to_run
    if is_main:
        print(
            "[info] "
            f"num_update_steps_per_epoch={num_update_steps_per_epoch}, "
            f"num_epochs_to_run={num_epochs_to_run}, "
            f"max_train_steps={max_train_steps}"
        )

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
            rr = safe_gather_for_metrics(accelerator, roll.rewards, name='rewards')
            total += float(rr.sum().item())
            count += int(rr.numel())
            if int(args.num_eval_batches) > 0 and (i + 1) >= int(args.num_eval_batches):
                break
        model.train()
        return (total / max(1, count)) if count > 0 else None

    # ------------------ train loop ------------------
    for epoch in range(num_epochs_to_run):
        dl_len = len(dl)
        for batch_idx, batch in enumerate(dl, start=1):
            _stageD_batch_t0 = time.perf_counter()
            _stageD_train_dbg(
                accelerator,
                f"train.batch_start epoch={epoch} batch_idx={batch_idx}/{dl_len} global_step={global_step} "
                f"accum_slot={((batch_idx - 1) % int(args.grad_accum)) + 1}/{int(args.grad_accum)}"
            )

            roll = rollout_and_cache(
                accelerator=accelerator,
                model=model,
                tokenizer=tok,
                batch=batch,
                cfg=cfg,
                ref_model=ref_model,
                micro_batch_logps=(int(args.micro_batch_logps) if int(args.micro_batch_logps) > 0 else None),
            )

            _stageD_train_dbg(
                accelerator,
                f"train.rollout_done batch_idx={batch_idx} completion={tuple(roll.completion_ids.shape)}",
                _stageD_batch_t0,
            )
            _stageD_phase_t0 = time.perf_counter()

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

                _stageD_train_dbg(accelerator, f"new_logps.start batch_idx={batch_idx} mb={st}:{ed}")
                _stageD_mb_t0 = time.perf_counter()

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

                _stageD_train_dbg(
                    accelerator,
                    f"new_logps.done batch_idx={batch_idx} mb={st}:{ed} shape={tuple(new_logps_mb.shape)}",
                    _stageD_mb_t0,
                )
                _stageD_mb_t0 = time.perf_counter()

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

                _stageD_train_dbg(accelerator, f"backward.done batch_idx={batch_idx} mb={st}:{ed}", _stageD_mb_t0)

                loss_weighted_sum += float(loss_mb.detach().item()) * bs_mb
                del new_logps_mb, loss_mb

            if do_step:
                _stageD_train_dbg(accelerator, f"optimizer_section.start batch_idx={batch_idx} global_step={global_step}")
                _stageD_opt_t0 = time.perf_counter()

                reason_only_step_stats: Dict[str, float] = {}
                reason_only_snapshot = None
                if reason_only_freeze_active:
                    unwrapped_for_freeze = accelerator.unwrap_model(model)

                    # These diagnostics copy many trainable tensors to CPU.
                    # For smoke tests, disable them by default to avoid very slow 0/1 steps.
                    # Enable with SMARTCITY_REASON_ONLY_GRAD_STATS=1 or SMARTCITY_REASON_ONLY_DELTA_STATS=1.
                    if os.environ.get("SMARTCITY_REASON_ONLY_GRAD_STATS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        grad_stats = reason_only_grad_norms(
                            unwrapped_for_freeze,
                            num_universal=int(args.mixlora_num_universal),
                            num_pred=int(args.mixlora_num_pred),
                            num_reason=int(args.mixlora_num_reason),
                        )
                        reason_only_step_stats.update({f"reason_only/{k}": v for k, v in grad_stats.items()})

                    if os.environ.get("SMARTCITY_REASON_ONLY_DELTA_STATS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        reason_only_snapshot = capture_reason_only_values(
                            unwrapped_for_freeze,
                            num_universal=int(args.mixlora_num_universal),
                            num_pred=int(args.mixlora_num_pred),
                            num_reason=int(args.mixlora_num_reason),
                        )

                grad_norm = accelerator.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                _stageD_train_dbg(accelerator, "clip_grad.done", _stageD_opt_t0)
                _stageD_opt_t0 = time.perf_counter()

                optim.step()
                _stageD_train_dbg(accelerator, "optim.step.done", _stageD_opt_t0)
                _stageD_opt_t0 = time.perf_counter()
                if reason_only_freeze_active:
                    unwrapped_for_freeze = accelerator.unwrap_model(model)

                    # With weight_decay=0.0 and gradient hooks, U/P slices should not update.
                    # The restore pass can be expensive or shape-incompatible under ZeRO-3,
                    # so keep it disabled during smoke unless explicitly requested.
                    if os.environ.get("SMARTCITY_REASON_ONLY_RESTORE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        restore_reason_only_frozen_slices(unwrapped_for_freeze)

                    if reason_only_snapshot is not None:
                        delta_stats = reason_only_delta_norms(reason_only_snapshot)
                        reason_only_step_stats.update({f"reason_only/{k}": v for k, v in delta_stats.items()})
                optim.zero_grad(set_to_none=True)
                _stageD_train_dbg(accelerator, "zero_grad.done", _stageD_opt_t0)

                global_step += 1
                if is_main:
                    progress_bar.update(1)

                with torch.no_grad():
                    r_all = safe_gather_for_metrics(accelerator, roll.rewards, name='rewards')
                    r_mean = float(r_all.mean().item())
                    r_std = float(r_all.std(unbiased=False).item())
                    r_min = float(r_all.min().item())
                    r_max = float(r_all.max().item())

                    loss_value = loss_weighted_sum / max(1.0, float(Btot))

                    len_stop = safe_gather_for_metrics(accelerator, roll.stop_mask.sum(dim=1).float(), name='len_stop')
                    len_loss = safe_gather_for_metrics(accelerator, roll.loss_mask.sum(dim=1).float(), name='len_loss')
                    len_stop_mean = float(len_stop.mean().item())
                    len_loss_mean = float(len_loss.mean().item())

                    lr = float(optim.param_groups[0]["lr"])
                    gnorm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                if is_main and (global_step % int(args.log_steps) == 0):
                    progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "r": f"{r_mean:.4f}"})
                    if reason_only_step_stats:
                        print(
                            "[reason-only-freeze] "
                            f"step={global_step} "
                            f"grad(U/P/R)="
                            f"{reason_only_step_stats.get('reason_only/grad_norm_U', 0.0):.3e}/"
                            f"{reason_only_step_stats.get('reason_only/grad_norm_P', 0.0):.3e}/"
                            f"{reason_only_step_stats.get('reason_only/grad_norm_R', 0.0):.3e} "
                            f"delta(U/P/R)="
                            f"{reason_only_step_stats.get('reason_only/delta_norm_U', 0.0):.3e}/"
                            f"{reason_only_step_stats.get('reason_only/delta_norm_P', 0.0):.3e}/"
                            f"{reason_only_step_stats.get('reason_only/delta_norm_R', 0.0):.3e}"
                        )

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
                if reason_only_step_stats:
                    log_dict.update(reason_only_step_stats)
                wlog(log_dict, step=global_step)

                # Console reward metrics are needed when wandb is disabled.
                if is_main and (global_step % int(args.log_steps) == 0):
                    metric_keys = [
                        "train/loss",
                        "train/reward_mean",
                        "train/reward_std",
                        "train/reward_min",
                        "train/reward_max",
                        "reward/freeform_used_rate",
                        "reward/freeform_format_mean",
                        "reward/answer_score_mean",
                        "reward/citation_mean",
                        "reward/supporting_fact_mean",
                        "reward/missing_evidence_handling_mean",
                        "reward/citation_valid_rate",
                        "reward/key_evidence_recall",
                        "reward/source_type_coverage",
                        "reward/citation_count_mean",
                        "reward/invalid_link_penalty_mean",
                        "reward/hallucinated_evidence_penalty_mean",
                        "reward/future_evidence_penalty_mean",
                        "reward/overcite_penalty_mean",
                        "reward/think_close_rate",
                        "reward/miss_think_close_rate",
                    ]
                    parts = []
                    for k in metric_keys:
                        if k in log_dict:
                            try:
                                parts.append(f"{k}={float(log_dict[k]):.4f}")
                            except Exception:
                                parts.append(f"{k}={log_dict[k]}")
                    if parts:
                        print(f"[metrics] step={global_step} " + " ".join(parts), flush=True)

                if args.log_text_steps and (global_step % int(args.log_text_steps) == 0) and roll.texts and is_main:
                    import wandb
                    wlog({"sample/text": wandb.Html(f"<pre>{roll.texts[0]}</pre>")}, step=global_step)

                if int(args.save_steps) > 0 and (global_step % int(args.save_steps) == 0):
                    if bool(args.save_trainable):
                        save_checkpoint_trainable(
                            args.output_dir,
                            global_step,
                            model,
                            tok,
                            accelerator,
                            int(args.save_total_limit),
                        )

                    # Full checkpoints are huge, so by default only save full at final.
                    # If explicitly requested, save full at intermediate checkpoints too.
                    if bool(args.save_full_model) and bool(args.save_full_every_checkpoint):
                        save_checkpoint_full(
                            args.output_dir,
                            global_step,
                            model,
                            tok,
                            accelerator,
                            max_shard_size=str(args.full_save_max_shard_size),
                        )

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

    if bool(args.save_final):
        if bool(args.save_trainable):
            save_checkpoint_trainable(
                args.output_dir,
                global_step,
                model,
                tok,
                accelerator,
                int(args.save_total_limit),
            )

        if bool(args.save_full_model):
            save_checkpoint_full(
                args.output_dir,
                global_step,
                model,
                tok,
                accelerator,
                max_shard_size=str(args.full_save_max_shard_size),
            )
    elif is_main:
        print("[save] final checkpoint disabled (--save_final false)")

    if is_main:
        print(f"[done] global_step={global_step}")

    if wandb_run is not None and is_main:
        wandb_run.finish()


if __name__ == "__main__":
    main()
