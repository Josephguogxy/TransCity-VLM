from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import os
import time

import torch
import torch.distributed as dist
from accelerate import Accelerator

from .config import GRPOConfig
from .logprobs import make_stop_mask, get_per_token_logps_smartcity, trim_right_padded_3d
from .rewards import ExampleMeta, compute_rewards

def _stageD_rollout_dbg(accelerator: Accelerator, msg: str, t0=None) -> None:
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



_TRUE_ENV = {"1", "true", "yes", "y", "on"}
_ZERO3_GEN_GATHER_LOGGED = False
_ZERO3_GEN_FALLBACK_LOGGED = False


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_ENV


def _dist_rank() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or "0")


def _rank0_print(msg: str) -> None:
    if _dist_rank() == 0:
        print(msg, flush=True)


def _zero_stage(model, accelerator: Accelerator) -> int:
    try:
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        stage = getattr(plugin, "zero_stage", None)
        if stage is not None:
            return int(stage)

        cfg = getattr(plugin, "deepspeed_config", None) or {}
        stage = (cfg.get("zero_optimization", {}) or {}).get("stage", 0)
        if stage is not None:
            return int(stage)
    except Exception:
        pass

    try:
        if hasattr(model, "zero_optimization_stage"):
            return int(model.zero_optimization_stage())
    except Exception:
        pass

    try:
        cfg = getattr(model, "config", {}) or {}
        return int(((cfg.get("zero_optimization", {}) or {}).get("stage", 0)) or 0)
    except Exception:
        return 0


@contextmanager
def zero3_gather_for_generation(model, accelerator: Accelerator):
    """
    ZeRO-3 rollout acceleration.

    Plain ZeRO-3 can gather/release sharded parameters every token/layer during
    autoregressive rollout. This context gathers policy weights once for generation.

    Disable with:
      SMARTCITY_DS3_GATHER_FOR_GENERATION=0
    """
    global _ZERO3_GEN_GATHER_LOGGED, _ZERO3_GEN_FALLBACK_LOGGED

    try:
        unwrapped = accelerator.unwrap_model(model)
    except Exception:
        unwrapped = model

    # StageD hardfix: never run ZeRO-3 gather when actual stage is not 3.
    actual_zero = os.environ.get("SMARTCITY_ACTUAL_ZERO_STAGE", "").strip()
    if actual_zero and actual_zero != "3":
        if _env_true("SMARTCITY_DS3_GATHER_DEBUG", "0"):
            _rank0_print(f"[stageD] skip ZeRO-3 generation gather because actual_ds_zero_stage={actual_zero}")
        yield unwrapped
        return

    if not _env_true("SMARTCITY_DS3_GATHER_FOR_GENERATION", "0"):
        yield unwrapped
        return

    if _zero_stage(model, accelerator) != 3:
        yield unwrapped
        return

    # Preferred DeepSpeed helper.
    try:
        from deepspeed.runtime.zero.stage3 import unwrap_model_for_generation

        if _dist_rank() == 0 and not _ZERO3_GEN_GATHER_LOGGED:
            _rank0_print("[stageD] using DeepSpeed unwrap_model_for_generation for ZeRO-3 generation")
            _ZERO3_GEN_GATHER_LOGGED = True

        with unwrap_model_for_generation(model):
            try:
                yield accelerator.unwrap_model(model)
            except Exception:
                yield unwrapped
        return

    except Exception as e:
        if _dist_rank() == 0 and (_env_true("SMARTCITY_DS3_GATHER_DEBUG", "0") or not _ZERO3_GEN_FALLBACK_LOGGED):
            _rank0_print(f"[stageD][warn] unwrap_model_for_generation failed: {repr(e)}; trying GatheredParameters")
            _ZERO3_GEN_FALLBACK_LOGGED = True

    # Fallback path.
    try:
        import deepspeed

        params = list(unwrapped.parameters())

        if _dist_rank() == 0 and not _ZERO3_GEN_GATHER_LOGGED:
            _rank0_print("[stageD] using deepspeed.zero.GatheredParameters for ZeRO-3 generation")
            _ZERO3_GEN_GATHER_LOGGED = True

        with deepspeed.zero.GatheredParameters(params):
            yield unwrapped

    except Exception as e:
        if _dist_rank() == 0:
            _rank0_print(f"[stageD][warn] GatheredParameters failed: {repr(e)}; using normal sharded generation")
        yield unwrapped

@dataclass
class RolloutBatch:
    rep: Dict[str, torch.Tensor]

    chunk_embeds: torch.Tensor
    chunk_mask: torch.Tensor
    image_embeds: torch.Tensor
    image_mask: torch.Tensor

    completion_ids: torch.LongTensor

    # IMPORTANT:
    # - stop_mask: for correct conditioning/logp computation (includes think tokens)
    # - loss_mask: for RL loss (by default only answer-after-</think>)
    stop_mask: torch.LongTensor
    loss_mask: torch.LongTensor

    old_per_token_logps: Optional[torch.Tensor]
    ref_per_token_logps: Optional[torch.Tensor]

    rewards: torch.Tensor
    advantages: torch.Tensor
    metas_rep: List[ExampleMeta]
    reward_info: Optional[Dict[str, float]] = None   # ✅ 新增
    texts: Optional[List[str]] = None                # ✅ 可选：你要 log 文本就加


def _repeat_interleave_any(x, repeats: int):
    if torch.is_tensor(x):
        return x.repeat_interleave(repeats, dim=0)
    return sum(([a] * repeats for a in x), [])


def _build_loss_mask_answer_only(
    completion_ids: torch.LongTensor,
    stop_mask: torch.LongTensor,
    think_close_id: Optional[int],
) -> torch.LongTensor:
    """
    loss_mask = stop_mask but zero-out tokens up to and including first </think>.
    NOTE: We do NOT change stop_mask used in attention/logp, only loss masking.
    """
    if think_close_id is None or not isinstance(think_close_id, int) or think_close_id < 0:
        return stop_mask

    B, T = completion_ids.shape
    loss_mask = stop_mask.clone()

    for b in range(B):
        valid = stop_mask[b].bool()
        hits = torch.nonzero((completion_ids[b] == int(think_close_id)) & valid, as_tuple=False)
        if hits.numel() > 0:
            j = int(hits[0].item())
            loss_mask[b, : j + 1] = 0

    return loss_mask


@torch.no_grad()
def rollout_and_cache(
    accelerator: Accelerator,
    model,
    tokenizer,
    batch: Dict,
    cfg: GRPOConfig,
    *,
    ref_model=None,
    micro_batch_logps: Optional[int] = None,
) -> RolloutBatch:
    cfg.validate()
    m = accelerator.unwrap_model(model)
    m.eval()

    # ---- metas ----
    metas_in = batch.get("meta", None)
    if metas_in is None:
        B0 = batch["prefix_ids"].size(0)
        metas = [ExampleMeta(task="", target="", answer="") for _ in range(B0)]
    else:
        metas = []
        for m0 in metas_in:
            metas.append(ExampleMeta.from_any(m0))

    metas_rep = _repeat_interleave_any(metas, cfg.num_generations)

    # ---- 1) encode modalities ----
    dec_emb = m.decoder.get_input_embeddings()
    dev = dec_emb.weight.device
    dtype = dec_emb.weight.dtype
    B = int(batch["prefix_ids"].size(0))
    _stageD_rollout_t0 = time.perf_counter()
    _stageD_phase_t0 = time.perf_counter()
    _stageD_rollout_dbg(
        accelerator,
        f"rollout.start B={B} G={cfg.num_generations} max_new={cfg.max_new_tokens} beta_kl={cfg.beta_kl} "
        f"prefix_shape={tuple(batch['prefix_ids'].shape)} suffix_shape={tuple(batch['suffix_ids'].shape)}"
    )

    image_embeds = None
    image_mask = None

    if hasattr(m, "encode_modalities"):
        out = m.encode_modalities(
            chunk_txts_all=batch.get("chunk_txts", None),
            pixel_values=batch.get("pixel_values", None),
            n_images=batch.get("n_images", None),
            batch_size=B,
            device=dev,
            dtype=dtype,
        )

        # 兼容不同返回格式：
        # 1) (chunk_embeds, chunk_mask, image_embeds, image_mask, extra)
        # 2) (chunk_embeds, chunk_mask, extra)
        if isinstance(out, (list, tuple)) and len(out) >= 4:
            chunk_embeds, chunk_mask, image_embeds, image_mask = out[:4]
        else:
            chunk_embeds, chunk_mask = out[:2]
    else:
        chunk_embeds, chunk_mask, _ = m.encode_chunks(
            chunk_txts_all=batch.get("chunk_txts", None),
            batch_size=B,
            device=dev,
            dtype=dtype,
        )

    # ✅ 确保 image_* 一定有定义（纯文本任务就是空）
    if image_embeds is None or image_mask is None:
        image_embeds = chunk_embeds.new_zeros((B, 0, chunk_embeds.size(-1)))
        image_mask = torch.zeros((B, 0), device=dev, dtype=torch.long)

    # ✅ 关键：先 trim 掉 chunk/image 的右侧 padding，避免后面 repeat 放大无效 token
    chunk_embeds, chunk_mask = trim_right_padded_3d(chunk_embeds, chunk_mask)
    image_embeds, image_mask = trim_right_padded_3d(image_embeds, image_mask)

    _stageD_rollout_dbg(
        accelerator,
        f"encode_modalities.done chunk={tuple(chunk_embeds.shape)} image={tuple(image_embeds.shape)} "
        f"chunk_tokens={int(chunk_mask.sum().item()) if torch.is_tensor(chunk_mask) else -1} "
        f"image_tokens={int(image_mask.sum().item()) if torch.is_tensor(image_mask) else -1}",
        _stageD_phase_t0,
    )
    _stageD_phase_t0 = time.perf_counter()

    # ---- 2) repeat prompts (B -> B*G) ----
    keys = ["prefix_ids", "prefix_mask", "suffix_ids", "suffix_mask", "task_ids"]
    rep: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k in batch and batch[k] is not None:
            rep[k] = _repeat_interleave_any(batch[k].to(dev), cfg.num_generations)

    chunk_embeds_rep = chunk_embeds.repeat_interleave(cfg.num_generations, dim=0).to(dev, dtype=dtype)
    chunk_mask_rep   = chunk_mask.repeat_interleave(cfg.num_generations, dim=0).to(dev)
    image_embeds_rep = image_embeds.repeat_interleave(cfg.num_generations, dim=0).to(dev, dtype=dtype)
    image_mask_rep   = image_mask.repeat_interleave(cfg.num_generations, dim=0).to(dev)

    # ---- 3) generate ----
    stop_ids = cfg.stop_token_ids

    t_gen0 = time.perf_counter()
    with zero3_gather_for_generation(model, accelerator) as m_gen:
        m_gen.eval()
        if hasattr(m_gen, "generate_with_modalities_cached"):
            completion_ids = m_gen.generate_with_modalities_cached(
                prefix_ids=rep["prefix_ids"],
                prefix_mask=rep["prefix_mask"],
                suffix_ids=rep["suffix_ids"],
                suffix_mask=rep["suffix_mask"],
                chunk_embeds=chunk_embeds_rep,
                chunk_mask=chunk_mask_rep,
                image_embeds=image_embeds_rep,
                image_mask=image_mask_rep,
                task_ids=rep.get("task_ids", None),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=cfg.do_sample,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stop_token_ids=stop_ids,
            )
        else:
            completion_ids = m_gen.generate_cached(
                prefix_ids=rep["prefix_ids"],
                prefix_mask=rep["prefix_mask"],
                suffix_ids=rep["suffix_ids"],
                suffix_mask=rep["suffix_mask"],
                chunk_embeds=chunk_embeds_rep,
                chunk_mask=chunk_mask_rep,
                task_ids=rep.get("task_ids", None),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=cfg.do_sample,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stop_token_ids=stop_ids,
            )

    # StageD safety:
    # generation runs under no_grad / eval path. Make completion_ids a normal
    # detached tensor before it is reused by logprob/backward code.
    completion_ids = completion_ids.detach().clone()

    if _env_true("SMARTCITY_TIMING", "0") and _dist_rank() == int(os.environ.get("SMARTCITY_TIMING_RANK", "0") or "0"):
        print(
            f"[stageD-timing][rank{_dist_rank()}] generate.done completion={tuple(completion_ids.shape)} "
            f"| dt={time.perf_counter() - t_gen0:.2f}s",
            flush=True,
        )

    stop_mask = make_stop_mask(
        completion_ids,
        eos_id=tokenizer.eos_token_id,
        stop_token_ids=stop_ids,
    )

    # loss mask: answer-only after </think>
    loss_mask = stop_mask
    if getattr(cfg, "loss_on_answer_only", False):
        close_id = tokenizer.convert_tokens_to_ids("</think>")
        close_id = int(close_id) if isinstance(close_id, int) and close_id >= 0 else None
        loss_mask = _build_loss_mask_answer_only(completion_ids, stop_mask, close_id)

    # ---- 4) decode -> reward ----
    texts = tokenizer.batch_decode(
        [ids[msk.bool()].tolist() for ids, msk in zip(completion_ids, stop_mask)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reward_list, reward_info = compute_rewards(texts, metas_rep, return_info=True)
    rewards = torch.tensor(reward_list, device=dev, dtype=torch.float32)

    _stageD_rollout_dbg(
        accelerator,
        f"decode_reward.done n_texts={len(texts)} reward_mean={float(rewards.mean().item()) if rewards.numel() else 0.0:.4f}",
        _stageD_phase_t0,
    )
    _stageD_phase_t0 = time.perf_counter()

    # ---- group advantage ----
    G = int(cfg.num_generations)
    rewards_g = rewards.view(-1, G)
    mean = rewards_g.mean(dim=1, keepdim=True)
    std = rewards_g.std(dim=1, keepdim=True, unbiased=False)
    advantages = (rewards_g - mean) / (std + float(cfg.adv_eps))
    advantages = advantages.reshape(-1).to(dev)

    # ---- 5) old logps + optional ref logps ----
    # IMPORTANT: logps must be computed with stop_mask (correct conditioning).
    _stageD_rollout_dbg(accelerator, "old_logps.start")
    _stageD_phase_t0 = time.perf_counter()

    old_logps = get_per_token_logps_smartcity(
        m,
        prefix_ids=rep["prefix_ids"], prefix_mask=rep["prefix_mask"],
        suffix_ids=rep["suffix_ids"], suffix_mask=rep["suffix_mask"],
        chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
        image_embeds=image_embeds_rep, image_mask=image_mask_rep,
        completion_ids=completion_ids, completion_mask=stop_mask,
        task_ids=rep.get("task_ids", None),
        temperature=cfg.temperature,
        micro_batch_size=micro_batch_logps,
    )

    _stageD_rollout_dbg(
        accelerator,
        f"old_logps.done shape={tuple(old_logps.shape) if torch.is_tensor(old_logps) else None}",
        _stageD_phase_t0,
    )
    _stageD_phase_t0 = time.perf_counter()

    ref_logps = None
    if cfg.beta_kl != 0.0 and ref_model is not None:
        _stageD_rollout_dbg(accelerator, "ref_logps.start")
        _stageD_phase_t0 = time.perf_counter()
        rm = accelerator.unwrap_model(ref_model)
        rm.eval()
        ref_logps = get_per_token_logps_smartcity(
            rm,
            prefix_ids=rep["prefix_ids"], prefix_mask=rep["prefix_mask"],
            suffix_ids=rep["suffix_ids"], suffix_mask=rep["suffix_mask"],
            chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
            image_embeds=image_embeds_rep, image_mask=image_mask_rep,
            completion_ids=completion_ids, completion_mask=stop_mask,
            task_ids=rep.get("task_ids", None),
            temperature=cfg.temperature,
            micro_batch_size=micro_batch_logps,
        )

    if ref_logps is not None:
        _stageD_rollout_dbg(
            accelerator,
            f"ref_logps.done shape={tuple(ref_logps.shape) if torch.is_tensor(ref_logps) else None}",
            _stageD_phase_t0,
        )

    _stageD_rollout_dbg(accelerator, "rollout.done", _stageD_rollout_t0)

    return RolloutBatch(
        rep=rep,
        chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
        image_embeds=image_embeds_rep, image_mask=image_mask_rep,
        completion_ids=completion_ids,
        stop_mask=stop_mask,
        loss_mask=loss_mask,
        old_per_token_logps=old_logps,
        ref_per_token_logps=ref_logps,
        rewards=rewards,
        advantages=advantages,
        metas_rep=metas_rep,
        reward_info=reward_info,   # ✅ 新增
        texts=texts,               # ✅ 如果加了 texts 字段
    )


# ===== SmartCity StageD ZeRO-3 generation gather override v2 =====
# This intentionally redefines zero3_gather_for_generation at module load time.
# It overrides any earlier/incomplete implementation in this file.

from contextlib import contextmanager as _smartcity_contextmanager
import os as _smartcity_os
import time as _smartcity_time


def _smartcity_env_true(name: str, default: str = "0") -> bool:
    return _smartcity_os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _smartcity_dist_rank() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return int(_dist.get_rank())
    except Exception:
        pass

    for key in ("RANK", "LOCAL_RANK"):
        try:
            return int(_smartcity_os.environ.get(key, "0") or "0")
        except Exception:
            pass
    return 0


def _smartcity_timing_enabled() -> bool:
    if not _smartcity_env_true("SMARTCITY_TIMING", "0"):
        return False
    if _smartcity_env_true("SMARTCITY_TIMING_ALL_RANKS", "0"):
        return True
    return _smartcity_dist_rank() == int(_smartcity_os.environ.get("SMARTCITY_TIMING_RANK", "0") or "0")


def _smartcity_rank0_print(msg: str) -> None:
    if _smartcity_dist_rank() == 0:
        print(msg, flush=True)


def _smartcity_zero_stage(model, accelerator) -> int:
    # Explicit override for this debug path.
    if _smartcity_env_true("SMARTCITY_FORCE_ZERO3_GATHER", "0"):
        return 3

    # Accelerate plugin path.
    try:
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        stage = getattr(plugin, "zero_stage", None)
        if stage is not None:
            return int(stage)

        cfg = getattr(plugin, "deepspeed_config", None) or {}
        stage = (cfg.get("zero_optimization", {}) or {}).get("stage", None)
        if stage is not None:
            return int(stage)
    except Exception:
        pass

    # DeepSpeed engine path.
    try:
        if hasattr(model, "zero_optimization_stage"):
            return int(model.zero_optimization_stage())
    except Exception:
        pass

    # ZeRO-3 optimizer has parameter_offload.
    try:
        opt = getattr(model, "optimizer", None)
        if opt is not None and hasattr(opt, "parameter_offload"):
            return 3
    except Exception:
        pass

    try:
        cfg = getattr(model, "config", {}) or {}
        return int(((cfg.get("zero_optimization", {}) or {}).get("stage", 0)) or 0)
    except Exception:
        return 0


def _iter_zero3_module_params(module):
    params = []

    try:
        params.extend([p for _, p in module.named_parameters(recurse=True)])
    except Exception:
        try:
            params.extend(list(module.parameters()))
        except Exception:
            pass

    # DeepSpeed external params, if present.
    try:
        ext = module.ds_external_parameters()
        for item in ext:
            if isinstance(item, tuple) and len(item) == 2:
                params.append(item[1])
            else:
                params.append(item)
    except Exception:
        pass

    out = []
    seen = set()
    for param in params:
        if id(param) not in seen:
            out.append(param)
            seen.add(id(param))
    return out


def _zero3_offload_manager(ds_engine):
    opt = getattr(ds_engine, "optimizer", None)
    if opt is None:
        return None
    return getattr(opt, "parameter_offload", opt)


def _remove_zero3_hooks_for_generation(ds_engine) -> None:
    """
    TRL-style ZeRO-3 generation fast path.

    Important details:
      1. GatheredParameters gathers policy weights once.
      2. ZeRO-3 forward/backward hooks are removed during autoregressive generation.
      3. ds_active_sub_modules is cleared before hook removal.
    """
    offload = _zero3_offload_manager(ds_engine)
    if offload is None:
        return

    module = getattr(offload, "module", None)
    if module is not None:
        for param in _iter_zero3_module_params(module):
            active = getattr(param, "ds_active_sub_modules", None)
            if active is not None:
                try:
                    active.clear()
                except Exception:
                    pass

    # Newer DeepSpeed has _remove_hooks.
    if hasattr(offload, "_remove_hooks"):
        try:
            offload._remove_hooks(getattr(offload, "forward_hooks", []))
            offload._remove_hooks(getattr(offload, "backward_hooks", []))
            return
        except Exception:
            pass

    # Fallback manual hook removal.
    for hook in list(getattr(offload, "forward_hooks", []) or []):
        try:
            hook.remove()
        except Exception:
            pass

    for hook in list(getattr(offload, "backward_hooks", []) or []):
        try:
            hook.remove()
        except Exception:
            pass

    try:
        offload.forward_hooks = []
    except Exception:
        pass
    try:
        offload.backward_hooks = []
    except Exception:
        pass


def _add_zero3_hooks_after_generation(ds_engine) -> None:
    offload = _zero3_offload_manager(ds_engine)
    if offload is None:
        return

    module = getattr(offload, "module", None)
    if module is None:
        return

    if hasattr(offload, "_register_deepspeed_module"):
        offload._register_deepspeed_module(module)
    elif hasattr(offload, "_register_hooks_recursively"):
        offload._register_hooks_recursively(module)


@_smartcity_contextmanager
def zero3_gather_for_generation(model, accelerator):
    """
    SmartCity Stage D ZeRO-3 rollout acceleration.

    This implements the same idea as TRL ds3_gather_for_generation=True:
    gather ZeRO-3 sharded policy weights once before autoregressive rollout,
    remove ZeRO hooks during generation, then restore hooks after generation.

    Env:
      SMARTCITY_DS3_GATHER_FOR_GENERATION=1   enable, default off
      SMARTCITY_FORCE_ZERO3_GATHER=1           force path even if stage detection fails
      SMARTCITY_DS3_USE_OFFICIAL_HELPER=1      use DeepSpeed official helper instead
      SMARTCITY_DS3_GATHER_DEBUG=1             print fallback warnings
    """
    try:
        unwrapped = accelerator.unwrap_model(model)
    except Exception:
        unwrapped = model

    # StageD hardfix v2: never run ZeRO-3 gather when actual stage is not 3.
    actual_zero = _smartcity_os.environ.get("SMARTCITY_ACTUAL_ZERO_STAGE", "").strip()
    if actual_zero and actual_zero != "3":
        if _smartcity_env_true("SMARTCITY_DS3_GATHER_DEBUG", "0"):
            _smartcity_rank0_print(f"[stageD] skip ZeRO-3 generation gather because actual_ds_zero_stage={actual_zero}")
        yield unwrapped
        return

    if not _smartcity_env_true("SMARTCITY_DS3_GATHER_FOR_GENERATION", "0"):
        yield unwrapped
        return

    stage = _smartcity_zero_stage(model, accelerator)
    if stage != 3:
        if _smartcity_env_true("SMARTCITY_DS3_GATHER_DEBUG", "0"):
            _smartcity_rank0_print(f"[stageD][warn] skip ZeRO-3 generation gather because detected zero_stage={stage}")
        yield unwrapped
        return

    # Optional exact DeepSpeed helper path.
    if _smartcity_env_true("SMARTCITY_DS3_USE_OFFICIAL_HELPER", "0"):
        try:
            from deepspeed.runtime.zero.stage3 import unwrap_model_for_generation

            _smartcity_rank0_print("[stageD] ZeRO-3 generation: using DeepSpeed unwrap_model_for_generation")
            with unwrap_model_for_generation(model):
                try:
                    yield accelerator.unwrap_model(model)
                except Exception:
                    yield unwrapped
            return
        except Exception as e:
            _smartcity_rank0_print(
                f"[stageD][warn] official unwrap_model_for_generation failed: {repr(e)}; "
                "falling back to TRL-style GatheredParameters path"
            )

    # Recommended default path.
    try:
        import deepspeed

        params = list(model.parameters())
        _smartcity_rank0_print("[stageD] ZeRO-3 generation: GatheredParameters + remove ZeRO hooks once for rollout")

        with deepspeed.zero.GatheredParameters(params):
            _remove_zero3_hooks_for_generation(model)
            try:
                try:
                    yield accelerator.unwrap_model(model)
                except Exception:
                    yield unwrapped
            finally:
                _add_zero3_hooks_after_generation(model)
        return

    except Exception as e:
        _smartcity_rank0_print(
            f"[stageD][warn] ZeRO-3 gather-for-generation failed: {repr(e)}; "
            "using normal sharded generation"
        )
        yield unwrapped
