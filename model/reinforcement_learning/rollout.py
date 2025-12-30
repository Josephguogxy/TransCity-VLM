# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator

from .config import GRPOConfig
from .logprobs import make_stop_mask, get_per_token_logps_smartcity
from .rewards import ExampleMeta, compute_rewards


@dataclass
class RolloutBatch:
    rep: Dict[str, torch.Tensor]

    chunk_embeds: torch.Tensor
    chunk_mask: torch.Tensor
    image_embeds: torch.Tensor
    image_mask: torch.Tensor

    completion_ids: torch.LongTensor
    completion_mask: torch.LongTensor

    old_per_token_logps: Optional[torch.Tensor]
    ref_per_token_logps: Optional[torch.Tensor]

    rewards: torch.Tensor
    advantages: torch.Tensor
    metas_rep: List[ExampleMeta]


def _repeat_interleave_any(x, repeats: int):
    if torch.is_tensor(x):
        return x.repeat_interleave(repeats, dim=0)
    return sum(([a] * repeats for a in x), [])


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

    metas_in = batch.get("meta", None)
    if metas_in is None:
        B0 = batch["prefix_ids"].size(0)
        metas = [ExampleMeta(task="", target="", answer="") for _ in range(B0)]
    else:
        metas = []
        for m0 in metas_in:
            if isinstance(m0, ExampleMeta):
                metas.append(m0)
            elif isinstance(m0, dict):
                metas.append(
                    ExampleMeta(
                        task=str(m0.get("task", "") or ""),
                        target=str(m0.get("target", "") or ""),
                        answer=str(m0.get("answer", "") or ""),
                    )
                )
            else:
                metas.append(ExampleMeta(task="", target="", answer=""))

    metas_rep = _repeat_interleave_any(metas, cfg.num_generations)

    # ---- 1) encode modalities ----
    dec_emb = m.decoder.get_input_embeddings()
    dev = dec_emb.weight.device
    dtype = dec_emb.weight.dtype
    B = int(batch["prefix_ids"].size(0))

    if hasattr(m, "encode_modalities"):
        chunk_embeds, chunk_mask, image_embeds, image_mask, _ = m.encode_modalities(
            chunk_txts_all=batch.get("chunk_txts", None),
            pixel_values=batch.get("pixel_values", None),
            n_images=batch.get("n_images", None),
            batch_size=B,
            device=dev,
            dtype=dtype,
        )
    else:
        chunk_embeds, chunk_mask, _ = m.encode_chunks(
            chunk_txts_all=batch.get("chunk_txts", None),
            batch_size=B,
            device=dev,
            dtype=dtype,
        )
        image_embeds = chunk_embeds.new_zeros((B, 0, chunk_embeds.size(-1)))
        image_mask = torch.zeros((B, 0), device=dev, dtype=torch.long)

    # ---- 2) repeat prompts (B -> B*G) ----
    keys = ["prefix_ids", "prefix_mask", "suffix_ids", "suffix_mask", "task_ids"]
    rep: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k in batch and batch[k] is not None:
            rep[k] = _repeat_interleave_any(batch[k].to(dev), cfg.num_generations)

    chunk_embeds_rep = chunk_embeds.repeat_interleave(cfg.num_generations, dim=0).to(dev, dtype=dtype)
    chunk_mask_rep = chunk_mask.repeat_interleave(cfg.num_generations, dim=0).to(dev)
    image_embeds_rep = image_embeds.repeat_interleave(cfg.num_generations, dim=0).to(dev, dtype=dtype)
    image_mask_rep = image_mask.repeat_interleave(cfg.num_generations, dim=0).to(dev)

    # ---- 3) generate ----
    stop_ids = cfg.stop_token_ids

    if hasattr(m, "generate_with_modalities_cached"):
        completion_ids = m.generate_with_modalities_cached(
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
        completion_ids = m.generate_cached(
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

    completion_mask = make_stop_mask(
        completion_ids,
        eos_id=tokenizer.eos_token_id,
        stop_token_ids=stop_ids,
    )

    # ---- 4) decode -> reward ----
    texts = tokenizer.batch_decode(
        [ids[msk.bool()].tolist() for ids, msk in zip(completion_ids, completion_mask)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reward_list = compute_rewards(texts, metas_rep)

    # Use dev (decoder embedding device) to avoid device mismatches
    rewards = torch.tensor(reward_list, device=dev, dtype=torch.float32)

    # ---- group advantage ----
    G = int(cfg.num_generations)
    rewards_g = rewards.view(-1, G)
    mean = rewards_g.mean(dim=1, keepdim=True)
    std = rewards_g.std(dim=1, keepdim=True, unbiased=False)
    advantages = (rewards_g - mean) / (std + float(cfg.adv_eps))
    advantages = advantages.reshape(-1).to(dev)

    # ---- 5) old logps + optional ref logps ----
    old_logps = get_per_token_logps_smartcity(
        m,
        prefix_ids=rep["prefix_ids"], prefix_mask=rep["prefix_mask"],
        suffix_ids=rep["suffix_ids"], suffix_mask=rep["suffix_mask"],
        chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
        image_embeds=image_embeds_rep, image_mask=image_mask_rep,
        completion_ids=completion_ids, completion_mask=completion_mask,
        task_ids=rep.get("task_ids", None),
        temperature=cfg.temperature,
        micro_batch_size=micro_batch_logps,
    )

    ref_logps = None
    if cfg.beta_kl != 0.0 and ref_model is not None:
        rm = accelerator.unwrap_model(ref_model)
        rm.eval()
        ref_logps = get_per_token_logps_smartcity(
            rm,
            prefix_ids=rep["prefix_ids"], prefix_mask=rep["prefix_mask"],
            suffix_ids=rep["suffix_ids"], suffix_mask=rep["suffix_mask"],
            chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
            image_embeds=image_embeds_rep, image_mask=image_mask_rep,
            completion_ids=completion_ids, completion_mask=completion_mask,
            task_ids=rep.get("task_ids", None),
            temperature=cfg.temperature,
            micro_batch_size=micro_batch_logps,
        )

    return RolloutBatch(
        rep=rep,
        chunk_embeds=chunk_embeds_rep, chunk_mask=chunk_mask_rep,
        image_embeds=image_embeds_rep, image_mask=image_mask_rep,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_per_token_logps=old_logps,
        ref_per_token_logps=ref_logps,
        rewards=rewards,
        advantages=advantages,
        metas_rep=metas_rep,
    )
