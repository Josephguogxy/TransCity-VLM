# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import Optional, Sequence
import inspect
import torch
import torch.nn.functional as F

def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    index : [B, T]  (token ids)
    return: [B, T]  (log p(token))
    Memory-friendly: avoids explicitly building the [B,T,V] log_softmax tensor.
    """
    B, T, V = logits.shape
    logp = -F.cross_entropy(
        logits.reshape(-1, V),
        index.reshape(-1),
        reduction="none",
    )
    return logp.view(B, T)

def make_stop_mask(
    ids: torch.LongTensor,
    *,
    eos_id: int,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> torch.LongTensor:
    """
    mask=1: keep tokens up to the first EOS/stop token (inclusive), then 0 afterwards.
    ids: [B, T]
    """
    stop_ids = [int(eos_id)]
    if stop_token_ids:
        stop_ids.extend(int(x) for x in stop_token_ids if x is not None)
    stop_ids = list(dict.fromkeys(stop_ids))  # Deduplicate while preserving order

    if len(stop_ids) == 0:
        return torch.ones_like(ids, dtype=torch.long)

    stop_ids_t = torch.tensor(stop_ids, device=ids.device, dtype=ids.dtype)  # [S]
    is_stop = (ids.unsqueeze(-1) == stop_ids_t.view(1, 1, -1)).any(dim=-1)    # [B,T] bool

    c = is_stop.long().cumsum(dim=1)  # After the first stop token, c > 0
    mask = (c == 0) | (is_stop & (c == 1))  # Include the first stop token itself
    return mask.long()


def _decoder_accepts_logits_to_keep(decoder) -> bool:
    cache_name = "_has_logits_to_keep"
    if hasattr(decoder, cache_name):
        return bool(getattr(decoder, cache_name))
    try:
        sig = inspect.signature(decoder.forward)
        has = ("logits_to_keep" in sig.parameters)
    except (TypeError, ValueError):
        has = False
    setattr(decoder, cache_name, has)
    return has


def get_per_token_logps_smartcity(
    model,
    *,
    prefix_ids: torch.LongTensor,
    prefix_mask: torch.LongTensor,
    suffix_ids: torch.LongTensor,
    suffix_mask: torch.LongTensor,
    chunk_embeds: torch.Tensor,
    chunk_mask: torch.LongTensor,
    image_embeds: torch.Tensor,
    image_mask: torch.LongTensor,
    completion_ids: torch.LongTensor,
    completion_mask: torch.LongTensor,
    task_ids: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    micro_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Return per-token log-probabilities for the completion only: [B, T_completion].
    - Memory optimization 1: try passing logits_to_keep=T+1 (only keep tail logits)
    - Memory optimization 2: compute logp via cross_entropy (no explicit [B,T,V] log_softmax)
    - Supports micro-batching (set smaller values like 4/8 if memory is tight)
    """
    decoder = model.decoder
    emb = decoder.get_input_embeddings()
    dev = emb.weight.device
    dtype = emb.weight.dtype

    # Move tensors to the decoder device (chunk/image embeds follow decoder dtype)
    prefix_ids = prefix_ids.to(dev)
    prefix_mask = prefix_mask.to(dev)
    suffix_ids = suffix_ids.to(dev)
    suffix_mask = suffix_mask.to(dev)
    completion_ids = completion_ids.to(dev)
    completion_mask = completion_mask.to(dev)

    chunk_embeds = chunk_embeds.to(dev, dtype=dtype)
    image_embeds = image_embeds.to(dev, dtype=dtype)
    chunk_mask = chunk_mask.to(dev)
    image_mask = image_mask.to(dev)

    if task_ids is not None:
        task_ids = task_ids.to(dev)

    B = int(completion_ids.size(0))
    T = int(completion_ids.size(1))
    mb = int(micro_batch_size) if micro_batch_size and micro_batch_size > 0 else B

    outs: list[torch.Tensor] = []
    for st in range(0, B, mb):
        ed = min(B, st + mb)
        sl = slice(st, ed)

        pre_ids = prefix_ids[sl]
        pre_m   = prefix_mask[sl]
        suf_ids = suffix_ids[sl]
        suf_m   = suffix_mask[sl]
        comp_ids = completion_ids[sl]
        comp_m   = completion_mask[sl]

        ch_e = chunk_embeds[sl]
        ch_m = chunk_mask[sl]
        im_e = image_embeds[sl]
        im_m = image_mask[sl]
        ti   = task_ids[sl] if task_ids is not None else None

        pre_emb  = emb(pre_ids)
        suf_emb  = emb(suf_ids)
        comp_emb = emb(comp_ids)

        prompt_emb  = torch.cat([pre_emb, ch_e, im_e, suf_emb], dim=1)
        prompt_mask = torch.cat([pre_m,   ch_m, im_m, suf_m], dim=1)

        inputs_embeds  = torch.cat([prompt_emb, comp_emb], dim=1)
        attention_mask = torch.cat([prompt_mask, comp_m.long()], dim=1)

        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        decoder_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        if ti is not None and hasattr(model, "_decoder_accepts_task_ids") and model._decoder_accepts_task_ids():
            decoder_kwargs["task_ids"] = ti

        # Keep only tail logits: we need T token logps, so logits_to_keep=T+1, then drop the last logit
        logits_keep = T + 1
        used_keep = False
        if _decoder_accepts_logits_to_keep(decoder):
            decoder_kwargs["logits_to_keep"] = logits_keep
            used_keep = True

        try:
            out = decoder(**decoder_kwargs)
            logits = out.logits
        except TypeError:
            # Some transformers versions/models do not support logits_to_keep: fallback
            if "logits_to_keep" in decoder_kwargs:
                decoder_kwargs.pop("logits_to_keep", None)
            out = decoder(**decoder_kwargs)
            logits = out.logits
            used_keep = False

        # Align to [bs, T, V]: keep positions P-1..P+T-2
        if used_keep and logits.size(1) == logits_keep:
            logits = logits[:, :-1, :]  # -> [bs, T, V]
        else:
            # transformers<=4.48 or models without logits_to_keep: slice manually
            if logits.size(1) >= logits_keep:
                logits = logits[:, -logits_keep:-1, :]
            else:
                # Extreme fallback (unlikely): use the legacy slicing logic
                P = prompt_emb.size(1)
                start = max(P - 1, 0)
                logits = logits[:, start : start + T, :]

        temp = float(temperature) if (temperature is not None and temperature > 0) else 1.0
        if temp != 1.0:
            logits = logits / temp

        per_tok_logp = selective_log_softmax(logits, comp_ids)
        per_tok_logp = per_tok_logp * comp_m.to(per_tok_logp.dtype)
        outs.append(per_tok_logp)

    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
