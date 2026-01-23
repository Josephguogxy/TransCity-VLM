from __future__ import annotations

from typing import Optional, Sequence, Tuple, List
import torch
import torch.nn.functional as F


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    index:  [B, T]
    return: logp for selected tokens, [B, T]
    """
    B, T, V = logits.shape
    logp = -F.cross_entropy(
        logits.reshape(-1, V),
        index.reshape(-1),
        reduction="none",
    )
    return logp.view(B, T)


def _trim_left_padded(
    ids: torch.LongTensor,
    mask: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    left padding => valid tokens at the END
    ids/mask: [B, T]
    """
    max_len = int(mask.sum(dim=1).max().item())
    max_len = max(1, max_len)
    if max_len < ids.size(1):
        ids = ids[:, -max_len:]
        mask = mask[:, -max_len:]
    return ids, mask


def _trim_right_padded(
    ids: torch.LongTensor,
    mask: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
    """
    right padding => valid tokens at the START
    ids/mask: [B, T]
    return: (trimmed_ids, trimmed_mask, T_eff)
    """
    full_T = int(ids.size(1))
    max_len = int(mask.sum(dim=1).max().item()) if mask.numel() else 0
    max_len = max(1, max_len)
    if max_len < full_T:
        ids2 = ids[:, :max_len]
        m2 = mask[:, :max_len]
        return ids2, m2, max_len
    return ids, mask, full_T


def trim_right_padded_3d(
    embeds: torch.Tensor,
    mask: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    if mask.numel() == 0:
        return embeds[:, :0, :], mask[:, :0]

    max_len = int(mask.sum(dim=1).max().item())
    if max_len <= 0:
        return embeds[:, :0, :], mask[:, :0]

    if max_len < embeds.size(1):
        embeds = embeds[:, :max_len, :]
        mask = mask[:, :max_len]
    return embeds, mask


def make_stop_mask(
    ids: torch.LongTensor,
    *,
    eos_id: int,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> torch.LongTensor:
    stop_ids = [int(eos_id)]
    if stop_token_ids:
        stop_ids.extend(int(x) for x in stop_token_ids if x is not None)
    stop_ids = list(dict.fromkeys(stop_ids))

    if len(stop_ids) == 0:
        return torch.ones_like(ids, dtype=torch.long)

    stop_ids_t = torch.tensor(stop_ids, device=ids.device, dtype=ids.dtype)  # [S]
    is_stop = (ids.unsqueeze(-1) == stop_ids_t.view(1, 1, -1)).any(dim=-1)    # [B,T] bool

    c = is_stop.long().cumsum(dim=1)
    mask = (c == 0) | (is_stop & (c == 1))
    return mask.long()


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
    decoder = model.decoder
    emb = decoder.get_input_embeddings()
    dev = emb.weight.device
    dtype = emb.weight.dtype

    # move to device
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
    T_full = int(completion_ids.size(1))
    mb = int(micro_batch_size) if micro_batch_size and micro_batch_size > 0 else B

    outs: List[torch.Tensor] = []
    for st in range(0, B, mb):
        ed = min(B, st + mb)
        sl = slice(st, ed)

        pre_ids = prefix_ids[sl]
        pre_m = prefix_mask[sl]
        suf_ids = suffix_ids[sl]
        suf_m = suffix_mask[sl]
        comp_ids_full = completion_ids[sl]
        comp_m_full = completion_mask[sl]

        ch_e = chunk_embeds[sl]
        ch_m = chunk_mask[sl]
        im_e = image_embeds[sl]
        im_m = image_mask[sl]
        ti = task_ids[sl] if task_ids is not None else None

        # ---- trim paddings ----
        pre_ids, pre_m = _trim_left_padded(pre_ids, pre_m)
        suf_ids, suf_m = _trim_left_padded(suf_ids, suf_m)

        comp_ids, comp_m, T_eff = _trim_right_padded(comp_ids_full, comp_m_full)

        ch_e, ch_m = trim_right_padded_3d(ch_e, ch_m)
        im_e, im_m = trim_right_padded_3d(im_e, im_m)

        # embeds
        pre_emb = emb(pre_ids)
        suf_emb = emb(suf_ids)
        comp_emb = emb(comp_ids)

        inputs_embeds = torch.cat([pre_emb, ch_e, im_e, suf_emb, comp_emb], dim=1)
        attention_mask = torch.cat([pre_m, ch_m, im_m, suf_m, comp_m.long()], dim=1)

        # prompt length
        P = pre_emb.size(1) + ch_e.size(1) + im_e.size(1) + suf_emb.size(1)

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

        logits_keep = int(T_eff) + 1

        # try logits_to_keep
        used_keep = False
        try:
            out = decoder(**{**decoder_kwargs, "logits_to_keep": logits_keep})
            logits = out.logits
            used_keep = True
        except TypeError:
            out = decoder(**decoder_kwargs)
            logits = out.logits
            used_keep = False

        if used_keep and logits.size(1) == logits_keep:
            logits = logits[:, :-1, :]
        else:
            if logits.size(1) >= logits_keep:
                logits = logits[:, -logits_keep:-1, :]
            else:
                start = max(P - 1, 0)
                logits = logits[:, start:start + T_eff, :]

        temp = float(temperature) if (temperature is not None and temperature > 0) else 1.0
        if temp != 1.0:
            logits = logits / temp

        per_tok_logp = selective_log_softmax(logits, comp_ids)
        per_tok_logp = per_tok_logp * comp_m.to(per_tok_logp.dtype)

        if T_eff < T_full:
            padded = per_tok_logp.new_zeros((per_tok_logp.size(0), T_full))
            padded[:, :T_eff] = per_tok_logp
            per_tok_logp = padded

        outs.append(per_tok_logp)

        del out, logits, inputs_embeds, attention_mask, position_ids, pre_emb, suf_emb, comp_emb

    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
