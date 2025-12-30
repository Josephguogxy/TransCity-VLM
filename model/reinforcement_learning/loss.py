# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import Optional
import torch


def grpo_loss(
    *,
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor] = None,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    beta_kl: float = 0.0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> torch.Tensor:
    """
    - PPO-style clipping (enabled when old_logps is provided)
    - If old_logps is None: TRL-style GRPO (ratio = exp(x - x.detach) = 1, but keeps gradients)
    - KL: exp(ref - logp) - (ref - logp) - 1 (TRL formulation)
    """
    mask = completion_mask.to(per_token_logps.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)

    adv = advantages.to(per_token_logps.dtype).view(-1, 1)

    if old_per_token_logps is None:

        policy_term = torch.exp(per_token_logps - per_token_logps.detach()) * adv
    else:
        old = old_per_token_logps.to(per_token_logps.dtype)
        log_ratio = per_token_logps - old
        ratio = torch.exp(log_ratio)
        ratio_clipped = torch.clamp(ratio, 1 - float(epsilon_low), 1 + float(epsilon_high))
        policy_term = torch.minimum(ratio * adv, ratio_clipped * adv)

    per_tok_loss = -policy_term

    if beta_kl != 0.0 and ref_per_token_logps is not None:
        ref = ref_per_token_logps.to(per_token_logps.dtype)
        delta = ref - per_token_logps
        per_tok_kl = torch.exp(delta) - delta - 1.0
        per_tok_loss = per_tok_loss + float(beta_kl) * per_tok_kl

    loss = (per_tok_loss * mask).sum(dim=1) / denom
    return loss.mean()
