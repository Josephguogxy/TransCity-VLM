# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import torch
import torch.nn as nn
from .group import GroupingBlock

class ImageProjectorGrouping(nn.Module):
    """
    Vision -> LLM: linear projection + L grouping stages, producing M concept tokens.
    """
    def __init__(self, d_vision: int, d_llm: int, m_tokens: int = 4, n_stages: int = 2, n_heads: int = 8, tau: float = 1.0):
        super().__init__()
        self.proj_in = nn.Linear(d_vision, d_llm)
        self.blocks = nn.ModuleList([GroupingBlock(d_llm, m_tokens, n_heads=n_heads, n_layers=2, tau=tau) for _ in range(n_stages)])
        self.out = nn.Identity()

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(patch_feats)      # [B, Np, D]
        for blk in self.blocks:
            z = blk(z)                     # [B, M, D]
        return self.out(z)                 # [B, M, D]
