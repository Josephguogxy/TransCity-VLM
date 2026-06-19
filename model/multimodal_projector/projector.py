# Copyright (c) 2024 torchtorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch
import torch.nn as nn
from .group import GroupingBlock

class ImageProjectorGrouping(nn.Module):
    """
    Vision → LLM：线性对齐 + L 层 Grouping，输出 M 个概念 token。
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
