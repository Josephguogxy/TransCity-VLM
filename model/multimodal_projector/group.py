# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupingBlock(nn.Module):
    """
    Concept grouping block: learned queries interact with the input sequence -> soft assignment -> aggregation.
    """
    def __init__(self, d_model: int, m_queries: int, n_heads: int = 8, n_layers: int = 2, tau: float = 1.0):
        super().__init__()
        self.m = int(m_queries)
        self.tau = float(tau)
        self.query = nn.Parameter(torch.randn(self.m, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        return: [B, M, D]
        """
        B, N, D = x.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1)             # [B,M,D]
        h = self.encoder(torch.cat([q, x], dim=1))                 # [B,M+N,D]
        q_new, x_new = h[:, :self.m], h[:, self.m:]               # [B,M,D], [B,N,D]
        sim = F.normalize(q_new, dim=-1) @ F.normalize(x_new, dim=-1).transpose(-1, -2)  # [B,M,N]
        assign = torch.softmax(sim / self.tau, dim=-1)            # [B,M,N]
        grouped = assign @ x_new                                   # [B,M,D]
        return q_new + self.mlp(grouped)                           # [B,M,D]
