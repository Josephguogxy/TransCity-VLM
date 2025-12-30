# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MABPostLN(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, ln: bool = True):
        super().__init__()
        self.dim = dim_q
        self.num_heads = int(num_heads)
        assert self.dim % self.num_heads == 0
        self.w_q = nn.Linear(dim_q, self.dim)
        self.w_k = nn.Linear(dim_kv, self.dim)
        self.w_v = nn.Linear(dim_kv, self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.ln0 = nn.LayerNorm(self.dim) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(self.dim) if ln else nn.Identity()
        for m in (self.w_q, self.w_k, self.w_v, self.proj):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.w_q(Q); k = self.w_k(K); v = self.w_v(K)
        B, Sq, D = q.shape
        def _heads(x): return x.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)
        qh, kh, vh = _heads(q), _heads(k), _heads(v)
        attn = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(D // self.num_heads)
        if key_padding_mask is not None:
            pad = (key_padding_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, Sq, D)
        o = self.ln0(q + ctx)
        o = self.ln1(o + F.gelu(self.proj(o)))
        return o

class AdapterVPMA(nn.Module):
    """
    vPMA: S learnable seeds compress token-level text representations into S 'soft tokens', then map them to the LLM embedding dimension.
    """
    def __init__(self, d_enc: int, d_dec: int, num_seeds: int = 4, num_heads: int = 4, ln: bool = True):
        super().__init__()
        self.num_seeds = int(num_seeds)
        self.d_dec = int(d_dec)
        self.seeds = nn.Parameter(torch.empty(1, self.num_seeds, self.d_dec))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MABPostLN(dim_q=self.d_dec, dim_kv=d_enc, num_heads=num_heads, ln=ln)

    def forward(self, token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        B = token_states.size(0)
        # Important: align token_states dtype to this module parameter dtype (often float32).
        param_dtype = self.seeds.dtype
        if token_states.dtype != param_dtype:
            token_states = token_states.to(param_dtype)
        Q = self.seeds.expand(B, -1, -1)  # [B,S,D]
        return self.mab(Q, token_states, key_padding_mask=token_mask)  # [B,S,D]
