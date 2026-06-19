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
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel

class ChunkTextEncoder(nn.Module):
    """
    RoBERTa/BGE 等：返回 token-level hidden 与 mask。
    """
    def __init__(self, model_name: str, max_length: Optional[int] = None, trainable: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # For RoBERTa chunk encoding we use token-level last_hidden_state, not pooler_output.
        # add_pooling_layer=False avoids randomly initialized pooler.dense.* warnings.
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else None),
                trust_remote_code=True,
                add_pooling_layer=False,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else None),
                trust_remote_code=True,
            )
        for p in self.model.parameters():
            p.requires_grad_(bool(trainable))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = int(self.config.hidden_size)
        self.max_length = max_length or getattr(self.config, "max_position_embeddings", 512) or 512

    def forward_tokens(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(texts) == 0:
            dev = next(self.model.parameters()).device
            return torch.empty(0, 0, self.hidden_size, device=dev), torch.empty(0, 0, dtype=torch.long, device=dev)
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            verbose=False,
        )
        dev = next(self.model.parameters()).device
        tok = {k: v.to(dev) for k, v in tok.items()}
        out = self.model(**tok, return_dict=True)
        hidden = out.last_hidden_state  # [N,T,H]
        mask = tok["attention_mask"]    # [N,T]
        return hidden, mask
