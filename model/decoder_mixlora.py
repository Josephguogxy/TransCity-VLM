# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .mixlora_masked import MixLoRAConfig, apply_mixlora, MixLoRAWrapper

def _local_device_map() -> Optional[Dict[str, int | str]]:
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            return {"": idx}
        except Exception:
            return None
    return None

class DecoderWithMixLoRA(nn.Module):
    """
    Build a decoder with MixLoRA injected (optional; SmartCityLLM already supports injection internally).
    """
    def __init__(
        self,
        decoder_name: str,
        *,
        quantization_config=None,
        mixlora_cfg_overrides: Optional[Dict[str, Any]] = None,
        force_device_map_none: bool = False,
    ):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            device_map=(None if force_device_map_none else _local_device_map()),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )

        emb = base.get_input_embeddings()
        dtype = emb.weight.dtype
        falcon_ffn_override = {
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [], "k": [], "v": [], "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        }
        mix_cfg = MixLoRAConfig(
            num_experts=20,
            lora_r=8, lora_alpha=16.0,
            dropout=0.0,
            torch_dtype=dtype,
            universal_at="front",
            share_router_for_qkv=True,
            share_router_for_wi=True,
            enable_attention=False,
            enable_ffn=True,
            freeze_backbone=True,
            model_type="falcon",
            target_modules_override=falcon_ffn_override,
            ensure_nonzero_gating=True,
            nonzero_epsilon=0.0,
        )
        if mixlora_cfg_overrides:
            tmo = mixlora_cfg_overrides.get("target_modules_override", falcon_ffn_override)
            mix_cfg.target_modules_override = tmo
            for k, v in mixlora_cfg_overrides.items():
                if k == "target_modules_override":
                    continue
                setattr(mix_cfg, k, v)

        base = apply_mixlora(base, mix_cfg)
        self.model = MixLoRAWrapper(base)
        self.embed_dim = emb.embedding_dim
        self.config = self.model.model.config

    def forward(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        return self.model(*args, task_ids=task_ids, **kwargs)

    def generate(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        return self.model.generate(*args, task_ids=task_ids, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_size: int):
        if hasattr(self.model, "resize_token_embeddings"):
            return self.model.resize_token_embeddings(new_size)
        if hasattr(self.model, "model") and hasattr(self.model.model, "resize_token_embeddings"):
            return self.model.model.resize_token_embeddings(new_size)
        raise AttributeError("Underlying model has no resize_token_embeddings")

    def gradient_checkpointing_enable(self, **kwargs):
        return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()
