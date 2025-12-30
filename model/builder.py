# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import Optional, Dict, Any
from .language_model.smartcity_llm import SmartCityLLMForCausalLM

def build_smartcity_llm(
    decoder_name: str,
    encoder_name: str,
    *,
    alpha_understand: float = 0.2,
    beta_prediction: float = 1.0,
    beta_reason: float = 1.0,
    lambda_rat: float = 0.3,
    num_chunk_tokens: int = 4,
    adapter_heads: int = 4,
    encoder_max_length: int = 512,
    imagebind_variant: str = "huge",
    num_image_tokens: int = 4,
    group_layers: int = 2,
    group_heads: int = 8,
    group_tau: float = 1.0,
    open_roberta: bool = True,
    open_vpma: bool = True,
    open_imagebind: bool = False,
    open_image_projector: bool = True,
    freeze_decoder: bool = True,
    # MixLoRA
    use_mixlora: bool = False,
    mixlora_cfg_overrides: Optional[Dict[str, Any]] = None,
) -> SmartCityLLMForCausalLM:
    return SmartCityLLMForCausalLM(
        decoder_name=decoder_name,
        encoder_name=encoder_name,
        alpha_understand=alpha_understand,
        beta_prediction=beta_prediction,
        beta_reason=beta_reason,
        lambda_rat=lambda_rat,
        num_chunk_tokens=num_chunk_tokens,
        adapter_heads=adapter_heads,
        encoder_max_length=encoder_max_length,
        imagebind_variant=imagebind_variant,
        num_image_tokens=num_image_tokens,
        group_layers=group_layers,
        group_heads=group_heads,
        group_tau=group_tau,
        open_roberta=open_roberta,
        open_vpma=open_vpma,
        open_imagebind=open_imagebind,
        open_image_projector=open_image_projector,
        freeze_decoder=freeze_decoder,
        use_mixlora=use_mixlora,
        mixlora_cfg_overrides=mixlora_cfg_overrides,
    )
