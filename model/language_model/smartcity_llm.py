# model/smartcity_llm.py
# Copyright (c) 2024 torchtorch Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import os, json, warnings, collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, DynamicCache
import torch.distributed as dist

import inspect
from transformers.modeling_utils import (
    load_state_dict as hf_load_state_dict,
    load_sharded_checkpoint as hf_load_sharded_checkpoint,
)

from ..multimodal_encoder.chunk_text_encoder import ChunkTextEncoder
from ..multimodal_encoder.imagebind_encoder import ImageBindVisionEncoder
from ..multimodal_projector.vpma import AdapterVPMA
from ..multimodal_projector.projector import ImageProjectorGrouping

_HAS_MIXLORA = True
try:
    from ..decoder_mixlora import MixLoRAConfig, apply_mixlora, MixLoRAWrapper
except Exception:
    _HAS_MIXLORA = False

SMARTCITY_TEXT_ONLY = os.environ.get("SMARTCITY_TEXT_ONLY", "0") == "1"

TASK_UNDERSTAND = 0
TASK_PREDICTION = 1
TASK_REASON     = 2


class SmartCityLLMForCausalLM(nn.Module):
    """
    SmartCityLLM:

      - chunk text -> vPMA (S=4 by default)
      - images (SigLIP/CLIP) -> grouping projector (M=64 by default)
      - optional MixLoRA injection (FFN only)
    """
    def __init__(
        self,
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
        open_roberta: bool = True,
        open_vpma: bool = True,
        imagebind_variant: str = "google/siglip-so400m-patch14-384",
        num_image_tokens: int = 64,
        group_layers: int = 2,
        group_heads: int = 8,
        group_tau: float = 1.0,
        open_imagebind: bool = False,
        open_image_projector: bool = True,
        freeze_decoder: bool = True,
        # === MixLoRA control ===
        use_mixlora: bool = False,
        mixlora_cfg_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # ---- LLM decoder ----
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available()
                         and torch.cuda.get_device_capability(0)[0] >= 8 else None),
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        for p in self.decoder.parameters():
            p.requires_grad_(not freeze_decoder)

        self.config = getattr(self.decoder, "config", None)
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            try:
                self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.decoder.gradient_checkpointing_enable()
        if self.config is not None:
            self.config.use_cache = False

        emb = self.decoder.get_input_embeddings()
        if emb is None:
            raise RuntimeError("Decoder has no input embeddings")
        self.embed_dim = int(emb.embedding_dim)

        # === Optional: inject MixLoRA (FFN only) ===
        self.use_mixlora: bool = False
        if use_mixlora:
            if not _HAS_MIXLORA:
                raise ImportError("MixLoRA dependency not found; ensure ../decoder_mixlora.py and ../mixlora_masked.py are importable.")
            dec_dtype = next(self.decoder.parameters()).dtype

            auto_model_type = str(getattr(getattr(self.decoder, "config", object()),
                                          "model_type", "llama")).lower()

            mix_cfg = MixLoRAConfig(
                num_experts=10,
                lora_r=8,
                lora_alpha=16.0,
                dropout=0.0,
                torch_dtype=dec_dtype,
                enable_attention=False,
                enable_ffn=True,
                share_router_for_wi=True,
                freeze_backbone=True,
                model_type=auto_model_type,
                target_modules_override=None,
                ensure_nonzero_gating=True,
                nonzero_epsilon=0.0,
            )
            if mixlora_cfg_overrides:
                tmo = mixlora_cfg_overrides.get("target_modules_override", None)
                if tmo is not None:
                    mix_cfg.target_modules_override = tmo
                for k, v in mixlora_cfg_overrides.items():
                    if k == "target_modules_override":
                        continue
                    setattr(mix_cfg, k, v)

            self.decoder = apply_mixlora(self.decoder, mix_cfg)
            self.decoder = MixLoRAWrapper(self.decoder)
            self.use_mixlora = True

        self.config = getattr(self.decoder, "config",
                              getattr(getattr(self.decoder, "model", None), "config", self.config))

        if hasattr(self.decoder, "enable_input_require_grads"):
            try:
                self.decoder.enable_input_require_grads()
            except Exception as e:
                print("[warn] enable_input_require_grads failed in __init__:", e)

        # ---- Text encoder + vPMA ----
        self.encoder = ChunkTextEncoder(
            model_name=encoder_name,
            max_length=encoder_max_length,
            trainable=open_roberta
        )
        self.adapter = AdapterVPMA(
            d_enc=self.encoder.hidden_size,
            d_dec=self.embed_dim,
            num_seeds=num_chunk_tokens,
            num_heads=adapter_heads,
            ln=True
        )
        for p in self.adapter.parameters():
            p.requires_grad_(bool(open_vpma))
        self.num_chunk_tokens = int(num_chunk_tokens)

        # ---- Vision Encoder + Projector ----
        self.vision = ImageBindVisionEncoder(variant=imagebind_variant, trainable=open_imagebind)

        var_str = str(imagebind_variant).lower()
        if "384" in var_str:
            self.vision_resolution = 384
        elif "336" in var_str:
            self.vision_resolution = 336
        elif "224" in var_str or "base" in var_str or "huge" in var_str:
            self.vision_resolution = 224
        else:
            self.vision_resolution = 384

        print(f"[SmartCityLLM] Vision variant: {imagebind_variant}, Resolution set to: {self.vision_resolution}")

        self.image_projector = ImageProjectorGrouping(
            d_vision=self.vision.hidden_size,
            d_llm=self.embed_dim,
            m_tokens=num_image_tokens,
            n_stages=group_layers,
            n_heads=group_heads,
            tau=group_tau
        )
        for p in self.image_projector.parameters():
            p.requires_grad_(bool(open_image_projector))
        self.num_image_tokens = int(num_image_tokens)

        # Align dtype with decoder
        dec_dtype = next(self.decoder.parameters()).dtype
        self.adapter.to(dtype=dec_dtype)
        self.image_projector.to(dtype=dec_dtype)

        # Loss weights
        self.alpha = float(alpha_understand)
        self.beta_pred = float(beta_prediction)
        self.beta_reason = float(beta_reason)
        self.lambda_rat = float(lambda_rat)

        self._cached_pad_id: Optional[int] = None
        self._ensure_cached_pad_id()

        self.mm_meta = dict(
            imagebind_variant=imagebind_variant, open_imagebind=open_imagebind,
            num_image_tokens=num_image_tokens, group_layers=group_layers, group_heads=group_heads, group_tau=group_tau,
            num_chunk_tokens=num_chunk_tokens, adapter_heads=adapter_heads, encoder_max_length=encoder_max_length,
            use_mixlora=self.use_mixlora,
        )

    # ---------------- utils ----------------
    def _ensure_cached_pad_id(self) -> int:
        pid = getattr(self, "_cached_pad_id", None)
        if pid is None:
            cfg = getattr(self.decoder, "config", None)
            if cfg is None and hasattr(self.decoder, "model") and hasattr(self.decoder.model, "config"):
                cfg = self.decoder.model.config
            if cfg is None:
                cfg = getattr(self, "config", None)

            if cfg is None:
                warnings.warn("Decoder has no config; fallback pad_token_id=0", RuntimeWarning)
                pad_id = 0
            else:
                pad_id = getattr(cfg, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(cfg, "eos_token_id", 0)
                if isinstance(pad_id, (list, tuple)):
                    pad_id = pad_id[0]
                if pad_id is None:
                    pad_id = 0
            self._cached_pad_id = int(pad_id)
        return self._cached_pad_id

    def _decoder_accepts_task_ids(self) -> bool:
        cache_name = "_decoder_has_task_ids"
        if hasattr(self, cache_name):
            return getattr(self, cache_name)

        try:
            sig = inspect.signature(self.decoder.forward)
            has = ("task_ids" in sig.parameters)
        except (TypeError, ValueError):
            has = False

        setattr(self, cache_name, has)
        return has

    def encode_modalities(
        self,
        chunk_txts_all,
        pixel_values: Optional[torch.Tensor] = None,
        n_images: Optional[torch.Tensor] = None,
        *,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          chunk_embeds, chunk_mask, image_embeds, image_mask, chunk_emb_all, img_proj_hook
        img_proj_hook is a scalar (0.0 * sum) used with ZeRO-3 to ensure all ranks trigger the same parameter-gather/tracing path.
        """
        B = int(batch_size)
        dec_emb = self.decoder.get_input_embeddings()
        dev = device or dec_emb.weight.device
        dtype = dtype or dec_emb.weight.dtype

        # Default hook
        img_proj_hook = torch.zeros((), device=dev, dtype=dtype)

        if SMARTCITY_TEXT_ONLY:
            chunk_embeds = torch.zeros(B, 0, self.embed_dim, device=dev, dtype=dtype)
            chunk_mask = torch.zeros(B, 0, dtype=torch.long, device=dev)
            image_embeds = torch.zeros(B, 0, self.embed_dim, device=dev, dtype=dtype)
            image_mask = torch.zeros(B, 0, dtype=torch.long, device=dev)
            chunk_emb_all = torch.zeros(1, max(1, self.num_chunk_tokens), self.embed_dim, device=dev, dtype=dtype)
            return chunk_embeds, chunk_mask, image_embeds, image_mask, chunk_emb_all, img_proj_hook

        # ===== 1) chunk text -> vPMA =====
        split_sizes, flat_txts = [], []
        if isinstance(chunk_txts_all, list):
            for lst in chunk_txts_all:
                n = len(lst) if isinstance(lst, list) else 0
                split_sizes.append(n)
                if n > 0:
                    flat_txts.extend(lst)
        else:
            split_sizes = [0] * B

        total_chunks = int(sum(split_sizes))
        local_max_chunks = max(1, max(split_sizes) if split_sizes else 1)

        if torch.distributed.is_initialized():
            t_max = torch.tensor([local_max_chunks], device=dev, dtype=torch.long)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            max_chunks = int(t_max.item())
        else:
            max_chunks = local_max_chunks

        S = self.num_chunk_tokens
        max_chunk_tokens = max_chunks * S

        texts_eff = flat_txts if total_chunks > 0 else [" "]
        if next(self.encoder.model.parameters()).device != dev:
            self.encoder.to(dev)
        if next(self.adapter.parameters()).device != dev:
            self.adapter.to(dev)

        token_h, token_m = self.encoder.forward_tokens(texts_eff)
        chunk_emb_all = self.adapter(token_h, token_m)
        if chunk_emb_all.dtype != dtype:
            chunk_emb_all = chunk_emb_all.to(dtype)

        per_emb, per_mask = [], []
        pos = 0
        for n in split_sizes:
            if n > 0:
                cur = chunk_emb_all[pos:pos + n]
                pos += n
                cur = cur.reshape(n * S, self.embed_dim)[:max_chunk_tokens]
                fill_n = min(max(0, n), max_chunks) * S
            else:
                cur = chunk_emb_all.new_zeros(0, self.embed_dim)
                fill_n = 0
            pad_n = max(0, max_chunk_tokens - cur.size(0))
            cur_padded = torch.cat([cur, cur.new_zeros(pad_n, self.embed_dim)], dim=0)
            per_emb.append(cur_padded)

            if fill_n > 0:
                m = torch.cat([
                    torch.ones(fill_n, dtype=torch.long, device=dev),
                    torch.zeros(pad_n, dtype=torch.long, device=dev),
                ], dim=0)
            else:
                m = torch.zeros(max_chunk_tokens, dtype=torch.long, device=dev)
            per_mask.append(m)

        if len(per_emb) > 0:
            chunk_embeds = torch.stack(per_emb, dim=0)
        else:
            chunk_embeds = chunk_emb_all.new_zeros(B, max_chunk_tokens, self.embed_dim)

        if len(per_mask) > 0:
            chunk_mask = torch.stack(per_mask, dim=0)
        else:
            chunk_mask = torch.zeros(B, max_chunk_tokens, dtype=torch.long, device=dev)

        # ===== 2) image → vision + projector =====
        def _sum_imgs(nimg):
            if nimg is None:
                return 0
            if torch.is_tensor(nimg):
                return int(nimg.sum().item())
            try:
                return int(sum(int(x) for x in nimg))
            except Exception:
                return 0

        has_local_imgs = (pixel_values is not None and n_images is not None and (_sum_imgs(n_images) > 0))
        use_vision_global = has_local_imgs
        if torch.distributed.is_initialized():
            flag = torch.tensor([1 if has_local_imgs else 0], device=dev, dtype=torch.long)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            use_vision_global = bool(flag.item() > 0)

        if not use_vision_global:
            image_embeds = torch.zeros(B, 0, self.embed_dim, device=dev, dtype=dtype)
            image_mask = torch.zeros(B, 0, dtype=torch.long, device=dev)
        else:
            if has_local_imgs:
                B_img, max_img, C, H, W = pixel_values.shape
                assert B_img == B, "pixel_values batch dimension does not match batch_size"
                pv = pixel_values.to(dev).view(B * max_img, C, H, W)

                train_vision = any(p.requires_grad for p in self.vision.parameters())
                train_proj = any(p.requires_grad for p in self.image_projector.parameters())

                ctx_v = nullcontext() if train_vision else torch.no_grad()
                with ctx_v:
                    vision_feats = self.vision(pv)  # [B*N, 729, D] (SigLIP)

                if next(self.image_projector.parameters()).device != dev:
                    self.image_projector.to(dev)
                proj_dtype = next(self.image_projector.parameters()).dtype

                ctx_p = nullcontext() if train_proj else torch.no_grad()
                with ctx_p:
                    img_emb_all = self.image_projector(vision_feats.to(dev, dtype=proj_dtype))  # [B*N, Mv, D_llm]

                img_emb_all = img_emb_all.to(dtype)

                Mv = self.num_image_tokens
                img_tokens = img_emb_all.reshape(B, max_img * Mv, self.embed_dim)

                per_mask_img = []
                for i in range(B):
                    valid = int(n_images[i].item()) if torch.is_tensor(n_images[i]) else int(n_images[i])
                    valid_tokens = max(0, min(valid, max_img)) * Mv
                    m = torch.cat([
                        torch.ones(valid_tokens, dtype=torch.long, device=dev),
                        torch.zeros(max_img * Mv - valid_tokens, dtype=torch.long, device=dev),
                    ], dim=0)
                    per_mask_img.append(m)

                image_embeds = img_tokens
                image_mask = torch.stack(per_mask_img, dim=0)

                # Optional: no hook needed here because projector params are already on the inputs_embeds path.
            else:
                # No local images but some ranks have images: create a hook to avoid ZeRO-3 divergence.
                H_img = getattr(self, "vision_resolution", 384)
                W_img = H_img
                dummy = torch.zeros(1, 3, H_img, W_img, device=dev, dtype=dtype)

                train_vision = any(p.requires_grad for p in self.vision.parameters())
                train_proj = any(p.requires_grad for p in self.image_projector.parameters())

                ctx_v = nullcontext() if train_vision else torch.no_grad()
                with ctx_v:
                    _dummy_feats = self.vision(dummy)

                if next(self.image_projector.parameters()).device != dev:
                    self.image_projector.to(dev)
                proj_dtype = next(self.image_projector.parameters()).dtype

                ctx_p = nullcontext() if train_proj else torch.no_grad()
                with ctx_p:
                    _dummy_proj = self.image_projector(_dummy_feats.to(dev, dtype=proj_dtype))

                # Key: attach projector params to the graph (zero gradient, but ensures consistent gather/tracing).
                img_proj_hook = _dummy_proj.sum().to(dtype) * 0.0

                image_embeds = torch.zeros(B, 0, self.embed_dim, device=dev, dtype=dtype)
                image_mask = torch.zeros(B, 0, dtype=torch.long, device=dev)

        return chunk_embeds, chunk_mask, image_embeds, image_mask, chunk_emb_all, img_proj_hook

    @torch.no_grad()
    def generate_with_modalities_cached(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
        chunk_embeds: torch.Tensor,
        chunk_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: Optional[int | List[int] | Tuple[int, ...]] = None,
        pad_token_id: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.LongTensor:
        decoder = self.decoder
        device = prefix_ids.device
        B = int(prefix_ids.size(0))

        cfg = getattr(decoder, "config", None)
        if eos_token_id is None and cfg is not None:
            eos_token_id = getattr(cfg, "eos_token_id", None)
        if pad_token_id is None and cfg is not None:
            pad_token_id = getattr(cfg, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else None

        eos_ids: List[int] = []
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, (list, tuple)):
            eos_ids = [int(x) for x in eos_token_id]
        else:
            eos_ids = [int(eos_token_id)]
        if stop_token_ids:
            eos_ids += [int(x) for x in stop_token_ids if x is not None]
        eos_ids = list(dict.fromkeys(eos_ids))

        def _trim_left(ids: torch.LongTensor, mask: torch.LongTensor):
            max_len = int(mask.sum(dim=1).max().item())
            max_len = max(1, max_len)
            if max_len < ids.size(1):
                ids = ids[:, -max_len:]
                mask = mask[:, -max_len:]
            return ids, mask

        def _trim_right_embeds(embeds: torch.Tensor, mask: torch.LongTensor):
            if mask.numel() == 0:
                return embeds[:, :0, :], mask[:, :0]
            max_len = int(mask.sum(dim=1).max().item())
            max_len = max(1, max_len)
            if max_len <= 0:
                return embeds[:, :0, :], mask[:, :0]
            if max_len < embeds.size(1):
                embeds = embeds[:, :max_len, :]
                mask = mask[:, :max_len]
            return embeds, mask

        prefix_ids, prefix_mask = _trim_left(prefix_ids, prefix_mask)
        suffix_ids, suffix_mask = _trim_left(suffix_ids, suffix_mask)
        chunk_embeds, chunk_mask = _trim_right_embeds(chunk_embeds, chunk_mask)
        image_embeds, image_mask = _trim_right_embeds(image_embeds, image_mask)

        was_training = decoder.training
        decoder.eval()

        prev_use_cache = None
        if cfg is not None:
            prev_use_cache = getattr(cfg, "use_cache", None)
            cfg.use_cache = True

        emb = decoder.get_input_embeddings()
        dtype = emb.weight.dtype

        prefix_ids = prefix_ids.to(device)
        suffix_ids = suffix_ids.to(device)
        prefix_mask = prefix_mask.to(device)
        suffix_mask = suffix_mask.to(device)

        chunk_embeds = chunk_embeds.to(device, dtype=dtype)
        image_embeds = image_embeds.to(device, dtype=dtype)
        chunk_mask = chunk_mask.to(device)
        image_mask = image_mask.to(device)

        if task_ids is not None:
            task_ids = task_ids.to(device)

        pre_emb = emb(prefix_ids)
        suf_emb = emb(suffix_ids)

        prompt_emb = torch.cat([pre_emb, chunk_embeds, image_embeds, suf_emb], dim=1)
        prompt_mask = torch.cat([prefix_mask, chunk_mask, image_mask, suffix_mask], dim=1)

        position_ids = prompt_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(prompt_mask.eq(0), 0)
        next_pos = prompt_mask.long().sum(dim=1)

        past_key_values = None
        inputs_embeds = prompt_emb
        attn_mask = prompt_mask

        generated: List[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        try:
            for _ in range(max_new_tokens):
                decoder_kwargs = dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                if task_ids is not None and self._decoder_accepts_task_ids():
                    decoder_kwargs["task_ids"] = task_ids

                outputs = decoder(**decoder_kwargs)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                scores = logits if (temperature is None or temperature == 0.0) else (logits / float(temperature))

                if do_sample:
                    probs = torch.softmax(scores, dim=-1)
                    if top_k and top_k > 0:
                        k = min(int(top_k), probs.size(-1))
                        topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
                        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                        sampled = torch.multinomial(topk_probs, num_samples=1)
                        next_token_ids = torch.gather(topk_idx, -1, sampled)
                    elif top_p and top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                        cumprobs = torch.cumsum(sorted_probs, dim=-1)
                        cutoff = cumprobs > top_p
                        cutoff[..., 0] = False
                        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        sampled = torch.multinomial(sorted_probs, num_samples=1)
                        next_token_ids = torch.gather(sorted_indices, -1, sampled)
                    else:
                        next_token_ids = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_ids = torch.argmax(scores, dim=-1, keepdim=True)

                if eos_ids:
                    prev_finished = finished.clone()
                    tok = next_token_ids.squeeze(-1)
                    eos_hit = torch.zeros_like(tok, dtype=torch.bool)
                    for eid in eos_ids:
                        eos_hit |= tok.eq(int(eid))
                    finished = finished | eos_hit

                    if pad_token_id is not None:
                        next_token_ids = next_token_ids.masked_fill(
                            prev_finished.unsqueeze(-1),
                            int(pad_token_id)
                        )

                generated.append(next_token_ids)
                if finished.all():
                    break

                inputs_embeds = emb(next_token_ids)
                step_mask = (~finished).long().view(B, 1)
                attn_mask = torch.cat([attn_mask, step_mask], dim=1)
                position_ids = next_pos.view(B, 1)
                next_pos = next_pos + (~finished).long()

        finally:
            if cfg is not None and prev_use_cache is not None:
                cfg.use_cache = prev_use_cache
            if was_training:
                decoder.train()

        gen_ids = torch.cat(generated, dim=1) if generated else torch.empty((B, 0), dtype=prefix_ids.dtype, device=device)
        return gen_ids

    # ---------------- forward ----------------
    def forward(self, **batch):
        chunk_txts_all = batch.get("chunk_txts", None)
        prefix_ids, prefix_mask = batch["prefix_ids"], batch["prefix_mask"]
        suffix_ids, suffix_mask = batch["suffix_ids"], batch["suffix_mask"]
        labels = batch["labels"]
        task_ids = batch.get("task_ids", None)

        B = int(labels.size(0))
        dec_emb = self.decoder.get_input_embeddings()
        dev, dtype = dec_emb.weight.device, dec_emb.weight.dtype

        # ===== 1) multimodal encoding (chunk + image) =====
        chunk_embeds, chunk_mask, image_embeds, image_mask, chunk_emb_all, img_proj_hook = self.encode_modalities(
            chunk_txts_all=chunk_txts_all,
            pixel_values=batch.get("pixel_values", None),
            n_images=batch.get("n_images", None),
            batch_size=B,
            device=dev,
            dtype=dtype,
        )

        # ===== 2) assemble inputs to decoder =====
        pre_emb = dec_emb(prefix_ids.to(dev))
        suf_emb = dec_emb(suffix_ids.to(dev))

        pad_id = self._ensure_cached_pad_id()
        lab_for_emb = labels.masked_fill(labels.eq(-100), pad_id).to(dev)
        ans_emb = dec_emb(lab_for_emb)

        inputs_embeds = torch.cat(
            [pre_emb, chunk_embeds, image_embeds, suf_emb, ans_emb], dim=1
        )
        attention_mask = torch.cat(
            [
                prefix_mask.to(dev),
                chunk_mask,
                image_mask,
                suffix_mask.to(dev),
                labels.ne(-100).long().to(dev),
            ],
            dim=1,
        )

        position_ids = attention_mask.to(torch.long).cumsum(dim=1) - 1
        position_ids.clamp_min_(0)

        decoder_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        if task_ids is not None and self._decoder_accepts_task_ids():
            decoder_kwargs["task_ids"] = task_ids

        outputs = self.decoder(**decoder_kwargs)
        logits = outputs.logits

        # ===== 3) Loss =====
        full_labels = torch.cat(
            [
                torch.full_like(prefix_ids, -100, device=dev),
                torch.full((B, chunk_mask.size(1)), -100, device=dev, dtype=labels.dtype),
                torch.full((B, image_mask.size(1)), -100, device=dev, dtype=labels.dtype),
                torch.full_like(suffix_ids, -100, device=dev),
                labels.to(dev),
            ],
            dim=1,
        )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()
        active = shift_labels.ne(-100)

        # Regardless of active labels, attach a dummy loss to keep logits connected to the graph.
        dummy_loss = logits.sum() * 0.0

        if active.any():
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)

            seg_think = batch.get("seg_think", None)
            seg_label = batch.get("seg_label", None)
            seg_target = batch.get("seg_target", None)

            if (seg_think is None) or (seg_label is None) or (seg_target is None):
                m_pred = active
                m_reason_label = active.new_zeros(active.shape, dtype=torch.bool)
                m_reason_think = active.new_zeros(active.shape, dtype=torch.bool)
                m_understand = active.new_zeros(active.shape, dtype=torch.bool)
            else:
                seg_think = seg_think.to(dev).bool()
                seg_label = seg_label.to(dev).bool()
                seg_target = seg_target.to(dev).bool()

                zeros_pre = torch.zeros_like(prefix_ids, dtype=torch.bool, device=dev)
                zeros_chunk = torch.zeros(B, chunk_mask.size(1), dtype=torch.bool, device=dev)
                zeros_img = torch.zeros(B, image_mask.size(1), dtype=torch.bool, device=dev)
                zeros_suf = torch.zeros_like(suffix_ids, dtype=torch.bool, device=dev)

                full_think = torch.cat([zeros_pre, zeros_chunk, zeros_img, zeros_suf, seg_think], dim=1)[:, 1:]
                full_label = torch.cat([zeros_pre, zeros_chunk, zeros_img, zeros_suf, seg_label], dim=1)[:, 1:]
                full_target = torch.cat([zeros_pre, zeros_chunk, zeros_img, zeros_suf, seg_target], dim=1)[:, 1:]

                if task_ids is not None:
                    ti = task_ids.to(dev)
                    is_understand = ti.eq(TASK_UNDERSTAND).view(-1, 1).expand_as(active)
                    is_prediction = ti.eq(TASK_PREDICTION).view(-1, 1).expand_as(active)
                    is_reason     = ti.eq(TASK_REASON).view(-1, 1).expand_as(active)
                else:
                    has_think     = seg_think.any(dim=1, keepdim=True).to(dev)
                    is_reason     = has_think.expand_as(active)
                    is_understand = ~is_reason
                    is_prediction = ~is_reason

                m_understand   = active & full_target & is_understand
                m_pred         = active & full_label  & is_prediction
                m_reason_label = active & full_label  & is_reason
                m_reason_think = active & full_think  & is_reason

            def _masked_mean(x, m):
                if m is None or (m.sum() == 0):
                    return torch.tensor(0.0, device=x.device, dtype=x.dtype)
                return x[m].mean()

            loss_understand = _masked_mean(loss_tok, m_understand)
            loss_prediction = _masked_mean(loss_tok, m_pred)
            loss_reason_label = _masked_mean(loss_tok, m_reason_label)
            loss_reason_think = _masked_mean(loss_tok, m_reason_think)

            real_loss = (
                self.alpha * loss_understand
                + self.beta_pred * loss_prediction
                + self.beta_reason * (loss_reason_label + self.lambda_rat * loss_reason_think)
            )
            loss = real_loss + dummy_loss
        else:
            loss = dummy_loss
            z = torch.tensor(0.0, device=dev)
            loss_understand = loss_prediction = loss_reason_label = loss_reason_think = z

        # ZeRO-3 consistency hook: chunk + projector (img_proj_hook is the scalar computed in encode_modalities)
        loss = loss + (chunk_emb_all.sum().to(loss.dtype) * 0.0) + img_proj_hook.to(loss.dtype)

        return {
            "loss": loss,
            "logits": logits,
            "loss_understand": loss_understand.detach(),
            "loss_prediction": loss_prediction.detach(),
            "loss_reason_label": loss_reason_label.detach(),
            "loss_reason_think": loss_reason_think.detach(),
        }

    def load_full_state_dict(self, load_dir: str, *, cast_to_target_dtype: bool = True, verbose: bool = True):
        pt_idx = os.path.join(load_dir, "pytorch_model.bin.index.json")
        st_idx = os.path.join(load_dir, "model.safetensors.index.json")
        if os.path.exists(pt_idx):
            index_path = pt_idx
            prefer_safe = False
        elif os.path.exists(st_idx):
            index_path = st_idx
            prefer_safe = True
        else:
            raise FileNotFoundError(f"[ZeRO-3] not found index json under {load_dir}")

        try:
            from deepspeed import zero as ds_zero
            GatheredParameters = ds_zero.GatheredParameters
        except Exception:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters

        try:
            from safetensors.torch import load_file as safe_load_file
            _has_st = True
        except Exception:
            _has_st = False

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        shard_cache = {}
        def _get_tensor_from_shard(shard_file: str, full_key: str) -> torch.Tensor:
            path = os.path.join(load_dir, shard_file)
            if path not in shard_cache:
                if prefer_safe or path.endswith(".safetensors"):
                    if not _has_st:
                        raise RuntimeError("safetensors not available but .safetensors shard is required.")
                    shard_cache[path] = safe_load_file(path, device="cpu")
                else:
                    shard_cache[path] = torch.load(path, map_location="cpu")
            return shard_cache[path][full_key]

        def _get_submodule(root: torch.nn.Module, dotted: str):
            try:
                return root.get_submodule(dotted)
            except Exception:
                return None

        def _resolve_param_tensor(root: torch.nn.Module, ckpt_key: str):
            key = ckpt_key
            if key.startswith("module."):
                key = key[len("module."):]
            if "." not in key:
                return None, None, key, "miss"
            prefix, leaf = key.rsplit(".", 1)

            cont = _get_submodule(root, prefix)
            if cont is not None and hasattr(cont, leaf) and isinstance(getattr(cont, leaf), torch.Tensor):
                return cont, leaf, f"{prefix}.{leaf}", "direct"

            if prefix.startswith("decoder."):
                alias_prefix = "decoder._wrapped." + prefix[len("decoder."):]
                cont2 = _get_submodule(root, alias_prefix)
                if cont2 is not None and hasattr(cont2, leaf) and isinstance(getattr(cont2, leaf), torch.Tensor):
                    return cont2, leaf, f"{alias_prefix}.{leaf}", "wrapper_alias"
                base = getattr(cont2, "base", None) if cont2 is not None else None
                if base is not None and hasattr(base, leaf) and isinstance(getattr(base, leaf), torch.Tensor):
                    return base, leaf, f"{alias_prefix}.base.{leaf}", "wrapper+base"

            if cont is not None:
                base = getattr(cont, "base", None)
                if base is not None and hasattr(base, leaf) and isinstance(getattr(base, leaf), torch.Tensor):
                    return base, leaf, f"{prefix}.base.{leaf}", "base_redirect"
            return None, None, key, "miss"

        loaded_cnt, skipped_cnt = 0, 0
        not_found, shape_mismatch, mappings_examples = [], [], []
        mode_counter = collections.Counter()
        is_dist = dist.is_initialized()
        rank0 = (not is_dist) or (dist.get_rank() == 0)

        with torch.no_grad():
            for ckpt_key, shard_file in weight_map.items():
                container, leaf_name, resolved_path, mode = _resolve_param_tensor(self, ckpt_key)
                mode_counter[mode] += 1
                if container is None or leaf_name is None:
                    skipped_cnt += 1
                    not_found.append(ckpt_key)
                    continue
                target: torch.Tensor = getattr(container, leaf_name)

                try:
                    gp_ctx = GatheredParameters([target], modifier_rank=0)
                except TypeError:
                    gp_ctx = GatheredParameters([target])

                with gp_ctx:
                    if rank0:
                        ckpt_t = _get_tensor_from_shard(shard_file, ckpt_key)
                        if cast_to_target_dtype and torch.is_tensor(ckpt_t) and ckpt_t.is_floating_point():
                            ckpt_t = ckpt_t.to(dtype=target.dtype)
                        if tuple(target.shape) != tuple(ckpt_t.shape):
                            shape_mismatch.append((ckpt_key, tuple(ckpt_t.shape), tuple(target.shape)))
                        else:
                            target.data.copy_(ckpt_t)
                            loaded_cnt += 1
                            if resolved_path != ckpt_key and len(mappings_examples) < 20:
                                mappings_examples.append((ckpt_key, resolved_path))
        shard_cache.clear()

        if verbose and rank0:
            print(f"[zero3-load] index={os.path.basename(index_path)} | loaded={loaded_cnt} | skipped={skipped_cnt} | mismatch={len(shape_mismatch)}")
            if sum(mode_counter.values()):
                print("  [modes] " + "; ".join([f"{k}={mode_counter[k]}" for k in mode_counter]))
            if shape_mismatch[:5]:
                print("  [mismatch] ", shape_mismatch[:5])
            if mappings_examples:
                print("  [remap] ", mappings_examples[:5])

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[SmartCityLLM] trainable: {trainable} | total: {total} | {100*trainable/total:.2f}%")

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            try:
                self.decoder.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                self.decoder.gradient_checkpointing_enable()
        if hasattr(self.decoder, "enable_input_require_grads"):
            try:
                self.decoder.enable_input_require_grads()
            except Exception as e:
                print("[warn] enable_input_require_grads failed:", e)
        if hasattr(self.encoder, "model") and hasattr(self.encoder.model, "gradient_checkpointing_enable"):
            try:
                self.encoder.model.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                self.encoder.model.gradient_checkpointing_enable()
        if hasattr(self.decoder, "config"):
            setattr(self.decoder.config, "use_cache", False)

    def gradient_checkpointing_disable(self):
        if hasattr(self.decoder, "gradient_checkpointing_disable"):
            self.decoder.gradient_checkpointing_disable()
        if hasattr(self.encoder, "model") and hasattr(self.encoder.model, "gradient_checkpointing_disable"):
            self.encoder.model.gradient_checkpointing_disable()
        if hasattr(self.decoder, "config"):
            setattr(self.decoder.config, "use_cache", True)


def log_trainable_breakdown(model):
    def cnt(m):
        try:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        except Exception:
            return 0
    items = []
    try:
        items.append(("encoder(RoBERTa)", cnt(model.encoder.model)))
    except Exception:
        pass
    try:
        items.append(("adapter_vPMA", cnt(model.adapter)))
    except Exception:
        pass
    try:
        items.append(("image_projector", cnt(model.image_projector)))
    except Exception:
        pass
    try:
        items.append(("decoder(LLM)", cnt(model.decoder)))
    except Exception:
        pass
    try:
        items.append(("vision_encoder", sum(p.numel() for p in model.vision.parameters() if p.requires_grad)))
    except Exception:
        pass
    print("==== Trainable breakdown ====")
    for k, v in items:
        print(f"{k:>20s}: {v:,}")
    print(f"{'TOTAL':>20s}: {sum(v for _, v in items):,}")
