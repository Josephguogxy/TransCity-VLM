# Copyright (c) 2024 torchtorch Authors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
import warnings
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel

# -----------------------------
# Optional: ImageBind backend (kept for backward compatibility)
# -----------------------------
_HAS_IMAGEBIND = False
try:
    from .imagebind.models import imagebind_model as _ibm
    from .imagebind.models.imagebind_model import ModalityType as _IB_ModalityType
    _HAS_IMAGEBIND = True
except Exception:
    pass

class ImageBindVisionEncoder(nn.Module):
    """
    Multimodal vision encoder wrapper.

    Supported backends:
      1. SigLIP (Google): recommended default. Output ~[B, 729, 1152] for 384px inputs.
      2. CLIP (OpenAI): classic baseline. Output ~[B, 576, 1024] for 336px inputs (CLS removed).
      3. ImageBind (Meta): legacy option for cross-modal alignment. Output [B, 1, 1024] (global feature).
    """

    def __init__(
        self,
        variant: str = "google/siglip-so400m-patch14-384",
        trainable: bool = False,
        *,
        output_tokens: Literal["patch", "global"] = "patch",
    ):
        super().__init__()
        self.variant = str(variant)
        self.output_tokens = output_tokens
        
        # === Auto-detect backend ===
        v_lower = self.variant.lower()
        if "siglip" in v_lower:
            self.backend = "siglip"
        elif "clip" in v_lower:
            self.backend = "clip"
        elif "imagebind" in v_lower or "huge" in v_lower:
            self.backend = "imagebind"
        else:
            # Default fallback to SigLIP (or CLIP)
            self.backend = "siglip"

        print(f"[VisionEncoder] Backend: {self.backend} | Variant: {self.variant}")

        # === 1. SigLIP Backend (SOTA) ===
        if self.backend == "siglip":
            # Load SigLIP vision model
            # Note: AutoModel may load a full SiglipModel; we may need to extract vision_model.
            # Alternatively, some repos expose a standalone vision tower; handle both cases.
            try:
                # Try loading the vision part directly (if the repo supports it).
                self.model = AutoModel.from_pretrained(self.variant)
            except Exception:
                # Some repos ship a full model; fall back to extracting vision_model.
                full_model = AutoModel.from_pretrained(self.variant)
                self.model = getattr(full_model, "vision_model", full_model)
            
            # Infer hidden_size (often 1152).
            self.hidden_size = getattr(self.model.config, "hidden_size", 1152)
            if hasattr(self.model.config, "vision_config"):
                 self.hidden_size = self.model.config.vision_config.hidden_size

        # === 2. CLIP Backend ===
        elif self.backend == "clip":
            self.model = CLIPVisionModel.from_pretrained(self.variant)
            self.hidden_size = self.model.config.hidden_size

        # === 3. ImageBind Backend (Legacy) ===
        elif self.backend == "imagebind":
            if not _HAS_IMAGEBIND:
                raise ImportError("ImageBind not installed.")
            if "huge" in v_lower:
                self.model = _ibm.imagebind_huge(pretrained=True)
            else:
                self.model = _ibm.imagebind_base(pretrained=True)
            self.hidden_size = 1024

        # Freeze/unfreeze parameters
        for p in self.model.parameters():
            p.requires_grad_(bool(trainable))
        if not trainable:
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 3, H, W] (SigLIP: 384, CLIP: 336)
        Output: [B, N_patches, D]
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        x = pixel_values.to(device=device, dtype=dtype)

        # === SigLIP Logic ===
        if self.backend == "siglip":
            # SigLIP forward: returns BaseModelOutputWithPooling
            # outputs.last_hidden_state provides patch features.
            # Shape: [B, 729, 1152] for 384px inputs.
            outputs = self.model.vision_model(pixel_values=x)
            features = outputs.last_hidden_state 
            return features

        # === CLIP Logic ===
        elif self.backend == "clip":
            outputs = self.model(pixel_values=x)
            # [B, 1+576, 1024]
            features = outputs.last_hidden_state
            # Drop the CLS token and keep spatial patches.
            return features[:, 1:, :]

        # === ImageBind Logic ===
        elif self.backend == "imagebind":
            # Extract features (ImageBind typically exposes a global feature).
            # Official ImageBind API returns global features unless the code is modified.
            # Keep the original behavior: return [B, 1, 1024].
            feats = self.model(x)
            if isinstance(feats, dict):
                feats = feats[next(iter(feats))]
            
            if feats.dim() == 2:
                feats = feats.unsqueeze(1) # [B, 1, 1024]
            
            return feats