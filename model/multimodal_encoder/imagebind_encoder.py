# model/multimodal_encoder/imagebind_encoder.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
import warnings
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel

# -----------------------------
# Optional: ImageBind backend (保留兼容性)
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
    2025H2 多模态视觉编码器
    支持后端：
      1. SigLIP (Google): SOTA 推荐。输出 [B, 729, 1152] (384px)
      2. CLIP (OpenAI): 经典基准。输出 [B, 576, 1024] (336px)
      3. ImageBind (Meta): 仅用于全模态对齐。输出 [B, 1, 1024]
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
        
        # === 自动判定后端 ===
        v_lower = self.variant.lower()
        if "siglip" in v_lower:
            self.backend = "siglip"
        elif "clip" in v_lower:
            self.backend = "clip"
        elif "imagebind" in v_lower or "huge" in v_lower:
            self.backend = "imagebind"
        else:
            # 默认 fallback 到 SigLIP (2025趋势) 或 CLIP
            self.backend = "siglip"

        print(f"[VisionEncoder] Backend: {self.backend} | Variant: {self.variant}")

        # === 1. SigLIP Backend (SOTA) ===
        if self.backend == "siglip":
            # 加载 SigLIP Vision Model
            # 注意：AutoModel 会加载整个 SiglipModel，我们需要提取其中的 vision_model
            # 或者直接用 AutoModel 加载 vision tower 路径（如果是单独的repo），通常用 transformers 库加载整体
            try:
                # 尝试直接加载 Vision 部分 (如果 repo 支持)
                self.model = AutoModel.from_pretrained(self.variant)
            except Exception:
                # 某些 repo 可能是全量模型，尝试加载后取 vision_model
                full_model = AutoModel.from_pretrained(self.variant)
                self.model = getattr(full_model, "vision_model", full_model)
            
            # 自动获取 hidden_size (通常 1152)
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

        # 冻结/解冻逻辑
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
            # vision_model(x) 的 output.last_hidden_state 即为 patch features
            # 形状: [B, 729, 1152] (对于 384px)
            outputs = self.model.vision_model(pixel_values=x)
            features = outputs.last_hidden_state 
            return features

        # === CLIP Logic ===
        elif self.backend == "clip":
            outputs = self.model(pixel_values=x)
            # [B, 1+576, 1024]
            features = outputs.last_hidden_state
            # 去掉 CLS token，保留 Spatial Patches
            return features[:, 1:, :]

        # === ImageBind Logic ===
        elif self.backend == "imagebind":
            # 强行提取特征，虽然通常只有 Global
            # ImageBind 官方接口只给 Global，除非魔改源码
            # 这里保持原样，返回 [B, 1, 1024]
            feats = self.model(x)
            if isinstance(feats, dict):
                feats = feats[next(iter(feats))]
            
            if feats.dim() == 2:
                feats = feats.unsqueeze(1) # [B, 1, 1024]
            
            return feats
