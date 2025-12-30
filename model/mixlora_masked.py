# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import re
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Configuration
# =========================
@dataclass
class MixLoRAConfig:
    num_experts: int = 20
    lora_r: int = 8
    lora_alpha: float = 16.0
    dropout: float = 0.0
    torch_dtype: torch.dtype = torch.float16
    num_router_mlp_layers: int = 1
    router_hidden_dim: Optional[int] = None
    share_router_for_qkv: bool = True
    share_router_for_wi: bool = True
    universal_at: str = "front"
    universal_indices: Optional[Sequence[int]] = None
    pred_indices: Optional[Sequence[int]] = None
    reason_indices: Optional[Sequence[int]] = None
    model_type: Optional[str] = None
    target_modules_override: Optional[Dict[str, List[str]]] = None
    enable_attention: bool = True
    enable_ffn: bool = True
    freeze_backbone: bool = True
    ensure_nonzero_gating: bool = False
    nonzero_epsilon: float = 0.0

    num_tasks: int = 3

    num_universal: Optional[int] = None
    num_pred: Optional[int] = None
    num_reason: Optional[int] = None
    rank_universal_mul: float = 2.0

TARGET_MODULE_TYPE: Dict[str, Dict[str, List[str] or str]] = {
    "llama": {
        "atte": "self_attn",
        "ffn": "mlp",
        "q": ["self_attn.q_proj"],
        "k": ["self_attn.k_proj"],
        "v": ["self_attn.v_proj"],
        "o": ["self_attn.o_proj"],
        "wi": ["mlp.gate_proj", "mlp.up_proj"],
        "wo": ["mlp.down_proj"],
    },
    "mistral": {
        "atte": "self_attn",
        "ffn": "mlp",
        "q": ["self_attn.q_proj"],
        "k": ["self_attn.k_proj"],
        "v": ["self_attn.v_proj"],
        "o": ["self_attn.o_proj"],
        "wi": ["mlp.gate_proj", "mlp.up_proj"],
        "wo": ["mlp.down_proj"],
    },
    "qwen2": {
        "atte": "self_attn",
        "ffn": "mlp",
        "q": ["self_attn.q_proj"],
        "k": ["self_attn.k_proj"],
        "v": ["self_attn.v_proj"],
        "o": ["self_attn.o_proj"],
        "wi": ["mlp.gate_proj", "mlp.up_proj"],
        "wo": ["mlp.down_proj"],
    },
    "falcon": {
        "atte": "self_attention", "ffn": "mlp",
        "q": [], "k": [], "v": [], "o": [],
        "wi": ["mlp.gate_proj", "mlp.up_proj"],
        "wo": ["mlp.down_proj"],
    },
}


# =========================
# Utilities
# =========================
def _get_module(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    mod = root
    for attr in dotted.split("."):
        if not hasattr(mod, attr):
            return None
        mod = getattr(mod, attr)
    return mod


def _set_module(root: nn.Module, dotted: str, new_mod: nn.Module):
    parts = dotted.split(".")
    parent = root
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], new_mod)


def _extract_layer_id(name: str) -> Optional[int]:
    m = re.search(r"\.(\d+)$", name)
    return int(m.group(1)) if m else None


def _resolve_mapping(cfg: MixLoRAConfig, model: nn.Module) -> Dict[str, List[str] or str]:
    mt = cfg.model_type or getattr(getattr(model, "config", object()), "model_type", None) or "llama"
    if cfg.target_modules_override is not None:
        return cfg.target_modules_override
    return TARGET_MODULE_TYPE.get(mt, TARGET_MODULE_TYPE["llama"])


def _linear_like_dims(mod: nn.Module) -> Tuple[int, int]:
    """
    Return (in_features, out_features). Compatible with nn.Linear / bitsandbytes Linear4bit/Linear8bitLt / other modules with a weight.
    """
    if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
        return int(mod.in_features), int(mod.out_features)
    if hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
        wshape = tuple(mod.weight.shape)  # [out, in]
        if len(wshape) == 2:
            return int(wshape[1]), int(wshape[0])
    raise ValueError(f"Unable to infer linear layer dimensions: {type(mod)}")


# =========================
# Router: task hard mask + token-level softmax (optional epsilon-gating)
# =========================
class MaskedTokenRouter(nn.Module):
    """
    Token-level router + task-level hard mask.

    Task IDs (cfg.num_tasks):
      - 3-task mode: 0 = understand, 1 = prediction, 2 = reason
      - 2-task legacy mode: 0 = prediction, 1 = reason

    Masking (U/P/R are expert groups):
      - understand -> U only
      - prediction -> U ∪ P
      - reason -> U ∪ R
    """
    def __init__(self, input_dim: int, num_experts: int, cfg: MixLoRAConfig):
        super().__init__()
        self.num_experts = num_experts
        self.cfg = cfg

        layers: List[nn.Module] = [nn.Dropout(cfg.dropout)]
        if cfg.num_router_mlp_layers == 1:
            layers.append(nn.Linear(input_dim, num_experts, dtype=cfg.torch_dtype))
        else:
            hd = cfg.router_hidden_dim or input_dim
            layers += [nn.Linear(input_dim, hd, dtype=cfg.torch_dtype), nn.ReLU()]
            for _ in range(cfg.num_router_mlp_layers - 2):
                layers += [nn.Dropout(cfg.dropout), nn.Linear(hd, hd, dtype=cfg.torch_dtype), nn.ReLU()]
            layers += [nn.Dropout(cfg.dropout), nn.Linear(hd, num_experts, dtype=cfg.torch_dtype)]
        self.mlp = nn.Sequential(*layers)

        # [T, E] mask table (tasks x experts)
        self.register_buffer("mask_table", self._build_mask_table(cfg), persistent=False)

        # Getter injected externally (set by wrapper)
        self._get_task_ids: Optional[Callable[[], Optional[torch.Tensor]]] = None

        # Cache for shared router (Q/K/V or WI groups)
        self._cached_routing_weight: Optional[torch.Tensor] = None

    @staticmethod
    def _indices_from_cfg(cfg: MixLoRAConfig) -> Tuple[List[int], List[int], List[int]]:
        E = int(cfg.num_experts)

        # Preferred: auto-split by counts (recommended)
        if cfg.num_universal is not None:
            u = int(cfg.num_universal)
            if cfg.num_pred is not None and cfg.num_reason is not None:
                p = int(cfg.num_pred); r = int(cfg.num_reason)
                assert u + p + r == E, f"Sum of U/P/R counts ({u}+{p}+{r}) must equal num_experts={E}"
            else:
                remaining = E - u
                assert remaining >= 0 and remaining % 2 == 0, "When only U count is set, remaining experts must be divisible between P and R."
                p = remaining // 2; r = remaining - p
            if cfg.universal_at == "front":
                U = list(range(0, u))
                P = list(range(u, u + p))
                R = list(range(u + p, u + p + r))
            else:
                R = list(range(0, r))
                P = list(range(r, r + p))
                U = list(range(r + p, r + p + u))
            return U, P, R

        # Backward-compatible: explicit indices
        if cfg.universal_indices is not None or cfg.pred_indices is not None or cfg.reason_indices is not None:
            U = list(cfg.universal_indices) if cfg.universal_indices is not None else \
                ([0, 1] if cfg.universal_at == "front" else [E - 2, E - 1])
            rest = [i for i in range(E) if i not in U]
            if cfg.pred_indices is not None:
                P = list(cfg.pred_indices)
            else:
                P = rest[: len(rest) // 2]
            if cfg.reason_indices is not None:
                R = list(cfg.reason_indices)
            else:
                R = [i for i in rest if i not in P]
            return U, P, R

        # Legacy default: 2 U experts; split the rest evenly into P/R
        U = [0, 1] if cfg.universal_at == "front" else [E - 2, E - 1]
        remaining = [i for i in range(E) if i not in U]
        assert len(remaining) % 2 == 0, "Remaining experts must be evenly split between P and R."
        half = len(remaining) // 2
        P = remaining[:half]
        R = remaining[half:]
        return U, P, R

    def _build_mask_table(self, cfg: MixLoRAConfig) -> torch.Tensor:
        U, P, R = self._indices_from_cfg(cfg)
        E = int(cfg.num_experts)
        nt = int(getattr(cfg, "num_tasks", 2))

        if nt == 3:
            table = torch.zeros(3, E, dtype=torch.bool)
            table[0, U] = True              # understand: U only
            table[1, U] = True; table[1, P] = True  # pred: U ∪ P
            table[2, U] = True; table[2, R] = True  # reason: U ∪ R
            return table

        if nt == 2:
            table = torch.zeros(2, E, dtype=torch.bool)
            table[0, U] = True; table[0, P] = True  # pred: U ∪ P
            table[1, U] = True; table[1, R] = True  # reason: U ∪ R
            return table

        raise ValueError(f"Unsupported num_tasks={nt}, expect 2 or 3")

    def set_task_getter(self, fn: Callable[[], Optional[torch.Tensor]]):
        self._get_task_ids = fn

    def clear_cache(self):
        self._cached_routing_weight = None

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        ensure_nonzero: bool = False,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        very_neg = torch.finfo(logits.dtype).min  # fp16: -65504
        logits = logits.masked_fill(~mask, very_neg)
        probs = torch.softmax(logits, dim=-1)

        if ensure_nonzero:
            dtype_tiny = torch.finfo(probs.dtype).tiny  # Smallest positive for dtype; fp16 ~6e-5
            eps_val = float(dtype_tiny if (eps is None or eps <= 0.0) else eps)
            eps_t = torch.as_tensor(eps_val, dtype=probs.dtype, device=probs.device)
            add = (~mask).to(probs.dtype) * eps_t
            probs = probs + add
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps_t * probs.shape[-1])
            probs = probs / denom

        return probs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, D]; returns routing_weight: [B, T, E]
        """
        device = hidden_states.device
        routing_logits = self.mlp(hidden_states)  # [B, T, E]
        if self._get_task_ids is None:
            raise RuntimeError("MaskedTokenRouter has no task_ids getter; wrap the model with MixLoRAWrapper.")
        task_ids = self._get_task_ids()
        if task_ids is None:
            raise RuntimeError("task_ids not set for this batch; pass task_ids to forward()/generate().")

        if task_ids.device != device:
            task_ids = task_ids.to(device)
        if task_ids.dtype != torch.long:
            task_ids = task_ids.long()
        mask_table = self.mask_table
        if mask_table.device != device:
            mask_table = mask_table.to(device)

        num_tasks = int(mask_table.size(0))
        tmin = int(task_ids.min().item()) if task_ids.numel() else 0
        tmax = int(task_ids.max().item()) if task_ids.numel() else 0
        if tmin < 0 or tmax >= num_tasks:
            raise RuntimeError(f"task_ids out of range: [{tmin},{tmax}], but num_tasks={num_tasks}")

        mask_btE = mask_table[task_ids].unsqueeze(1).expand_as(routing_logits)
        routing_weight = self._masked_softmax(
            routing_logits,
            mask_btE,
            ensure_nonzero=self.cfg.ensure_nonzero_gating,
            eps=self.cfg.nonzero_epsilon,
        )

        self._cached_routing_weight = routing_weight
        return routing_weight

    def get_cached_routing_weight(self) -> Optional[torch.Tensor]:
        return self._cached_routing_weight


# =========================
# MixLoRA expert aggregation (minimal support for variable rank)
# =========================
class MixLoRAExperts(nn.Module):
    """
    Supports both fixed-rank (legacy) and per-expert variable rank (enabled when r_per_expert is provided).

    - Fixed rank:
        A: [E*r, in], B: [out, E*r]
    - Variable rank:
        A: [sum(r_i), in], B: [out, sum(r_i)]
        with per-expert scaling alpha/sqrt(r_i).
    """
    def __init__(self, in_features: int, out_features: int, cfg: MixLoRAConfig,
                 r_per_expert: Optional[List[int]] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = int(cfg.num_experts)
        self.dropout = nn.Dropout(cfg.dropout)

        if r_per_expert is None:
            # === Legacy path: fixed rank ===
            self.rank = int(cfg.lora_r)
            self._variable_r = False
            self.A = nn.Parameter(torch.empty((self.rank * self.num_experts, self.in_features), dtype=cfg.torch_dtype))
            self.B = nn.Parameter(torch.empty((self.out_features, self.rank * self.num_experts), dtype=cfg.torch_dtype))
            self.scaling = cfg.lora_alpha / math.sqrt(max(1.0, float(self.rank)))
        else:
            # === New path: variable rank ===
            self._variable_r = True
            self.r_per = [int(max(1, int(r))) for r in r_per_expert]
            assert len(self.r_per) == self.num_experts, "r_per_expert length must equal num_experts"
            self.Rsum = int(sum(self.r_per))
            self._offsets = [0]
            for r in self.r_per:
                self._offsets.append(self._offsets[-1] + r)
            self.A = nn.Parameter(torch.empty((self.Rsum, self.in_features), dtype=cfg.torch_dtype))
            self.B = nn.Parameter(torch.empty((self.out_features, self.Rsum), dtype=cfg.torch_dtype))
            scale = [cfg.lora_alpha / math.sqrt(max(1.0, float(r))) for r in self.r_per]
            self.register_buffer("scaling_per", torch.tensor(scale, dtype=cfg.torch_dtype), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        if not getattr(self, "_variable_r", False):
            # Legacy path
            xr = F.linear(x, self.A)  # [B,T,E*r]
            xr = xr.view(x.shape[:-1] + (self.num_experts, self.rank))                               # [B,T,E,r]
            xr = (xr * gate.unsqueeze(-1)).reshape(x.shape[:-1] + (self.num_experts * self.rank,))  # [B,T,E*r]
            delta = F.linear(xr, self.B) * self.scaling                                             # [B,T,out]
            return (residual + delta).to(residual.dtype)
        else:
            # Variable rank
            xr = F.linear(x, self.A)  # [B,T,Rsum]
            parts = []
            scales = self.scaling_per.to(xr.dtype)
            for i in range(self.num_experts):
                s = xr[..., self._offsets[i]: self._offsets[i+1]]          # [B,T,r_i]
                s = s * gate[..., i:i+1]                                   # gate
                s = s * scales[i]                                          # alpha/sqrt(r_i)
                parts.append(s)
            xr_weighted = torch.cat(parts, dim=-1)                          # [B,T,Rsum]
            delta = F.linear(xr_weighted, self.B)                           # [B,T,out]
            return (residual + delta).to(residual.dtype)


# =========================
# Linear wrapper
# =========================
class MixLoRALinear(nn.Module):
    """
    Note: this module only holds a weak reference (weakref) to the Router and does not register it as a submodule.

    This ensures state_dict/safetensors store the Router only once (registered at the layer root),
    avoiding shared-weight serialization issues.
    """
    def __init__(self, base_linear: nn.Module, router: MaskedTokenRouter, use_cache: bool, cfg: MixLoRAConfig):
        super().__init__()
        self.base = base_linear  # Base linear layer (may be 4/8-bit)
        in_f, out_f = _linear_like_dims(base_linear)
        self._router_ref = weakref.ref(router)  # Weak reference
        self.use_cache = use_cache

        # Auto-set variable ranks from U/P/R: U rank = rank_universal_mul * lora_r; P/R rank = lora_r
        try:
            U, P, R = MaskedTokenRouter._indices_from_cfg(cfg)
        except Exception:
            U, P, R = [], [], []
        r_base = int(cfg.lora_r)
        r_u = max(1, int(round(r_base * float(getattr(cfg, "rank_universal_mul", 2.0)))))
        r_per = [r_base] * int(cfg.num_experts)
        for i in U:
            if 0 <= i < len(r_per):
                r_per[i] = r_u

        self.mixlora = MixLoRAExperts(in_f, out_f, cfg, r_per_expert=r_per)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.base(hidden_states)
        router = self._router_ref()
        if router is None:
            raise RuntimeError("Router has been freed or was not registered.")
        if not self.use_cache:
            gate = router(hidden_states)  # First module in the group: compute and cache
        else:
            gate = router.get_cached_routing_weight()
            if gate is None:  # Fallback
                gate = router(hidden_states)
        return self.mixlora(hidden_states, gate, base_out)


# =========================
# Injection: monkey-patch and replace linear submodules (register one Router per layer)
# =========================
def apply_mixlora(model: nn.Module, cfg: MixLoRAConfig) -> nn.Module:
    mapping = _resolve_mapping(cfg, model)

    # Collect each layer (module names end with .<id>)
    layers_by_id: Dict[int, nn.Module] = {}
    for name, module in model.named_modules():
        lid = _extract_layer_id(name)
        if lid is not None:
            layers_by_id[lid] = module

    if not layers_by_id:
        raise RuntimeError("Failed to auto-detect layers (module names do not end with .<layer_id>). Use cfg.target_modules_override to specify mappings.")

    routers: List[MaskedTokenRouter] = []

    def _register_router(layer_root: nn.Module, router: MaskedTokenRouter, counter: int) -> str:
        name = f"_mixlora_shared_router_{counter}"
        setattr(layer_root, name, router)  # Register as a submodule (only once)
        return name

    def _inject_group(layer_root: nn.Module, names: List[str], share_router: bool, router_counter: List[int]):
        router: Optional[MaskedTokenRouter] = None
        use_cache = False
        for nm in names:
            target = _get_module(layer_root, nm)
            if target is None:
                continue
            try:
                in_f, _ = _linear_like_dims(target)
            except Exception:
                continue

            if router is None or not share_router:
                router = MaskedTokenRouter(in_f, cfg.num_experts, cfg)
                routers.append(router)
                _register_router(layer_root, router, router_counter[0])
                router_counter[0] += 1
                use_cache = False
            wrapped = MixLoRALinear(target, router, use_cache=use_cache, cfg=cfg)
            _set_module(layer_root, nm, wrapped)
            use_cache = share_router  # Subsequent modules in the group reuse cached routing weights

    # Inject per layer
    for lid in sorted(layers_by_id.keys()):
        layer_root = layers_by_id[lid]
        counter = [0]
        if cfg.enable_attention:
            if cfg.share_router_for_qkv:
                qkv = mapping.get("q", []) + mapping.get("k", []) + mapping.get("v", [])
                _inject_group(layer_root, qkv, share_router=True, router_counter=counter)
                _inject_group(layer_root, mapping.get("o", []), share_router=False, router_counter=counter)
            else:
                for nm in mapping.get("q", []): _inject_group(layer_root, [nm], share_router=False, router_counter=counter)
                for nm in mapping.get("k", []): _inject_group(layer_root, [nm], share_router=False, router_counter=counter)
                for nm in mapping.get("v", []): _inject_group(layer_root, [nm], share_router=False, router_counter=counter)
                _inject_group(layer_root, mapping.get("o", []), share_router=False, router_counter=counter)
        if cfg.enable_ffn:
            if cfg.share_router_for_wi:
                _inject_group(layer_root, mapping.get("wi", []), share_router=True, router_counter=counter)
                _inject_group(layer_root, mapping.get("wo", []), share_router=False, router_counter=counter)
            else:
                for nm in mapping.get("wi", []): _inject_group(layer_root, [nm], share_router=False, router_counter=counter)
                _inject_group(layer_root, mapping.get("wo", []), share_router=False, router_counter=counter)

    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            trainable = ("mixlora" in name) or ("router" in name)
            p.requires_grad = bool(trainable)

    model._mixlora_router_list = routers  # type: ignore[attr-defined]
    model._task_ids: Optional[torch.Tensor] = None  # type: ignore[attr-defined]
    return model


# =========================
# Runtime wrapper: inject task_ids per batch, clear cache, and support generate()
# =========================
class MixLoRAWrapper(nn.Module):
    """
    Usage:
        base = apply_mixlora(model, cfg)
        wrapped = MixLoRAWrapper(base)

        # task_ids semantics depend on cfg.num_tasks:
        #   - 3-task mode: 0=understand, 1=prediction, 2=reason
        #   - 2-task mode: 0=prediction, 1=reason
        outputs = wrapped(..., task_ids=torch.LongTensor[B])
        wrapped.generate(..., task_ids=torch.LongTensor[B])
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        if not hasattr(model, "_mixlora_router_list"):
            raise RuntimeError("Please call apply_mixlora(model, cfg) before using MixLoRAWrapper.")
        self._wrapped = model
        for r in self._wrapped._mixlora_router_list:   # type: ignore[attr-defined]
            r.set_task_getter(lambda m=self: getattr(m._wrapped, "_task_ids"))

    @property
    def model(self) -> nn.Module:
        return self._wrapped

    @property
    def config(self):
        if hasattr(self._wrapped, "config"):
            return self._wrapped.config
        if hasattr(self._wrapped, "model") and hasattr(self._wrapped.model, "config"):
            return self._wrapped.model.config
        return None

    def forward(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        if task_ids is not None:
            prefer = None
            if "inputs_embeds" in kwargs and torch.is_tensor(kwargs["inputs_embeds"]):
                prefer = kwargs["inputs_embeds"].device
            elif "input_ids" in kwargs and torch.is_tensor(kwargs["input_ids"]):
                prefer = kwargs["input_ids"].device
            if prefer is None:
                try:
                    prefer = next(self._wrapped.parameters()).device
                except StopIteration:
                    prefer = task_ids.device
            if task_ids.device != prefer:
                task_ids = task_ids.to(prefer)

        if hasattr(self._wrapped, "_mixlora_router_list"):
            for r in self._wrapped._mixlora_router_list:
                r.clear_cache()
        setattr(self._wrapped, "_task_ids", task_ids)

        try:
            out = self._wrapped(*args, **kwargs)
        finally:
            if (not self.training) or (not torch.is_grad_enabled()):
                if hasattr(self._wrapped, "_mixlora_router_list"):
                    for r in self._wrapped._mixlora_router_list:
                        r.clear_cache()
                setattr(self._wrapped, "_task_ids", None)
            else:
                if hasattr(self._wrapped, "_mixlora_router_list"):
                    for r in self._wrapped._mixlora_router_list:
                        r.clear_cache()
        return out

    def generate(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        if task_ids is not None:
            prefer = None
            if len(args) >= 1 and torch.is_tensor(args[0]):  # input_ids may be passed as the first positional arg
                prefer = args[0].device
            if "inputs_embeds" in kwargs and torch.is_tensor(kwargs["inputs_embeds"]):
                prefer = kwargs["inputs_embeds"].device
            elif "input_ids" in kwargs and torch.is_tensor(kwargs["input_ids"]):
                prefer = kwargs["input_ids"].device
            if prefer is None:
                try:
                    prefer = next(self._wrapped.parameters()).device
                except StopIteration:
                    prefer = task_ids.device
            if task_ids.device != prefer:
                task_ids = task_ids.to(prefer)

            beams = int(kwargs.get("num_beams", 1))
            nret  = int(kwargs.get("num_return_sequences", 1))
            factor = max(beams, nret)
            if factor > 1 and task_ids.dim() == 1:
                task_ids = task_ids.repeat_interleave(factor)

        if hasattr(self._wrapped, "_mixlora_router_list"):
            for r in self._wrapped._mixlora_router_list:
                r.clear_cache()
        setattr(self._wrapped, "_task_ids", task_ids)
        try:
            out = self._wrapped.generate(*args, **kwargs)
        finally:
            if hasattr(self._wrapped, "_mixlora_router_list"):
                for r in self._wrapped._mixlora_router_list:
                    r.clear_cache()
            setattr(self._wrapped, "_task_ids", None)
        return out

    def get_input_embeddings(self):
        return self._wrapped.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self._wrapped.set_input_embeddings(new_embeddings)

    def gradient_checkpointing_enable(self, **kwargs):
        return self._wrapped.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self._wrapped.gradient_checkpointing_disable()

    def resize_token_embeddings(self, new_num_tokens: int):
        return self._wrapped.resize_token_embeddings(new_num_tokens)

    def enable_input_require_grads(self):
        if hasattr(self._wrapped, "enable_input_require_grads"):
            return self._wrapped.enable_input_require_grads()
        emb = self._wrapped.get_input_embeddings()
        if emb is None:
            return
        def _make_require_grad(module, inputs, output):
            if torch.is_tensor(output):
                return output.requires_grad_(True)
            elif isinstance(output, (tuple, list)):
                return type(output)(o.requires_grad_(True) if torch.is_tensor(o) else o for o in output)
            return output
        emb.register_forward_hook(_make_require_grad)


__all__ = [
    "MixLoRAConfig",
    "MaskedTokenRouter",
    "MixLoRAExperts",
    "MixLoRALinear",
    "apply_mixlora",
    "MixLoRAWrapper",
]
