# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import os
import json

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import Dataset as HFDataset, Features, Value, Sequence
from transformers import PreTrainedTokenizerBase
from collections.abc import Mapping

# =========================
# env switches
# =========================
SMARTCITY_DEBUG = os.environ.get("SMARTCITY_DEBUG", "0") == "1"
SMARTCITY_MAX_BAD_LINES = int(os.environ.get("SMARTCITY_MAX_BAD_LINES", "0"))
SMARTCITY_LOG_EVERY = int(os.environ.get("SMARTCITY_LOG_EVERY", "100000"))

# =========================
# constants / schema
# =========================
FIELD_ORDER = ["POI", "News", "Accident", "HopSensor", "HopBA"]
CHUNK_MARKER = "[chunk token]"
IMAGE_MARKER = "[image token]"

TASK2ROUTER = {"understand": 0, "prediction": 1, "reason": 2}

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

PROMPT_STYLE_QWEN3 = "qwen3"
PROMPT_STYLE_PLAIN = "plain"
PROMPT_STYLES = {PROMPT_STYLE_QWEN3, PROMPT_STYLE_PLAIN}

SMARTCITY_FEATURES = Features({
    "task": Value("string"),
    "chunks": {
        "POI":       Sequence(Value("string")),
        "News":      Sequence(Value("string")),
        "Accident":  Sequence(Value("string")),
        "HopSensor": Sequence(Value("string")),
        "HopBA":     Sequence(Value("string")),
    },
    "images":   Sequence(Value("string")),  # List of paths/URIs (no decoding here)
    "prompt":   Value("string"),
    "thinking": Sequence(Value("string")),
    "target":   Value("string"),
    "answer":   Value("string"),
})

# =========================
# rank/debug
# =========================
def _is_rank0() -> bool:
    rk = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "-1"
    try:
        return int(rk) in (-1, 0)
    except Exception:
        return True

def _dprint(*args, **kwargs):
    if _is_rank0() and SMARTCITY_DEBUG:
        print(*args, **kwargs)

# =========================
# dict helpers
# =========================
def _get_ci(d: Dict[str, Any] | None, key: str, default=None, aliases: tuple[str, ...] = ()) -> Any:
    """Case-insensitive lookup; supports aliases."""
    if d is None:
        return default
    if not isinstance(d, Mapping):
        try:
            d = dict(d)
        except Exception:
            return default
    if key in d:
        return d[key]
    for a in aliases:
        if a in d:
            return d[a]
    wanted = {key.lower(), *(a.lower() for a in aliases)}
    for k, v in d.items():
        try:
            if k.lower() in wanted:
                return v
        except Exception:
            continue
    return default

def _normalize_chunks_ci(raw: Dict[str, Any] | None) -> Dict[str, list[str]]:
    """Normalize chunks according to FIELD_ORDER (case-insensitive); fill missing fields with empty lists."""
    out: Dict[str, list[str]] = {}
    src = raw if isinstance(raw, dict) else {}
    for canon in FIELD_ORDER:
        val = _get_ci(src, canon, default=[])
        if val is None:
            arr = []
        elif isinstance(val, list):
            arr = val
        else:
            arr = [val]
        out[canon] = [str(s) for s in arr if s is not None]
    return out

# =========================
# prompt builder (Qwen3 style)
# =========================
def _validate_prompt_style(prompt_style: str) -> str:
    s = (prompt_style or PROMPT_STYLE_QWEN3).strip().lower()
    if s not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {sorted(PROMPT_STYLES)}; got {prompt_style!r}")
    return s

def _format_message(role: str, text: str, *, prompt_style: str) -> str:
    prompt_style = _validate_prompt_style(prompt_style)
    role = (role or "").strip().lower()
    text = (text or "").strip()

    if prompt_style == PROMPT_STYLE_PLAIN:
        # plain: minimal readable format, supports multi-turn chat
        if role == "system":
            return f"System: {text}".strip()
        if role == "user":
            return f"User: {text}".strip()
        if role == "assistant":
            return f"Assistant: {text}".strip()
        return text.strip()

    # qwen3
    if role not in ("system", "user", "assistant"):
        return ""
    return f"<|im_start|>{role}\n{text}<|im_end|>"

def _build_prompt_from_messages(
    msgs: list,
    *,
    prompt_style: str,
    drop_last_assistant: bool = True,
    add_generation_prompt: bool = True,
) -> Tuple[str, str, List[str]]:
    """
    Build a prompt from multi-turn messages (in order).
    - drop_last_assistant=True: treat the last assistant message as the label (exclude it from prompt history)
    - add_generation_prompt=True: append an "open" assistant segment at the end
    Returns: (prompt, last_assistant_text, images_from_user_turns)
    """
    prompt_style = _validate_prompt_style(prompt_style)

    def _as_str(x) -> str:
        return x if isinstance(x, str) else ""

    def _collect_text(content) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for seg in content:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    t = _as_str(seg.get("text", "")).strip()
                    if t:
                        parts.append(t)
                elif isinstance(seg, str):
                    t = seg.strip()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        elif isinstance(content, str):
            return content.strip()
        return ""

    def _collect_images(content) -> List[str]:
        paths: List[str] = []
        if isinstance(content, list):
            for seg in content:
                if not isinstance(seg, dict):
                    continue
                t = seg.get("type", None)
                if t == "image":
                    uri = seg.get("image", None)
                elif t == "image_url":
                    iu = seg.get("image_url", None)
                    uri = iu.get("url", None) if isinstance(iu, dict) else None
                else:
                    uri = None
                if not (isinstance(uri, str) and uri.strip()):
                    uri = seg.get("url", None) or seg.get("path", None)
                if isinstance(uri, str) and uri.strip():
                    paths.append(uri.strip())
        return paths

    parsed: List[Dict[str, Any]] = []
    images: List[str] = []

    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = _as_str(m.get("role", "")).lower()
            content = m.get("content", "")
            text = _collect_text(content)
            if role == "user":
                images.extend(_collect_images(content))
            if role in ("system", "user", "assistant"):
                parsed.append({"role": role, "text": text})

    # last assistant
    last_asst_text = ""
    last_asst_idx = None
    for i in range(len(parsed) - 1, -1, -1):
        if parsed[i]["role"] == "assistant" and (parsed[i]["text"] or "").strip():
            last_asst_text = parsed[i]["text"].strip()
            last_asst_idx = i
            break

    if drop_last_assistant and last_asst_idx is not None:
        history = parsed[:last_asst_idx]
    else:
        history = parsed

    parts: List[str] = []
    if prompt_style == PROMPT_STYLE_PLAIN:
        for msg in history:
            line = _format_message(msg["role"], msg["text"], prompt_style=prompt_style)
            if line:
                parts.append(line)
        if add_generation_prompt:
            parts.append("Assistant:")
        prompt = "\n".join([p for p in parts if p]).strip()
    else:
        for msg in history:
            seg = _format_message(msg["role"], msg["text"], prompt_style=prompt_style)
            if seg:
                parts.append(seg)
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

    # Deduplicate images while preserving order
    seen = set()
    images_dedup: List[str] = []
    for p in images:
        if p not in seen:
            seen.add(p)
            images_dedup.append(p)

    return prompt, last_asst_text, images_dedup

def _inject_markers_before_open_assistant(prompt: str, markers: List[str]) -> str:
    """
    Insert the marker block right before the final "open assistant" segment (if present).
    Goal: ensure the collator's "single insertion block" consistently lands near the end.
    """
    prompt = prompt or ""
    markers = [m for m in markers if m]
    if not markers:
        return prompt

    injection = "\n".join(markers).strip() + "\n"
    anchor = "<|im_start|>assistant\n"
    idx = prompt.rfind(anchor)
    if idx >= 0:
        return prompt[:idx] + injection + prompt[idx:]
    return (prompt + "\n" + injection).strip()

# =========================
# I/O
# =========================
def load_jsonl_dataset(
    path: str | Path,
    split_name: str = "train",
    *,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
) -> HFDataset:
    return _load_via_generator_flat(str(path), split_name, prompt_style=prompt_style, ensure_markers=ensure_markers)

def save_dataset_jsonl(ds: HFDataset, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _load_via_generator_flat(path: str, split_name: str, *, prompt_style: str, ensure_markers: bool) -> HFDataset:
    prompt_style = _validate_prompt_style(prompt_style)

    def gen():
        bad = 0
        use_orjson = False
        try:
            import orjson  # type: ignore
            use_orjson = True
        except Exception:
            pass

        with open(path, "r", encoding="utf-8", buffering=1 << 20) as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = (orjson.loads(s) if use_orjson else json.loads(s))
                except Exception as e:
                    bad += 1
                    if _is_rank0() and SMARTCITY_DEBUG and (bad <= 10 or bad % 100 == 0):
                        print(f"[JSONL] parse error line {ln}: {e} (bad={bad})")
                    if SMARTCITY_MAX_BAD_LINES and bad > SMARTCITY_MAX_BAD_LINES:
                        raise RuntimeError(f"Too many bad lines: {bad} (limit={SMARTCITY_MAX_BAD_LINES}). Last at {ln}")
                    continue

                yield _normalize_row_core(obj, prompt_style=prompt_style, ensure_markers=ensure_markers)

                if _is_rank0() and SMARTCITY_DEBUG and (ln % SMARTCITY_LOG_EVERY == 0):
                    _dprint(f"[JSONL] parsed {ln:,} lines ... (bad={bad})")

    ds = HFDataset.from_generator(gen, features=SMARTCITY_FEATURES)
    return ds

# =========================
# row normalize (core)
# =========================
def _normalize_row_core(
    ex: Dict[str, Any],
    *,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
) -> Dict[str, Any]:
    """
    Normalize to SMARTCITY_FEATURES:
      {task, chunks, images, prompt, thinking, target, answer}
    Supports multi-turn messages: build the prompt in order and use the last assistant turn as a fallback label.
    """
    prompt_style = _validate_prompt_style(prompt_style)

    try:
        ex = dict(ex)
    except Exception:
        pass

    out: Dict[str, Any] = {}

    # task
    task_in = _get_ci(ex, "task", default=None)
    task = str(task_in).strip().lower() if task_in is not None else ""
    out["task"] = task

    # chunks
    raw_chunks = _get_ci(ex, "chunks", default={})
    chunks = _normalize_chunks_ci(raw_chunks)
    out["chunks"] = chunks

    # thinking
    tk = _get_ci(ex, "thinking", default=[])
    if tk is None:
        tk = []
    if not isinstance(tk, list):
        tk = [str(tk)]
    else:
        tk = [str(t) for t in tk if t is not None]
    out["thinking"] = tk

    # top-level prompt/answer/target/images
    prompt = str(_get_ci(ex, "prompt", default="") or "")
    answer = str(_get_ci(ex, "answer", default="") or "")
    target = str(_get_ci(ex, "target", default="") or "")

    images_top = _get_ci(ex, "images", default=[]) or []
    if not isinstance(images_top, list):
        images_top = [images_top]
    images_top = [str(x).strip() for x in images_top if isinstance(x, str) and str(x).strip()]

    msgs = _get_ci(ex, "messages", default=None)

    images = images_top
    last_asst_text = ""

    if isinstance(msgs, list) and msgs:
        prompt_from_msgs, last_asst_text, imgs_from_msgs = _build_prompt_from_messages(
            msgs,
            prompt_style=prompt_style,
            drop_last_assistant=True,
            add_generation_prompt=True,
        )

        if not prompt and prompt_from_msgs:
            prompt = prompt_from_msgs

        # merge images
        seen = set()
        merged: List[str] = []
        for p in (images_top + imgs_from_msgs):
            if p and (p not in seen):
                seen.add(p)
                merged.append(p)
        images = merged

        # fill answer/target fallback
        tname = task.lower()
        if tname == "understand":
            if (not target) and last_asst_text:
                target = last_asst_text
        else:
            if (not answer) and last_asst_text:
                answer = last_asst_text

    # ensure markers (recommended: insert only once, right before the final open assistant segment)
    if ensure_markers and prompt:
        has_chunks = any((len(v) > 0) for v in (chunks or {}).values())
        has_images = len(images) > 0

        need = []
        if has_chunks and (CHUNK_MARKER not in prompt):
            need.append(CHUNK_MARKER)
        if has_images and (IMAGE_MARKER not in prompt):
            need.append(IMAGE_MARKER)
        if need:
            prompt = _inject_markers_before_open_assistant(prompt, need)

    out["prompt"] = prompt
    out["answer"] = answer
    out["target"] = target
    out["images"] = images

    # fallback keys
    for k in ("prompt", "answer", "target"):
        if k not in out or out[k] is None:
            out[k] = ""
    if "images" not in out or out["images"] is None:
        out["images"] = []

    return out

# =========================
# ensure on existing dataset
# =========================
def ensure_smartcity_features(
    ds: HFDataset,
    *,
    num_proc: int = 16,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
    messages_policy: str = "to_prompt",   # Kept for backward compatibility
    fold_messages_to: str = "News",        # Kept for backward compatibility
) -> HFDataset:
    prompt_style = _validate_prompt_style(prompt_style)
    expected_cols = set(SMARTCITY_FEATURES.keys())
    if set(ds.column_names) == expected_cols:
        return ds.cast(SMARTCITY_FEATURES)

    def _normalize_row(ex: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_row_core(ex, prompt_style=prompt_style, ensure_markers=ensure_markers)

    ds = ds.map(_normalize_row, num_proc=num_proc, desc="smartcity:normalize(messages→prompt/images)")
    to_drop = [c for c in ds.column_names if c not in expected_cols]
    if to_drop:
        ds = ds.remove_columns(to_drop)
    ds = ds.cast(SMARTCITY_FEATURES)
    return ds

# =========================
# chunk helper: 510 tokens/chunk
# =========================
def _chunk_by_token_ids(
    text: str,
    tok: PreTrainedTokenizerBase,
    *,
    chunk_size_tokens: int = 510,
    overlap: int = 0,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []

    overlap = max(0, int(overlap))
    chunk_size_tokens = max(1, int(chunk_size_tokens))
    step = max(1, chunk_size_tokens - overlap)

    out: List[str] = []
    for st in range(0, len(ids), step):
        piece = ids[st: st + chunk_size_tokens]
        if not piece:
            break
        s = tok.decode(piece, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if s:
            out.append(s)
        if st + chunk_size_tokens >= len(ids):
            break
    return out

# =========================
# prompt split: only consume first contiguous marker block
# =========================
def _split_prompt_by_first_marker_block(prompt: str) -> Tuple[str, str]:
    """
    A splitting strategy that works better for multi-turn prompts:
    - find the first marker (chunk or image) position l
    - starting from l, consume the contiguous block consisting of marker/whitespace/marker/...
    - prefix=prompt[:l], suffix=prompt[r:]
    - remove any remaining markers from suffix (avoid exposing marker text to the model)
    """
    prompt = prompt or ""
    idx_chunk = prompt.find(CHUNK_MARKER)
    idx_img = prompt.find(IMAGE_MARKER)
    idxs = [i for i in (idx_chunk, idx_img) if i >= 0]
    if not idxs:
        return prompt, ""

    l = min(idxs)
    r = l
    # Consume a contiguous "whitespace + marker" block
    while True:
        advanced = False
        while r < len(prompt) and prompt[r].isspace():
            r += 1
            advanced = True
        if prompt.startswith(CHUNK_MARKER, r):
            r += len(CHUNK_MARKER)
            advanced = True
            continue
        if prompt.startswith(IMAGE_MARKER, r):
            r += len(IMAGE_MARKER)
            advanced = True
            continue
        if not advanced:
            break

    pre = prompt[:l]
    suf = prompt[r:]
    # Clean residual markers in suffix (avoid semantic confusion from multiple marker occurrences)
    suf = suf.replace(CHUNK_MARKER, "").replace(IMAGE_MARKER, "")
    return pre, suf

# =========================
# collator (multimodal kept)
# =========================
class DataCollatorSmartCity:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        # chunk
        chunk_tokenizer: Optional[PreTrainedTokenizerBase] = None,  # roberta tokenizer
        chunk_size_tokens: int = 510,
        chunk_overlap: int = 0,
        join_fields_before_chunk: bool = True,
        add_field_header: bool = True,

        # image
        max_images_per_sample: int = 1,     # cap (per-sample limit)
        image_size: int = 384,              # Ensure 384 (SigLIP)

        # length/pad
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,

        # seg masks
        use_seg_masks: bool = True,

        # IMPORTANT: must match model config
        chunk_token_factor: int = 4,  # == model.num_chunk_tokens
        image_token_factor: int = 64, # == model.num_image_tokens

        # Added: accept this parameter
        max_chunks_per_sample: Optional[int] = None,

        add_eos_to_labels: bool = True,
        wrap_reason_with_tags: bool = True,
    ):
        self.tok = tokenizer
        self.chunk_tok = chunk_tokenizer if chunk_tokenizer is not None else tokenizer
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.chunk_overlap = int(chunk_overlap)
        self.join_fields_before_chunk = bool(join_fields_before_chunk)
        self.add_field_header = bool(add_field_header)

        self.max_images_per_sample = max(0, int(max_images_per_sample))
        self.image_size = int(image_size)

        # Store parameter
        self.max_chunks_per_sample = int(max_chunks_per_sample) if max_chunks_per_sample is not None else None

        self.max_len = max_length or getattr(tokenizer, "model_max_length", 2048)
        if self.max_len is None or self.max_len > 10_000_000:
            self.max_len = 2048

        self.pad_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        assert self.pad_id is not None and self.eos_id is not None, "pad/eos must not be None"

        self.lab_pad_id = int(label_pad_token_id)
        self.use_seg_masks = bool(use_seg_masks)

        self.chunk_token_factor = int(chunk_token_factor)
        self.image_token_factor = int(image_token_factor)

        self.add_eos_to_labels = bool(add_eos_to_labels)
        self.wrap_reason_with_tags = bool(wrap_reason_with_tags)

        # --- image transform (keep your old CLIP-like norm) ---
        import torchvision.transforms as T
        self._img_tf = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            # Normalization commonly used for SigLIP / ViT-22B and similar models
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _tok_ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]

    @staticmethod
    def _left_trim(seq, mask, need):
        if need <= 0:
            return seq, mask
        if need >= len(seq):
            return [], []
        return seq[need:], mask[need:]

    def _trim_labels_keep_eos(self, ids, seg_think, seg_label, seg_target, need_drop):
        if need_drop <= 0 or len(ids) <= 1:
            return ids, seg_think, seg_label, seg_target
        has_eos = self.add_eos_to_labels and (ids[-1] == self.eos_id)
        if has_eos:
            new_len = max(1, len(ids) - need_drop)
            if new_len == 1:
                return [self.eos_id], [seg_think[-1]], [seg_label[-1]], [seg_target[-1]]
            return (
                ids[: new_len - 1] + [self.eos_id],
                seg_think[: new_len - 1] + [seg_think[-1]],
                seg_label[: new_len - 1] + [seg_label[-1]],
                seg_target[: new_len - 1] + [seg_target[-1]],
            )
        new_len = max(1, len(ids) - need_drop)
        return ids[:new_len], seg_think[:new_len], seg_label[:new_len], seg_target[:new_len]

    def _to_pil(self, obj) -> Image.Image:
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(obj)).convert("RGB")
        if isinstance(obj, dict) and "path" in obj:
            return Image.open(obj["path"]).convert("RGB")
        if isinstance(obj, str):
            return Image.open(obj).convert("RGB")
        return Image.open(obj).convert("RGB")

    def _gather_image_paths(self, ex: Dict[str, Any]) -> List[str]:
        src = ex.get("images", []) or []
        if not isinstance(src, list):
            src = [src]
        paths = [str(x).strip() for x in src if isinstance(x, str) and str(x).strip()]
        if self.max_images_per_sample > 0:
            paths = paths[: self.max_images_per_sample]
        else:
            paths = []
        return paths

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)

        # ========== A) images -> pixel_values / n_images ==========
        img_paths_per_ex: List[List[str]] = []
        n_images_list: List[int] = []
        for ex in batch:
            paths = self._gather_image_paths(ex)
            img_paths_per_ex.append(paths)
            n_images_list.append(len(paths))

        batch_max_images = max(n_images_list) if (self.max_images_per_sample > 0 and n_images_list) else 0
        pixel_values: Optional[torch.Tensor] = None
        n_images: Optional[torch.Tensor] = None

        if batch_max_images > 0:
            pv_list: List[torch.Tensor] = []
            for i in range(B):
                paths = img_paths_per_ex[i]
                tensors: List[torch.Tensor] = []
                for p in paths[:batch_max_images]:
                    try:
                        im = self._to_pil(p)
                        tensors.append(self._img_tf(im))
                    except Exception:
                        tensors.append(torch.zeros(3, self.image_size, self.image_size))
                if len(tensors) < batch_max_images:
                    tensors += [torch.zeros(3, self.image_size, self.image_size)] * (batch_max_images - len(tensors))
                pv_list.append(torch.stack(tensors, dim=0))

            pixel_values = torch.stack(pv_list, dim=0)
            n_images = torch.tensor(n_images_list, dtype=torch.long)

        # ========== B) chunks: join field -> chunk by encoder tokens ==========
        chunk_txts: List[List[str]] = []
        for ex in batch:
            rows = ex.get("chunks", {}) or {}
            merged: List[str] = []
            if self.join_fields_before_chunk:
                for f in FIELD_ORDER:
                    arr = rows.get(f, None) or []
                    lines = [str(s).strip() for s in arr if isinstance(s, str) and str(s).strip()]
                    if not lines:
                        continue
                    big = "\n".join(lines)
                    if self.add_field_header:
                        big = f"{f}:\n{big}"
                    merged.extend(_chunk_by_token_ids(
                        big,
                        self.chunk_tok,
                        chunk_size_tokens=self.chunk_size_tokens,
                        overlap=self.chunk_overlap,
                    ))
            else:
                for f in FIELD_ORDER:
                    arr = rows.get(f, None)
                    if not arr:
                        continue
                    for s in arr:
                        if isinstance(s, str) and s.strip():
                            merged.append(s.strip())
            chunk_txts.append(merged)

        # ========== C) prompt -> prefix/suffix (safe split) ==========
        prefixes, suffixes = [], []
        for ex in batch:
            prompt = ex.get("prompt", "") or ""
            pre, suf = _split_prompt_by_first_marker_block(prompt)
            prefixes.append(pre)
            suffixes.append(suf)

        # ========== D) tokenize + labels ==========
        task_ids: List[int] = []
        pre_ids, pre_m = [], []
        suf_ids, suf_m = [], []
        ans_ids: List[List[int]] = []
        seg_think, seg_label, seg_target = [], [], []

        for i, ex in enumerate(batch):
            tname = (ex.get("task", "") or "").lower()
            task_ids.append(TASK2ROUTER.get(tname, 0))

            if tname in ("understand", "prediction"):
                anchor = "<|im_start|>assistant\n"
                empty_think = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

                suf_text = suffixes[i] or ""
                if anchor in suf_text:
                    pos = suf_text.rfind(anchor) + len(anchor)
                    after = suf_text[pos:]
                    if not after.lstrip().startswith(THINK_OPEN):
                        suf_text = suf_text[:pos] + empty_think + after
                    suffixes[i] = suf_text
                else:
                    pre_text = prefixes[i] or ""
                    if anchor in pre_text:
                        pos = pre_text.rfind(anchor) + len(anchor)
                        after = pre_text[pos:]
                        if not after.lstrip().startswith(THINK_OPEN):
                            pre_text = pre_text[:pos] + empty_think + after
                        prefixes[i] = pre_text

            p = self._tok_ids(prefixes[i])
            s = self._tok_ids(suffixes[i])
            pre_ids.append(p); pre_m.append([1]*len(p))
            suf_ids.append(s); suf_m.append([1]*len(s))

            if tname == "reason":
                think_list = ex.get("thinking", []) or []
                think_body = "\n".join([x for x in think_list if isinstance(x, str) and x.strip()])
                ans_text = ex.get("answer", "") or ""

                if self.wrap_reason_with_tags and think_body.strip():
                    think_text = f"{THINK_OPEN}{think_body}{THINK_CLOSE}\n\n"
                else:
                    think_text = ""

                think_ids = self._tok_ids(think_text) if think_text else []
                ans_only_ids = self._tok_ids(ans_text)

                lab = think_ids + ans_only_ids
                st = [True]*len(think_ids) + [False]*len(ans_only_ids)
                sl = [False]*len(think_ids) + [True]*len(ans_only_ids)
                su = [False]*len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id); st.append(False); sl.append(True); su.append(False)
                if len(lab) == 0:
                    lab = [self.eos_id]; st, sl, su = [False], [True], [False]

                ans_ids.append(lab); seg_think.append(st); seg_label.append(sl); seg_target.append(su)

            elif tname == "understand":
                tgt = ex.get("target", "") or ""
                lab = self._tok_ids(tgt)
                st = [False]*len(lab)
                sl = [False]*len(lab)
                su = [True]*len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id); st.append(False); sl.append(False); su.append(True)
                if len(lab) == 0:
                    lab = [self.eos_id]; st, sl, su = [False], [False], [True]

                ans_ids.append(lab); seg_think.append(st); seg_label.append(sl); seg_target.append(su)

            else:
                ans_text = ex.get("answer", "") or ""
                lab = self._tok_ids(ans_text)
                st = [False]*len(lab)
                sl = [True]*len(lab)
                su = [False]*len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id); st.append(False); sl.append(True); su.append(False)
                if len(lab) == 0:
                    lab = [self.eos_id]; st, sl, su = [False], [True], [False]

                ans_ids.append(lab); seg_think.append(st); seg_label.append(sl); seg_target.append(su)

        # ========== E) dynamic batch_max_chunks with image tokens in budget ==========
        max_pre_len = max((len(x) for x in pre_ids), default=0)
        max_suf_len = max((len(x) for x in suf_ids), default=0)
        max_ans_len = max((len(x) for x in ans_ids), default=1)

        raw_max_chunks = max((len(x) for x in chunk_txts), default=0)
        
        # Limit raw chunk count using max_chunks_per_sample
        if self.max_chunks_per_sample is not None and self.max_chunks_per_sample > 0:
            raw_max_chunks = min(raw_max_chunks, self.max_chunks_per_sample)

        raw_max_chunks = max(1, int(raw_max_chunks))

        img_block_len = (batch_max_images * self.image_token_factor) if batch_max_images > 0 else 0

        reserved = max_pre_len + max_suf_len + max_ans_len + img_block_len
        budget = self.max_len - reserved
        if budget <= 0:
            batch_max_chunks = 1
        else:
            batch_max_chunks = max(1, int(budget // self.chunk_token_factor))
            batch_max_chunks = min(batch_max_chunks, raw_max_chunks)

        for i in range(B):
            if len(chunk_txts[i]) > batch_max_chunks:
                chunk_txts[i] = chunk_txts[i][:batch_max_chunks]

        # ========== F) overflow trim (prefix -> suffix -> labels) ==========
        chunk_block_len = batch_max_chunks * self.chunk_token_factor
        for i in range(B):
            p_len, s_len, a_len = len(pre_ids[i]), len(suf_ids[i]), len(ans_ids[i])
            total = p_len + chunk_block_len + img_block_len + s_len + a_len
            overflow = max(0, total - self.max_len)
            if overflow > 0:
                cut = min(overflow, p_len)
                if cut > 0:
                    pre_ids[i], pre_m[i] = self._left_trim(pre_ids[i], pre_m[i], cut)
                    overflow -= cut
                if overflow > 0:
                    cut = min(overflow, s_len)
                    if cut > 0:
                        suf_ids[i], suf_m[i] = self._left_trim(suf_ids[i], suf_m[i], cut)
                        overflow -= cut
                if overflow > 0:
                    ans_ids[i], seg_think[i], seg_label[i], seg_target[i] = self._trim_labels_keep_eos(
                        ans_ids[i], seg_think[i], seg_label[i], seg_target[i], overflow
                    )

        # ========== G) padding ==========
        max_pre = max(1, max(len(x) for x in pre_ids)) if pre_ids else 1
        max_suf = max(1, max(len(x) for x in suf_ids)) if suf_ids else 1
        max_ans = max(1, max(len(x) for x in ans_ids)) if ans_ids else 1

        for i in range(B):
            dp = max_pre - len(pre_ids[i]); pre_ids[i] = [self.pad_id]*dp + pre_ids[i]; pre_m[i] = [0]*dp + pre_m[i]
            ds = max_suf - len(suf_ids[i]); suf_ids[i] = [self.pad_id]*ds + suf_ids[i]; suf_m[i] = [0]*ds + suf_m[i]
            da = max_ans - len(ans_ids[i])
            ans_ids[i]    = ans_ids[i]    + [self.lab_pad_id]*da
            seg_think[i]  = seg_think[i]  + [False]*da
            seg_label[i]  = seg_label[i]  + [False]*da
            seg_target[i] = seg_target[i] + [False]*da

        out: Dict[str, Any] = {
            "chunk_txts":  chunk_txts,
            "prefix_ids":  torch.tensor(pre_ids, dtype=torch.long),
            "prefix_mask": torch.tensor(pre_m,   dtype=torch.long),
            "suffix_ids":  torch.tensor(suf_ids, dtype=torch.long),
            "suffix_mask": torch.tensor(suf_m,   dtype=torch.long),
            "labels":      torch.tensor(ans_ids, dtype=torch.long),
            "task_ids":    torch.tensor(task_ids, dtype=torch.long),
        }

        if pixel_values is not None and n_images is not None:
            out["pixel_values"] = pixel_values
            out["n_images"] = n_images

        if self.use_seg_masks:
            out.update({
                "seg_think":  torch.tensor(seg_think,  dtype=torch.bool),
                "seg_label":  torch.tensor(seg_label,  dtype=torch.bool),
                "seg_target": torch.tensor(seg_target, dtype=torch.bool),
            })

        out["meta"] = [
            {"task": (ex.get("task", "") or ""), "target": (ex.get("target", "") or ""), "answer": (ex.get("answer", "") or "")}
            for ex in batch
        ]
        return out


DataCollatorTraffic = DataCollatorSmartCity

__all__ = [
    "load_jsonl_dataset",
    "save_dataset_jsonl",
    "ensure_smartcity_features",
    "DataCollatorSmartCity",
    "DataCollatorTraffic",
    "FIELD_ORDER", "TASK2ROUTER", "CHUNK_MARKER", "IMAGE_MARKER",
    "SMARTCITY_FEATURES",
    "PROMPT_STYLE_QWEN3", "PROMPT_STYLE_PLAIN",
]
