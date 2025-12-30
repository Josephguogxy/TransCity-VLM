# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class ExampleMeta:
    task: str
    target: str
    answer: str

_TAGS = ["</think>", "<answer>", "</answer>"]

_ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)

def extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if not m:
        return text.strip()
    return m.group(1).strip()

def soft_format_reward(text: str) -> float:
    """
    A lenient formatting reward to avoid an all-zero signal early in training.

    Assumes the prompt already pre-fills ``<think>``, so we mainly reward:
    - closing ``</think>``
    - producing ``<answer>...</answer>`` tags
    """
    r = 0.0
    for t in _TAGS:
        if t in text:
            r += 1.0 / len(_TAGS)
    return r

def accuracy_reward(text: str, meta: ExampleMeta) -> float:
    """
    Customize this to your use case. This is a minimal, extensible baseline:

    - prediction: use ``meta.answer`` as the ground-truth (or ``meta.target``), then do string / numeric matching
    - understand: return 0 for now (structure-only warmup)
    - reason: same as prediction (extract from the ``<answer>`` tag)
    """
    pred = extract_answer(text)
    gt = (meta.answer or meta.target or "").strip()
    if not gt:
        return 0.0

    if meta.task.lower() in ("prediction", "reason"):
        # Minimal baseline: exact string match (replace with numeric tolerance / task-specific metric later).
        return 1.0 if pred.strip() == gt else 0.0

    return 0.0

def compute_rewards(
    completions: List[str],
    metas_rep: List[ExampleMeta],
    *,
    w_format: float = 1.0,
    w_acc: float = 1.0,
) -> List[float]:
    rs: List[float] = []
    for text, meta in zip(completions, metas_rep):
        r = w_format * soft_format_reward(text) + w_acc * accuracy_reward(text, meta)
        rs.append(float(r))
    return rs
