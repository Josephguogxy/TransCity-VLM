# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import httpx

from traffic_service.config import get_settings, LLMSettings


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMChatMessage:
    role: str
    content: str


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    m = _CODE_FENCE_RE.search(s)
    if m:
        return (m.group(1) or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            return (parts[1] or "").strip()
    return s


def _extract_first_json_fragment(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None

    starts = [i for i in [s.find("{"), s.find("[")] if i != -1]
    if not starts:
        return None
    start = min(starts)

    stack: List[str] = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                continue
            top = stack[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                stack.pop()
                if not stack:
                    return s[start : i + 1]

    return None


def _repair_json(text: str) -> str:
    s = (text or "").strip()
    s = s.lstrip("\ufeff")
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r'(:\s*(?:null|true|false))\s*"', r"\1", s, flags=re.IGNORECASE)
    return s


class OpenAICompatibleClient:
    """
    Minimal OpenAI-style Chat Completions client (sync).
    You already call it via asyncio.to_thread(...) in agents, so sync is fine.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        default_model: str = "qwen3",
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.default_model = default_model

    def _chat_completions_url(self) -> str:
        b = self.base_url
        if b.endswith("/v1"):
            return f"{b}/chat/completions"
        if b.endswith("/v1/"):
            return f"{b.rstrip('/')}/chat/completions"
        return f"{b}/v1/chat/completions"

    def chat(
        self,
        messages: List[LLMChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = self._chat_completions_url()

        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra:
            payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Unexpected LLM response shape: {data}") from e

    def chat_json(
        self,
        messages: List[LLMChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        txt = self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, extra=extra)

        cleaned = _strip_code_fences(txt)
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        frag = _extract_first_json_fragment(cleaned) or cleaned

        try:
            obj = json.loads(frag)
            if not isinstance(obj, dict):
                raise LLMError(f"LLM JSON is not an object: {type(obj).__name__}")
            return obj
        except Exception:
            pass

        frag2 = _repair_json(frag)
        try:
            obj2 = json.loads(frag2)
            if not isinstance(obj2, dict):
                raise LLMError(f"LLM JSON is not an object after repair: {type(obj2).__name__}")
            return obj2
        except Exception as e:
            raise LLMError(f"Failed to parse JSON from LLM. Raw head: {(txt or '')[:800]}") from e


LLMRole = Literal["planner", "extractor", "verifier"]


def client_from_settings(cfg: LLMSettings) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        default_model=cfg.model,
    )


def client_for(role: LLMRole) -> OpenAICompatibleClient:
    s = get_settings()
    if role == "planner":
        cfg = s.planner_llm
    elif role == "extractor":
        cfg = s.extractor_llm
    elif role == "verifier":
        cfg = s.verifier_llm
    else:
        raise ValueError(f"Unknown LLM role: {role}")

    return client_from_settings(cfg)
