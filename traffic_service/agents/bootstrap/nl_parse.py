# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.core.context import ExecutionContext
from traffic_service.llm.clients import OpenAICompatibleClient, LLMChatMessage, client_for
from traffic_service.utils.time import parse_local_datetime

from traffic_service.llm.prompts import nl_parse_system, nl_parse_user

ROAD_REF_RE = re.compile(
    r"\b(?:i|interstate|us|u\.s\.|sr|state route|route|hwy|highway|ca)\s*[- ]?\s*\d+[a-z]?\b",
    re.IGNORECASE,
)
DIR_RE = re.compile(r"\b(northbound|southbound|eastbound|westbound|nb|sb|eb|wb)\b", re.IGNORECASE)
DIR_NORMALIZE = {"nb": "northbound", "sb": "southbound", "eb": "eastbound", "wb": "westbound"}

_ALLOWED_TIME_KIND = {"none", "point", "range"}


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class NLRequestParserAgent(ActionAgent):
    """
    Return a stable dict with all fields; no guessing in fallback.
    """
    name = "NLRequestParserAgent"

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 900,
        debug: bool = False,
    ) -> None:
        self.llm = llm or client_for("planner")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.debug = bool(debug)

    async def _call_llm(self, *, sys: str, user: str) -> Dict[str, Any]:
        data = await asyncio.to_thread(
            self.llm.chat_json,
            [LLMChatMessage("system", sys), LLMChatMessage("user", user)],
            model=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a dict")
        return data

    async def parse_nl(self, ctx: ExecutionContext, *, strict: bool = True) -> Dict[str, Any]:
        q = str(ctx.request.question)
        rt = getattr(ctx.request, "request_time_utc", None)

        sys = nl_parse_system()

        data: Dict[str, Any]
        err: Optional[Exception] = None
        try:
            user = nl_parse_user(question=q, request_time_utc=rt, refine=False)
            data = await self._call_llm(sys=sys, user=user)

            if "user_location_query" not in data and "primary_location_query" not in data:
                raise ValueError("LLM output missing user_location_query/primary_location_query")
        except Exception as e:
            err = e
            data = {}

        if not data:
            try:
                user2 = nl_parse_user(question=q, request_time_utc=rt, refine=True)
                data = await self._call_llm(sys=sys, user=user2)
                if "user_location_query" not in data and "primary_location_query" not in data:
                    raise ValueError("LLM refine output missing user_location_query/primary_location_query")
                err = None
            except Exception as e2:
                err = e2
                data = {}

        if not data:
            if strict:
                raise RuntimeError(f"NLRequestParserAgent failed (strict=true). {type(err).__name__}: {err}")

            data = {
                "user_location_query": None,
                "route_location_query": None,
                "destination_location_query": None,
                "road_ref": None,
                "direction_hint": None,
                "time_kind": "none",
                "time_point_local": None,
                "time_start_local": None,
                "time_end_local": None,
                "time_text": None,
                "time_confidence": 0.0,
                "confidence": 0.0,
                "notes": f"fallback_used: {type(err).__name__}",
                "primary_location_query": None,
            }

        # -----------------------------
        # Normalize output fields.
        # -----------------------------
        out: Dict[str, Any] = {}

        user_loc = _as_str(data.get("user_location_query")) or _as_str(data.get("primary_location_query"))
        out["user_location_query"] = user_loc
        out["primary_location_query"] = user_loc

        out["route_location_query"] = _as_str(data.get("route_location_query"))
        out["destination_location_query"] = _as_str(data.get("destination_location_query"))

        out["road_ref"] = _as_str(data.get("road_ref"))
        out["direction_hint"] = _as_str(data.get("direction_hint"))

        tk = (_as_str(data.get("time_kind")) or "none").lower()
        if tk not in _ALLOWED_TIME_KIND:
            tk = "none"
        out["time_kind"] = tk
        out["time_point_local"] = _as_str(data.get("time_point_local"))
        out["time_start_local"] = _as_str(data.get("time_start_local"))
        out["time_end_local"] = _as_str(data.get("time_end_local"))
        out["time_text"] = _as_str(data.get("time_text"))

        out["time_confidence"] = float(data.get("time_confidence") or 0.0)
        out["confidence"] = float(data.get("confidence") or 0.0)
        out["notes"] = _as_str(data.get("notes"))

        # -----------------------------
        # Regex fallback from the question text (not a guess).
        # -----------------------------
        if not out["road_ref"]:
            m = ROAD_REF_RE.search(q)
            if m:
                out["road_ref"] = m.group(0).strip()

        if not out["direction_hint"]:
            m = DIR_RE.search(q)
            if m:
                raw = m.group(0).strip().lower()
                out["direction_hint"] = DIR_NORMALIZE.get(raw, raw)

        # -----------------------------
        # Validate time fields; downgrade on invalid without filling values.
        # -----------------------------
        def _valid_dt(s: Optional[str]) -> bool:
            if not s:
                return False
            try:
                _ = parse_local_datetime(s)
                return True
            except Exception:
                return False

        if out["time_kind"] == "range":
            if not (_valid_dt(out["time_start_local"]) and _valid_dt(out["time_end_local"])):
                out["notes"] = ((out["notes"] or "") + " | invalid_range_time -> downgraded_to_none").strip(" |")
                out["time_kind"] = "none"
                out["time_start_local"] = None
                out["time_end_local"] = None

        if out["time_kind"] == "point":
            if not _valid_dt(out["time_point_local"]):
                out["notes"] = ((out["notes"] or "") + " | invalid_point_time -> downgraded_to_none").strip(" |")
                out["time_kind"] = "none"
                out["time_point_local"] = None

        if self.debug:
            print("[NLRequestParserAgent] out=", json.dumps(out, ensure_ascii=False, indent=2))

        return out
