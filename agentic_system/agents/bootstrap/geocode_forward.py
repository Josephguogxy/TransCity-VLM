# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from agentic_system.agents.base import ActionAgent
from agentic_system.clients.nominatim import NominatimClient
from agentic_system.core.context import ExecutionContext


ROAD_REF_RE = re.compile(
    r"\b(?:i|interstate|us|u\.s\.|sr|state route|route|hwy|highway|ca)\s*[- ]?\s*\d+[a-z]?\b",
    re.IGNORECASE,
)
DIR_RE = re.compile(r"\b(northbound|southbound|eastbound|westbound|nb|sb|eb|wb)\b", re.IGNORECASE)
FILLER_RE = re.compile(r"\b(on|at|near|in|by|around|between|towards|to)\b", re.IGNORECASE)
CONNECTORS = [" at ", " near ", " in ", " by ", " around ", " between ", " outside ", " inside ", " close to "]


def _normalize_spaces(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _tail_context_from_destination(dest: str, max_parts: int = 3) -> str:
    """
    Return only broader admin context from destination (state/country),
    and NEVER the destination place itself.

    Examples:
      "downtown Los Angeles, California, United States" -> "California, United States"
      "Los Angeles, California, United States" -> "California, United States"
      "Waltz, Santa Clarita, Los Angeles County, California, United States" -> "Los Angeles County, California, United States"
    """
    parts = [p.strip() for p in (dest or "").split(",") if p.strip()]
    if len(parts) <= 1:
        return ""
    tail = parts[1:]
    k = min(int(max_parts), len(tail))
    return ", ".join(tail[-k:])


class ForwardGeocodeAgent(ActionAgent):
    name = "ForwardGeocodeAgent"

    def __init__(
        self,
        client: Optional[NominatimClient] = None,
        *,
        per_query_limit: int = 5,
        debug: bool = False,
    ) -> None:
        if client is None:
            try:
                from agentic_system.config import settings
                reverse_url = getattr(settings, "nominatim_url", "https://nominatim.openstreetmap.org/reverse")
                timeout_s = float(getattr(settings, "http_timeout_s", 30.0))
            except Exception:
                reverse_url = "https://nominatim.openstreetmap.org/reverse"
                timeout_s = 30.0
            client = NominatimClient(reverse_url=reverse_url, timeout_s=timeout_s)

        self.client = client
        self.per_query_limit = int(per_query_limit)
        self.debug = bool(debug)

    def _build_candidates(
        self,
        primary: str,
        *,
        context_place: Optional[str] = None,
    ) -> List[str]:
        """
        primary: user_location_query (user/org location).
        context_place: optional context (route/destination); use admin tail for disambiguation.
        """
        q = _normalize_spaces(primary)
        ctx_place = _normalize_spaces(context_place) if context_place else ""

        cands: List[str] = []
        if q:
            cands.append(q)

        parts = [p.strip() for p in (q or "").split(",") if p.strip()]
        if len(parts) >= 2:
            drop_first = ", ".join(parts[1:])
            if drop_first:
                cands.append(drop_first)
            if len(parts) >= 3:
                cands.append(", ".join(parts[-3:]))

        low = q.lower()
        for sep in CONNECTORS:
            idx = low.find(sep)
            if idx >= 0:
                rhs = q[idx + len(sep):].strip()
                if rhs:
                    cands.append(rhs)

        cleaned = ROAD_REF_RE.sub(" ", q)
        cleaned = DIR_RE.sub(" ", cleaned)
        cleaned = FILLER_RE.sub(" ", cleaned)
        cleaned = _normalize_spaces(cleaned)
        if cleaned and cleaned.lower() != q.lower():
            cands.append(cleaned)

        ctx_tail = _tail_context_from_destination(ctx_place, max_parts=3) if ctx_place else ""
        if ctx_tail:
            for base in [q, cleaned]:
                base2 = _normalize_spaces(base)
                if base2 and ctx_tail.lower() not in base2.lower():
                    cands.append(f"{base2}, {ctx_tail}")

        seen = set()
        out: List[str] = []
        for s in cands:
            s2 = _normalize_spaces(s)
            if not s2:
                continue
            k = s2.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s2)
        return out

    async def geocode_user_location(self, ctx: ExecutionContext, *, parsed_key: str = "parsed_nl") -> Dict[str, Any]:
        parsed = ctx.require(parsed_key) or {}

        user_q = str(
            parsed.get("user_location_query")
            or parsed.get("primary_location_query")
            or ""
        ).strip()

        route_q = parsed.get("route_location_query")
        route_q = str(route_q).strip() if route_q else None

        dest_q = parsed.get("destination_location_query")
        dest_q = str(dest_q).strip() if dest_q else None

        context_place = dest_q or route_q

        cands = self._build_candidates(user_q, context_place=context_place)
        if self.debug:
            print(f"[ForwardGeocodeAgent] user_location={user_q!r} candidates={cands}")

        last_err: Optional[Exception] = None

        for cand in cands:
            try:
                items = await self.client.search(cand, limit=self.per_query_limit)
                best = self._pick_best(items, cand)
                if best:
                    return {
                        "query": cand,
                        "lat": float(best["lat"]),
                        "lon": float(best["lon"]),
                        "display_name": str(best.get("display_name") or cand),
                        "confidence": float(best.get("importance") or 0.7),
                        "source_url": str(getattr(self.client, "search_url", "")),
                    }
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"ForwardGeocodeAgent(user): no results. candidates={cands}. "
            f"last_err={type(last_err).__name__}: {last_err}"
        )

    async def geocode_primary(self, ctx: ExecutionContext, *, parsed_key: str = "parsed_nl") -> Dict[str, Any]:
        return await self.geocode_user_location(ctx, parsed_key=parsed_key)

    @staticmethod
    def _pick_best(items: List[dict], query: str) -> Optional[dict]:
        if not items:
            return None
        qlow = (query or "").lower()
        q_toks = [t for t in re.split(r"\W+", qlow) if t]

        def tok_match(name: str) -> int:
            n = (name or "").lower()
            return sum(1 for t in q_toks[:6] if t and t in n)

        def score(it: dict) -> float:
            imp = float(it.get("importance") or 0.0)
            cls = str(it.get("class") or "")
            typ = str(it.get("type") or "")
            name = str(it.get("display_name") or "")

            bonus = 0.0

            if cls in {"amenity", "shop", "office", "building", "tourism"}:
                bonus += 0.30
            if cls in {"place", "boundary"}:
                bonus += 0.10

            if typ in {"city", "town", "administrative"}:
                bonus -= 0.08

            m = tok_match(name)
            bonus += min(0.20, 0.05 * m)

            return imp + bonus

        items = list(items)
        items.sort(key=score, reverse=True)

        for it in items[:5]:
            cls = str(it.get("class") or "")
            name = str(it.get("display_name") or "")
            if cls in {"amenity", "shop", "office", "building"} and tok_match(name) >= 2:
                return it

        return items[0]
