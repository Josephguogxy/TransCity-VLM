# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Dict

from traffic_service.agents.base import ActionAgent
from traffic_service.core.context import ExecutionContext


def _get(d: Any, k: str, default=None):
    if isinstance(d, dict):
        return d.get(k, default)
    return getattr(d, k, default)


class RequestNormalizerAgent(ActionAgent):
    """
    Fill missing fields without overwriting user inputs; use ctx.update_request().
    """
    name = "RequestNormalizerAgent"

    async def apply_inferred_fields(
        self,
        ctx: ExecutionContext,
        *,
        parsed_key: str = "parsed_nl",
        geo_key: str = "geo",
        time_key: str = "time_window",
    ) -> Dict[str, Any]:
        parsed = ctx.get(parsed_key, {}) or {}
        geo = ctx.get(geo_key, {}) or {}
        tw = ctx.get(time_key, {}) or {}

        updates: Dict[str, Any] = {}

        if getattr(ctx.request, "lat", None) is None and _get(geo, "lat") is not None:
            updates["lat"] = float(_get(geo, "lat"))
        if getattr(ctx.request, "lon", None) is None and _get(geo, "lon") is not None:
            updates["lon"] = float(_get(geo, "lon"))

        if getattr(ctx.request, "local_start", None) is None and _get(tw, "local_start"):
            updates["local_start"] = str(_get(tw, "local_start"))
        if getattr(ctx.request, "local_end", None) is None and _get(tw, "local_end"):
            updates["local_end"] = str(_get(tw, "local_end"))

        road_ref = _get(parsed, "road_ref")
        direction_hint = _get(parsed, "direction_hint")

        if getattr(ctx.request, "freeway", None) is None and road_ref:
            updates["freeway"] = str(road_ref)
        if getattr(ctx.request, "direction", None) is None and direction_hint:
            updates["direction"] = str(direction_hint)

        ctx.update_request(updates)

        return {
            "updated": updates,
            "time_source": _get(tw, "source"),
            "tz_name": _get(tw, "tz_name"),
            "geo_display_name": _get(geo, "display_name"),
        }
