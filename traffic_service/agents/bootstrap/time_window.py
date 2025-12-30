# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.core.context import ExecutionContext
from traffic_service.utils.time import (
    timezone_name_from_latlon,
    parse_local_datetime,
    local_to_utc,
)


def _parse_request_time_utc(s: Optional[str]) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    ss = str(s).strip()
    try:
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class TimeWindowResolverAgent(ActionAgent):
    name = "TimeWindowResolverAgent"

    async def resolve_time_window(
        self,
        ctx: ExecutionContext,
        *,
        parsed_key: str = "parsed_nl",
        geo_key: str = "geo",
        default_past_hours: int = 4,
        default_future_hours: int = 2,
    ) -> Dict[str, Any]:
        parsed = ctx.require(parsed_key) or {}
        geo = ctx.require(geo_key) or {}

        lat = float(geo["lat"])
        lon = float(geo["lon"])
        tz_name = timezone_name_from_latlon(lat, lon)

        from zoneinfo import ZoneInfo

        rt_utc = _parse_request_time_utc(getattr(ctx.request, "request_time_utc", None))
        local_now = rt_utc.astimezone(ZoneInfo(tz_name)).replace(tzinfo=None)

        def fmt(dt: datetime) -> str:
            return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

        time_kind = (_as_str(parsed.get("time_kind")) or "none").lower()
        tp = _as_str(parsed.get("time_point_local"))
        ts = _as_str(parsed.get("time_start_local"))
        te = _as_str(parsed.get("time_end_local"))

        def to_utc_iso(local_str: str) -> str:
            dt_local = parse_local_datetime(local_str)
            dt_utc = local_to_utc(dt_local, tz_name)
            return dt_utc.isoformat().replace("+00:00", "Z")

        if time_kind == "range" and ts and te:
            start_dt = parse_local_datetime(ts)
            end_dt = parse_local_datetime(te)

            notes = "Used LLM-parsed time range."
            if end_dt < start_dt:
                start_dt, end_dt = end_dt, start_dt
                notes += " | swapped_start_end"

            local_start = fmt(start_dt)
            local_end = fmt(end_dt)

            return {
                "tz_name": tz_name,
                "local_start": local_start,
                "local_end": local_end,
                "utc_start": to_utc_iso(local_start),
                "utc_end": to_utc_iso(local_end),
                "source": "question_range",
                "notes": notes,
            }

        if time_kind == "point" and tp:
            t0 = parse_local_datetime(tp)
            start = t0 - timedelta(hours=int(default_past_hours))
            end = t0 + timedelta(hours=int(default_future_hours))

            local_start = fmt(start)
            local_end = fmt(end)

            return {
                "tz_name": tz_name,
                "local_start": local_start,
                "local_end": local_end,
                "utc_start": to_utc_iso(local_start),
                "utc_end": to_utc_iso(local_end),
                "source": "question_point",
                "notes": "Built window around LLM-parsed point time.",
            }

        if time_kind == "range" and (ts or te):
            anchor = ts or te
            try:
                t0 = parse_local_datetime(str(anchor))
                start = t0 - timedelta(hours=int(default_past_hours))
                end = t0 + timedelta(hours=int(default_future_hours))

                local_start = fmt(start)
                local_end = fmt(end)

                return {
                    "tz_name": tz_name,
                    "local_start": local_start,
                    "local_end": local_end,
                    "utc_start": to_utc_iso(local_start),
                    "utc_end": to_utc_iso(local_end),
                    "source": "question_point",
                    "notes": "Range incomplete; treated as point window.",
                }
            except Exception:
                pass

        start = local_now - timedelta(hours=int(default_past_hours))
        end = local_now + timedelta(hours=int(default_future_hours))

        local_start = fmt(start)
        local_end = fmt(end)

        return {
            "tz_name": tz_name,
            "local_start": local_start,
            "local_end": local_end,
            "utc_start": to_utc_iso(local_start),
            "utc_end": to_utc_iso(local_end),
            "source": "default_now",
            "notes": "No usable time parsed; used now-4h~now+2h.",
        }
