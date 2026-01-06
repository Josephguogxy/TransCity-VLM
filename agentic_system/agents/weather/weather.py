# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from agentic_system.schemas import PlanStep, WeatherHourly
from agentic_system.agents.base import ExecutionContext
from agentic_system.clients.open_meteo import OpenMeteoClient
from agentic_system.clients.cds import CdsEra5Client
from agentic_system.utils.time import (
    timezone_name_from_latlon,
    parse_local_datetime,
    local_to_utc,
)


@dataclass
class WeatherAgent:
    """
    Rule (your requirement):
      - if the whole query window is within now±7 days -> Open-Meteo
      - else -> CDS/ERA5
    """
    name: str = "WeatherAgent"
    recent_days: int = 7

    def __post_init__(self) -> None:
        self.open_meteo = OpenMeteoClient(timeout_s=15.0)
        self.cds = CdsEra5Client()

    @staticmethod
    def _window_within_now_plusminus_days(start_utc: datetime, end_utc: datetime, *, days: int) -> bool:
        now = datetime.now(timezone.utc)
        lo = now - timedelta(days=days)
        hi = now + timedelta(days=days)
        return (start_utc >= lo) and (end_utc <= hi)

    async def run(self, step: PlanStep, ctx: ExecutionContext) -> WeatherHourly:
        # ---- lat/lon MUST already be available here ----
        if ctx.request.lat is None or ctx.request.lon is None:
            raise RuntimeError("WeatherAgent requires ctx.request.lat/lon. Ensure plan runs geo_forward + normalize first.")

        lat = float(step.args.get("lat", ctx.request.lat))
        lon = float(step.args.get("lon", ctx.request.lon))

        if ctx.request.local_start is None or ctx.request.local_end is None:
            raise RuntimeError("WeatherAgent requires ctx.request.local_start/local_end. Ensure time_window + normalize ran.")

        local_start_s = str(step.args.get("local_start", ctx.request.local_start))
        local_end_s = str(step.args.get("local_end", ctx.request.local_end))

        recent_days = int(step.args.get("recent_days", self.recent_days))

        tz_name = timezone_name_from_latlon(lat, lon)
        ls = parse_local_datetime(local_start_s)  # naive local
        le = parse_local_datetime(local_end_s)    # naive local
        us = local_to_utc(ls, tz_name)            # aware utc
        ue = local_to_utc(le, tz_name)            # aware utc

        use_open_meteo = self._window_within_now_plusminus_days(us, ue, days=recent_days)

        # ---- Branch A: Open-Meteo (near-now) ----
        if use_open_meteo:
            try:
                return await self.open_meteo.fetch_hourly(
                    lat=lat,
                    lon=lon,
                    tz_name=tz_name,
                    local_start=ls,
                    local_end=le,
                    past_days=recent_days,
                    forecast_days=recent_days,
                )
            except Exception as e:
                print(f"[WARN] Open-Meteo failed, fallback to CDS. err={type(e).__name__}: {e}")

        # ---- Branch B: CDS/ERA5 (default) ----
        def _cds_fetch():
            t_c, ws, wd = self.cds.fetch_hourly_cached(lat=lat, lon=lon, start_utc=us, end_utc=ue)
            return WeatherHourly(
                start_local=ls.isoformat(sep=" "),
                end_local=le.isoformat(sep=" "),
                temperature_c=list(t_c),
                wind_speed_ms=list(ws),
                wind_direction_deg=list(wd),
            )

        return await asyncio.to_thread(_cds_fetch)
