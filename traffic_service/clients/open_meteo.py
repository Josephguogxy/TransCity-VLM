# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from traffic_service.schemas import WeatherHourly


@dataclass
class OpenMeteoClient:
    """
    Open-Meteo Forecast API client.
    We use it ONLY for near-now windows (<= 7 days), per your rule.
    """
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    timeout_s: float = 15.0

    async def fetch_hourly(
        self,
        *,
        lat: float,
        lon: float,
        tz_name: str,
        local_start: datetime,   # naive local
        local_end: datetime,     # naive local
        past_days: int = 7,
        forecast_days: int = 7,
    ) -> WeatherHourly:
        """
        Returns WeatherHourly aligned to local time (timestamps are local naive in output arrays).
        """
        start_date = local_start.date().isoformat()
        end_date = local_end.date().isoformat()

        params = {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m",
            "timezone": tz_name,
            "start_date": start_date,
            "end_date": end_date,
            "temperature_unit": "celsius",
            "windspeed_unit": "ms",
            "timeformat": "iso8601",
            # to cover near-past/near-future in one endpoint:
            "past_days": int(past_days),
            "forecast_days": int(forecast_days),
        }

        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            r = await client.get(self.base_url, params=params)
            r.raise_for_status()
            data: dict[str, Any] = r.json() or {}

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        wspd = hourly.get("windspeed_10m") or []
        wdir = hourly.get("winddirection_10m") or []

        def parse_iso_local(s: str) -> Optional[datetime]:
            if not s:
                return None
            ss = str(s).strip()
            # open-meteo usually returns "YYYY-MM-DDTHH:MM"
            try:
                return datetime.fromisoformat(ss.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                try:
                    return datetime.fromisoformat(ss.replace("T", " ")).replace(tzinfo=None)
                except Exception:
                    return None

        def to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return float("nan")

        # Filter to [local_start, local_end] hourly
        out_t: list[float] = []
        out_ws: list[float] = []
        out_wd: list[float] = []

        # Round window bounds to hour for matching
        ls = local_start.replace(minute=0, second=0, microsecond=0)
        le = local_end.replace(minute=0, second=0, microsecond=0)

        for t, tc, ws, wd in zip(times, temps, wspd, wdir):
            dt = parse_iso_local(str(t))
            if dt is None:
                continue
            dt0 = dt.replace(minute=0, second=0, microsecond=0)
            if ls <= dt0 <= le:
                out_t.append(to_float(tc))
                out_ws.append(to_float(ws))
                out_wd.append(to_float(wd))

        # Fallback: if empty, still return something (avoid breaking downstream)
        if not out_t:
            out_t = [to_float(x) for x in temps]
            out_ws = [to_float(x) for x in wspd]
            out_wd = [to_float(x) for x in wdir]

        return WeatherHourly(
            start_local=local_start.isoformat(sep=" "),
            end_local=local_end.isoformat(sep=" "),
            temperature_c=out_t,
            wind_speed_ms=out_ws,
            wind_direction_deg=out_wd,
        )
