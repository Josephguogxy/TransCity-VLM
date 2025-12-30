# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from traffic_service.clients.http import request_json


WORLDPOP_MIN_YEAR = 2000
WORLDPOP_MAX_YEAR = 2020


def clamp_worldpop_year(year: int) -> Tuple[int, str]:
    """
    WorldPop wpgppop docs list 2000-2020; clamp to avoid hard failures.
    """
    y = int(year)
    if y < WORLDPOP_MIN_YEAR:
        return WORLDPOP_MIN_YEAR, f"clamped_year:{y}->{WORLDPOP_MIN_YEAR}"
    if y > WORLDPOP_MAX_YEAR:
        return WORLDPOP_MAX_YEAR, f"clamped_year:{y}->{WORLDPOP_MAX_YEAR}"
    return y, ""


def make_circle_geojson_featurecollection(
    *,
    lat: float,
    lon: float,
    radius_km: float,
    points: int = 72,
) -> Dict[str, Any]:
    """
    Build an approximate circle as a GeoJSON polygon (WGS84).
    Small radii (1-20km) are acceptable for density features.

    WorldPop stats expects the geojson param as a JSON string.
    """
    lat = float(lat)
    lon = float(lon)
    r_km = float(radius_km)

    # 1 deg lat ≈ 111.32km
    lat_rad = math.radians(lat)
    dlat = r_km / 111.32
    dlon = r_km / (111.32 * max(1e-6, math.cos(lat_rad)))

    pts = max(12, int(points))
    ring = []
    for i in range(pts):
        theta = 2.0 * math.pi * i / (pts - 1)  # last point closes the ring
        x = lon + dlon * math.cos(theta)
        y = lat + dlat * math.sin(theta)
        ring.append([x, y])

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


@dataclass
class WorldPopStats:
    total_population: Optional[float]
    source_url: str
    notes: str


class WorldPopClient:
    """
    WorldPop Advanced Data API (v1):
      - stats: https://api.worldpop.org/v1/services/stats
      - tasks: https://api.worldpop.org/v1/tasks/{taskid}
    """

    BASE = "https://api.worldpop.org/v1"

    def __init__(self, *, api_key: Optional[str] = None, dataset: str = "wpgppop") -> None:
        self.api_key = api_key
        self.dataset = dataset

    async def total_population_stats(
        self,
        *,
        year: int,
        geojson_obj: Dict[str, Any],
        runasync: bool = False,
        timeout_s: float = 35.0,
        poll_timeout_s: float = 45.0,
        poll_interval_s: float = 0.8,
        debug: bool = False,
    ) -> WorldPopStats:
        """
        Return total_population within a GeoJSON area.

        - runasync=false: server tries to respond synchronously (~30s);
          on timeout it returns status=created + taskid for polling.
        """
        geojson_str = json.dumps(geojson_obj, ensure_ascii=False, separators=(",", ":"))

        params: Dict[str, Any] = {
            "dataset": self.dataset,
            "year": str(int(year)),
            "geojson": geojson_str,
            "runasync": "true" if runasync else "false",
        }
        if self.api_key:
            params["key"] = self.api_key

        url = f"{self.BASE}/services/stats"
        data, final_url = await request_json(url, params=params, timeout_s=timeout_s, debug=debug)

        if data is None:
            return WorldPopStats(total_population=None, source_url=str(final_url), notes="worldpop_empty_response")

        if not isinstance(data, dict):
            return WorldPopStats(total_population=None, source_url=str(final_url), notes="worldpop_unexpected_payload")

        if bool(data.get("error")):
            return WorldPopStats(
                total_population=None,
                source_url=str(final_url),
                notes=f"worldpop_error:{data.get('error_message')}",
            )

        status = str(data.get("status") or "").lower()

        if status == "finished":
            pop = _extract_total_population(data)
            return WorldPopStats(
                total_population=pop,
                source_url=str(final_url),
                notes="worldpop_finished_sync",
            )

        if status == "created":
            taskid = str(data.get("taskid") or "").strip()
            if not taskid:
                return WorldPopStats(total_population=None, source_url=str(final_url), notes="worldpop_created_no_taskid")
            return await self._poll_task(
                taskid=taskid,
                timeout_s=timeout_s,
                poll_timeout_s=poll_timeout_s,
                poll_interval_s=poll_interval_s,
                debug=debug,
            )

        if status:
            taskid = str(data.get("taskid") or "").strip()
            if taskid:
                return await self._poll_task(
                    taskid=taskid,
                    timeout_s=timeout_s,
                    poll_timeout_s=poll_timeout_s,
                    poll_interval_s=poll_interval_s,
                    debug=debug,
                )
            return WorldPopStats(total_population=None, source_url=str(final_url), notes=f"worldpop_status:{status}")

        return WorldPopStats(total_population=None, source_url=str(final_url), notes="worldpop_no_status")

    async def _poll_task(
        self,
        *,
        taskid: str,
        timeout_s: float,
        poll_timeout_s: float,
        poll_interval_s: float,
        debug: bool = False,
    ) -> WorldPopStats:
        """
        Poll task results: GET https://api.worldpop.org/v1/tasks/{taskid}
        """
        task_url = f"{self.BASE}/tasks/{taskid}"
        deadline = time.monotonic() + float(poll_timeout_s)

        last_notes = ""
        while True:
            data, final_url = await request_json(task_url, timeout_s=timeout_s, debug=debug)

            if data is None:
                last_notes = "worldpop_task_empty"
            elif isinstance(data, dict):
                if bool(data.get("error")):
                    return WorldPopStats(
                        total_population=None,
                        source_url=str(final_url),
                        notes=f"worldpop_task_error:{data.get('error_message')}",
                    )

                status = str(data.get("status") or "").lower()
                if status == "finished":
                    pop = _extract_total_population(data)
                    return WorldPopStats(
                        total_population=pop,
                        source_url=str(final_url),
                        notes="worldpop_finished_task",
                    )

                last_notes = f"worldpop_task_status:{status or 'unknown'}"
            else:
                last_notes = "worldpop_task_unexpected_payload"

            if time.monotonic() >= deadline:
                return WorldPopStats(
                    total_population=None,
                    source_url=str(task_url),
                    notes=f"worldpop_task_timeout:{last_notes}",
                )

            await asyncio.sleep(float(poll_interval_s))


def _extract_total_population(payload: Dict[str, Any]) -> Optional[float]:
    """
    Extract data.total_population from WorldPop payload.
    """
    try:
        d = payload.get("data") or {}
        v = d.get("total_population", None)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None
