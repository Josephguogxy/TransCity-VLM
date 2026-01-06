# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from agentic_system.agents.base import ActionAgent
from agentic_system.core.context import ExecutionContext

from agentic_system.db.flow_repo import fetch_all_sensors


def _haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371000.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * (np.sin(dlmb / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))


class NearestSensorAgent(ActionAgent):
    name = "NearestSensorAgent"

    def __init__(self) -> None:
        self._loaded = False
        self._rows: List[Dict[str, Any]] = []
        self._lat: Optional[np.ndarray] = None
        self._lon: Optional[np.ndarray] = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rows = await asyncio.to_thread(fetch_all_sensors)
        if not rows:
            raise RuntimeError("No sensors in DB. Check sensors table / import.")
        self._rows = rows
        self._lat = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
        self._lon = np.array([float(r["lon"]) for r in rows], dtype=np.float64)
        self._loaded = True

    async def match(self, ctx: ExecutionContext) -> Dict[str, Any]:
        await self._ensure_loaded()

        if ctx.request.lat is None or ctx.request.lon is None:
            raise RuntimeError("NearestSensorAgent requires ctx.request.lat/lon (should be filled before this step).")

        lat = float(ctx.request.lat)
        lon = float(ctx.request.lon)

        d = _haversine_m(lat, lon, self._lat, self._lon)  # type: ignore[arg-type]
        i = int(np.argmin(d))
        best = dict(self._rows[i])
        best["distance_m"] = float(d[i])

        updates = {}
        if getattr(ctx.request, "sensor_id", None) is None:
            code = best.get("sensor_code")
            updates["sensor_id"] = str(code) if code else str(best["sensor_idx"])
        if getattr(ctx.request, "freeway", None) is None and best.get("fwy"):
            updates["freeway"] = str(best["fwy"])
        if getattr(ctx.request, "direction", None) is None and best.get("direction"):
            updates["direction"] = str(best["direction"])
        if getattr(ctx.request, "lanes", None) is None and best.get("lanes") is not None:
            updates["lanes"] = int(best["lanes"])

        ctx.update_request(updates)
        return best
