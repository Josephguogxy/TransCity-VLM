# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.clients.overpass import OverpassClient
from traffic_service.core.context import ExecutionContext


class RoadContextAgent(ActionAgent):
    name = "RoadContextAgent"

    def __init__(self, client: Optional[OverpassClient] = None) -> None:
        if client is None:
            try:
                from traffic_service.config import settings
                base = getattr(settings, "overpass_url", "https://overpass-api.de/api/interpreter")
            except Exception:
                base = "https://overpass-api.de/api/interpreter"
            endpoints = [
                base,
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.nchc.org.tw/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter",
            ]
            client = OverpassClient(endpoints=endpoints, timeout_s=60, max_retries=2)
        self.client = client

    async def find_nearby_roads(
        self,
        ctx: ExecutionContext,
        *,
        lat: float | None = None,
        lon: float | None = None,
        radius_m: int = 1500,
        soft_fail: bool = True,
    ) -> list[dict[str, Any]]:
        lat2 = float(lat if lat is not None else ctx.request.lat)
        lon2 = float(lon if lon is not None else ctx.request.lon)

        q = f"""
[out:json][timeout:60];
(
  way(around:{int(radius_m)},{lat2},{lon2})[highway~"^(motorway|trunk|primary)$"];
);
out tags center 30;
""".strip()

        try:
            data = await self.client.post(q)
            roads = []
            for el in (data.get("elements", []) or []):
                tags = el.get("tags", {}) or {}
                roads.append(
                    {
                        "name": tags.get("name"),
                        "ref": tags.get("ref"),
                        "highway": tags.get("highway"),
                        "lanes": tags.get("lanes"),
                        "center": el.get("center"),
                    }
                )

            seen = set()
            out = []
            for r in roads:
                key = (r.get("ref"), r.get("name"), r.get("highway"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            return out

        except Exception as e:
            if soft_fail:
                print(f"[WARN] RoadContextAgent failed -> return []. {type(e).__name__}: {e}")
                return []
            raise
