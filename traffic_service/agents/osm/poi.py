# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.clients.overpass import OverpassClient
from traffic_service.core.context import ExecutionContext
from traffic_service.schemas import POISummary


class POIAgent(ActionAgent):
    name = "POIAgent"

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

    def _classify(self, tags: dict) -> str:
        amenity = tags.get("amenity")
        shop = tags.get("shop")
        tourism = tags.get("tourism")
        highway = tags.get("highway")

        if amenity in {"fuel"}:
            return "fuel"
        if amenity in {"fast_food"}:
            return "fast_food"
        if amenity in {"restaurant", "cafe"}:
            return "food"
        if amenity in {"hospital", "clinic", "doctors"}:
            return "healthcare"
        if amenity in {"school", "university", "college"}:
            return "education"
        if highway == "services":
            return "highway_services"
        if shop:
            return "shop"
        if tourism:
            return "tourism"
        return "other_poi"

    async def fetch_poi_summary(
        self,
        ctx: ExecutionContext,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: float = 3.0,
        soft_fail: bool = True,
    ) -> POISummary:
        lat2 = float(lat if lat is not None else ctx.request.lat)
        lon2 = float(lon if lon is not None else ctx.request.lon)
        radius_m = int(float(radius_km) * 1000)

        q_a = f"""
[out:json][timeout:60];
(
  node(around:{radius_m},{lat2},{lon2})[amenity];
  way(around:{radius_m},{lat2},{lon2})[amenity];
  relation(around:{radius_m},{lat2},{lon2})[amenity];

  node(around:{radius_m},{lat2},{lon2})[highway="services"];
  way(around:{radius_m},{lat2},{lon2})[highway="services"];
);
out tags 300;
""".strip()

        q_b = f"""
[out:json][timeout:60];
(
  node(around:{radius_m},{lat2},{lon2})[shop];
  way(around:{radius_m},{lat2},{lon2})[shop];

  node(around:{radius_m},{lat2},{lon2})[tourism];
  way(around:{radius_m},{lat2},{lon2})[tourism];
);
out tags 300;
""".strip()

        counts: dict[str, int] = {}

        try:
            for q in (q_a, q_b):
                try:
                    data = await self.client.post(q)
                except Exception:
                    continue
                for el in (data.get("elements", []) or []):
                    tags = el.get("tags", {}) or {}
                    cat = self._classify(tags)
                    counts[cat] = counts.get(cat, 0) + 1

            return POISummary(
                radius_km=float(radius_km),
                counts_by_category=dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
            )

        except Exception as e:
            if soft_fail:
                print(f"[WARN] POIAgent failed -> empty. {type(e).__name__}: {e}")
                return POISummary(radius_km=float(radius_km), counts_by_category={})
            raise
