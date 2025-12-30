# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.clients.census import CensusClient
from traffic_service.clients.worldpop import (
    WorldPopClient,
    clamp_worldpop_year,
    make_circle_geojson_featurecollection,
)
from traffic_service.core.context import ExecutionContext
from traffic_service.schemas import DemographicsSummary
from traffic_service.utils.geo import circle_area_km2


@dataclass
class _DemographicsAgg:
    radius_km: float
    pop_density: Optional[float]
    housing_density: Optional[float]
    notes: str


def _is_likely_us(lat: float, lon: float) -> bool:
    """
    Rough US check without reverse geocoding; used to decide Census fallback.
    Covers contiguous US + Alaska + Hawaii.
    """
    lat = float(lat)
    lon = float(lon)
    return (18.0 <= lat <= 72.0) and (-170.0 <= lon <= -50.0)


class DemographicsAgent(ActionAgent):
    name = "DemographicsAgent"

    def __init__(
        self,
        census: Optional[CensusClient] = None,
        worldpop: Optional[WorldPopClient] = None,
    ) -> None:
        # Census
        if census is None:
            try:
                from traffic_service.config import settings
                census_key = getattr(settings, "census_api_key", None)
            except Exception:
                census_key = None
            census = CensusClient(census_api_key=census_key)
        self.census = census

        # WorldPop
        if worldpop is None:
            try:
                from traffic_service.config import settings
                wp_key = getattr(settings, "worldpop_api_key", None)
            except Exception:
                wp_key = None
            worldpop = WorldPopClient(api_key=wp_key, dataset="wpgppop")
        self.worldpop = worldpop

    async def _fetch_worldpop(self, *, lat: float, lon: float, local_date: str, radius_km: float) -> _DemographicsAgg:
        try:
            year_raw = int(str(local_date)[:4])
        except Exception:
            year_raw = datetime.utcnow().year

        year, year_note = clamp_worldpop_year(year_raw)

        geojson_fc = make_circle_geojson_featurecollection(lat=lat, lon=lon, radius_km=radius_km, points=72)
        area = circle_area_km2(radius_km)

        wp = await self.worldpop.total_population_stats(
            year=year,
            geojson_obj=geojson_fc,
            runasync=False,
            timeout_s=35.0,
            poll_timeout_s=45.0,
            poll_interval_s=0.8,
            debug=False,
        )

        if wp.total_population is None or area <= 0:
            note = f"WorldPop no data ({wp.notes})"
            if year_note:
                note += f" | {year_note}"
            note += f" | source={wp.source_url}"
            return _DemographicsAgg(radius_km=radius_km, pop_density=None, housing_density=None, notes=note)

        pop_density = float(wp.total_population) / float(area)

        note = f"WorldPop wpgppop stats ({wp.notes})"
        if year_note:
            note += f" | {year_note}"
        note += f" | source={wp.source_url}"

        return _DemographicsAgg(radius_km=radius_km, pop_density=pop_density, housing_density=None, notes=note)

    async def _fetch_census_fallback(self, *, lat: float, lon: float, local_date: str, radius_km: float) -> _DemographicsAgg:
        """
        Fallback only: use when WorldPop has no data and location is in the US.
        """
        layer_id = 2
        try:
            year = int(str(local_date)[:4])
        except Exception:
            year = datetime.utcnow().year

        geoids = await self.census.tiger_block_groups(layer_id=layer_id, lat=lat, lon=lon, radius_km=radius_km)

        seen = set()
        bg_keys: list[tuple[str, str, str, str]] = []
        for g in geoids:
            g2 = (g or "").strip()
            if len(g2) < 12:
                continue
            g2 = g2[:12]
            if not g2.isdigit():
                continue
            state = g2[0:2]
            county = g2[2:5]
            tract = g2[5:11]
            bg = g2[11:12]
            key = (state, county, tract, bg)
            if key in seen:
                continue
            seen.add(key)
            bg_keys.append(key)

        pop_sum = 0
        housing_sum = 0
        for (state, county, tract, bg) in bg_keys:
            try:
                pop, housing = await self.census.acs5_bg(year=year, state=state, county=county, tract=tract, bg=bg)
            except Exception:
                pop, housing = (0, 0)
            pop_sum += int(pop)
            housing_sum += int(housing)

        area = circle_area_km2(radius_km)
        pop_density = pop_sum / area if area > 0 else None
        housing_density = housing_sum / area if area > 0 else None

        return _DemographicsAgg(
            radius_km=radius_km,
            pop_density=pop_density,
            housing_density=housing_density,
            notes="Census ACS5 + TIGERweb fallback",
        )

    async def compute_density(
        self,
        ctx: ExecutionContext,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        local_date: Optional[str] = None,
        radius_km: float = 5.0,
        soft_fail: bool = True,
    ) -> DemographicsSummary:
        lat2 = float(lat if lat is not None else ctx.request.lat)
        lon2 = float(lon if lon is not None else ctx.request.lon)

        if local_date is None:
            local_date = str(ctx.request.local_start).split(" ")[0]

        try:
            wp = await self._fetch_worldpop(lat=lat2, lon=lon2, local_date=str(local_date), radius_km=float(radius_km))
            if wp.pop_density is not None:
                return DemographicsSummary(
                    radius_km=float(radius_km),
                    population_density=wp.pop_density,
                    housing_density=wp.housing_density,
                    notes=wp.notes,
                )

            if _is_likely_us(lat2, lon2):
                cb = await self._fetch_census_fallback(
                    lat=lat2, lon=lon2, local_date=str(local_date), radius_km=float(radius_km)
                )
                return DemographicsSummary(
                    radius_km=float(radius_km),
                    population_density=cb.pop_density,
                    housing_density=cb.housing_density,
                    notes=f"{wp.notes} | {cb.notes}",
                )

            raise RuntimeError(f"no demographics data (WorldPop empty; non-US so no Census fallback). notes={wp.notes}")

        except Exception as e:
            if soft_fail:
                print(f"[WARN] DemographicsAgent failed -> empty. {type(e).__name__}: {e}")
                return DemographicsSummary(
                    radius_km=float(radius_km),
                    population_density=None,
                    housing_density=None,
                    notes="failed",
                )
            raise
