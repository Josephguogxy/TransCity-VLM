# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from agentic_system.clients.http import request_json


class CensusClient:
    TIGER_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"
    CENSUS_BASE = "https://api.census.gov/data"

    def __init__(self, *, census_api_key: Optional[str] = None) -> None:
        self.census_api_key = census_api_key

        # cache single BG (year,state,county,tract,bg) -> (pop,housing)
        self._acs_cache_bg: Dict[Tuple[int, str, str, str, str], Tuple[int, int]] = {}

        # cache whole tract (year,state,county,tract) -> {bg -> (pop,housing)}
        self._acs_cache_tract: Dict[Tuple[int, str, str, str], Dict[str, Tuple[int, int]]] = {}

    async def tiger_block_groups(
        self,
        *,
        layer_id: int,
        lat: float,
        lon: float,
        radius_km: float,
        geoid_len: int = 12,
    ) -> List[str]:
        """
        Query TIGERweb to get GEOID(12) for block groups intersecting a circle around (lat,lon).
        """
        url = f"{self.TIGER_BASE}/{int(layer_id)}/query"
        params = {
            "f": "json",
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": str(int(float(radius_km) * 1000)),
            "units": "esriSRUnit_Meter",
            "outFields": "GEOID",
            "returnGeometry": "false",
        }

        data, _ = await request_json(url, params=params, timeout_s=30, max_retries=2)

        geoids: list[str] = []
        if isinstance(data, dict):
            for f in data.get("features", []) or []:
                attrs = f.get("attributes") or {}
                g = attrs.get("GEOID")
                if not isinstance(g, str):
                    continue

                g2 = g.strip()
                if len(g2) >= geoid_len:
                    g2 = g2[:geoid_len]
                else:
                    continue

                if len(g2) == geoid_len and g2.isdigit():
                    geoids.append(g2)

        return sorted(set(geoids))

    async def acs5_bg(self, *, year: int, state: str, county: str, tract: str, bg: str) -> Tuple[int, int]:
        """
        Fallback: request a single block group.
        """
        key = (int(year), str(state), str(county), str(tract), str(bg))
        if key in self._acs_cache_bg:
            return self._acs_cache_bg[key]

        url = f"{self.CENSUS_BASE}/{int(year)}/acs/acs5"
        params = {
            "get": "B01003_001E,B25001_001E",
            "for": f"block group:{bg}",
            "in": f"state:{state} county:{county} tract:{tract}",
        }
        if self.census_api_key:
            params["key"] = self.census_api_key

        rows, _ = await request_json(url, params=params, timeout_s=30, max_retries=2)
        pop, housing = self._parse_acs_row(rows, bg_hint=str(bg))
        self._acs_cache_bg[key] = (pop, housing)
        return (pop, housing)

    async def acs5_block_groups_in_tract(
        self,
        *,
        year: int,
        state: str,
        county: str,
        tract: str,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Optimization: fetch all block groups in a tract in one request:
        for=block group:* & in=state:.. county:.. tract:..
        Returns: { "1": (pop,housing), "2": (...), ... }
        """
        tkey = (int(year), str(state), str(county), str(tract))
        if tkey in self._acs_cache_tract:
            return self._acs_cache_tract[tkey]

        url = f"{self.CENSUS_BASE}/{int(year)}/acs/acs5"
        params = {
            "get": "B01003_001E,B25001_001E",
            "for": "block group:*",
            "in": f"state:{state} county:{county} tract:{tract}",
        }
        if self.census_api_key:
            params["key"] = self.census_api_key

        rows, meta = await request_json(url, params=params, timeout_s=30, max_retries=2)

        # 204 / empty
        if not isinstance(rows, list) or len(rows) < 2:
            self._acs_cache_tract[tkey] = {}
            return {}

        header = rows[0]
        if not isinstance(header, list):
            self._acs_cache_tract[tkey] = {}
            return {}

        # resolve column indices (robust)
        def find_col(*cands: str) -> Optional[int]:
            for c in cands:
                if c in header:
                    return header.index(c)
            # case-insensitive fallback
            low = [str(x).lower() for x in header]
            for c in cands:
                c2 = c.lower()
                if c2 in low:
                    return low.index(c2)
            return None

        i_pop = find_col("B01003_001E")
        i_hou = find_col("B25001_001E")
        i_bg = find_col("block group", "block group:")  # normally "block group"

        if i_pop is None or i_hou is None or i_bg is None:
            self._acs_cache_tract[tkey] = {}
            return {}

        out: Dict[str, Tuple[int, int]] = {}
        for row in rows[1:]:
            if not isinstance(row, list) or len(row) <= max(i_pop, i_hou, i_bg):
                continue
            bg = str(row[i_bg]).strip()
            pop = _safe_int(row[i_pop])
            housing = _safe_int(row[i_hou])
            if bg:
                out[bg] = (pop, housing)
                # also warm single-BG cache
                self._acs_cache_bg[(int(year), str(state), str(county), str(tract), bg)] = (pop, housing)

        self._acs_cache_tract[tkey] = out
        return out

    @staticmethod
    def _parse_acs_row(rows: object, bg_hint: str) -> Tuple[int, int]:
        """
        Parse response of single BG query: rows = [header, values]
        """
        if not isinstance(rows, list) or len(rows) < 2:
            return (0, 0)
        header = rows[0]
        values = rows[1]
        if not isinstance(header, list) or not isinstance(values, list):
            return (0, 0)
        m = dict(zip(header, values))
        pop = _safe_int(m.get("B01003_001E", 0))
        housing = _safe_int(m.get("B25001_001E", 0))
        return (pop, housing)


def _safe_int(v: object) -> int:
    try:
        return int(float(v or 0))
    except Exception:
        return 0
