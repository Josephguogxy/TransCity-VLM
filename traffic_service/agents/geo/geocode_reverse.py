# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Optional

from traffic_service.agents.base import ActionAgent
from traffic_service.clients.nominatim import NominatimClient
from traffic_service.core.context import ExecutionContext
from traffic_service.schemas import PlaceInfo


class ReverseGeocodeAgent(ActionAgent):
    name = "ReverseGeocodeAgent"

    def __init__(self, client: Optional[NominatimClient] = None) -> None:
        if client is None:
            try:
                from traffic_service.config import settings
                reverse_url = getattr(settings, "nominatim_url", "https://nominatim.openstreetmap.org/reverse")
                timeout_s = float(getattr(settings, "http_timeout_s", 30.0))
            except Exception:
                reverse_url = "https://nominatim.openstreetmap.org/reverse"
                timeout_s = 30.0
            client = NominatimClient(reverse_url=reverse_url, timeout_s=timeout_s)
        self.client = client

    async def get_place(
        self,
        ctx: ExecutionContext,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> PlaceInfo:
        lat2 = float(lat if lat is not None else ctx.request.lat)
        lon2 = float(lon if lon is not None else ctx.request.lon)

        data = await self.client.reverse(lat2, lon2, zoom=14)
        addr = data.get("address", {}) or {}
        return PlaceInfo(
            display_name=data.get("display_name", ""),
            city=addr.get("city") or addr.get("town") or addr.get("village"),
            county=addr.get("county"),
            state=addr.get("state"),
            country=addr.get("country"),
        )
