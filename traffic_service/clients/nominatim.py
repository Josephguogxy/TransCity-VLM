# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traffic_service.clients.http import request_json


def _derive_search_url(nominatim_url: str) -> str:
    u = (nominatim_url or "").rstrip("/")
    if u.endswith("/reverse"):
        return u[:-len("/reverse")] + "/search"
    if u.endswith("/search"):
        return u
    return u + "/search"


class NominatimClient:
    def __init__(
        self,
        *,
        reverse_url: str,
        search_url: Optional[str] = None,
        timeout_s: float = 20.0,
        max_retries: int = 2,
        user_agent: str = "traffic-multiagent-template/0.1 (research)",
        trust_env: bool = True,
        verify_tls: bool = True,
        debug: bool = False,
    ) -> None:
        self.reverse_url = reverse_url
        self.search_url = search_url or _derive_search_url(reverse_url)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.headers = {"User-Agent": user_agent}
        self.trust_env = bool(trust_env)
        self.verify_tls = bool(verify_tls)
        self.debug = bool(debug)

    async def search(self, query: str, *, limit: int = 5) -> List[dict]:
        params = {"format": "jsonv2", "q": query, "limit": int(limit), "addressdetails": 1}
        data, _ = await request_json(
            self.search_url,
            params=params,
            headers=self.headers,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            trust_env=self.trust_env,
            verify_tls=self.verify_tls,
            debug=self.debug,
        )
        return data if isinstance(data, list) else []

    async def reverse(self, lat: float, lon: float, *, zoom: int = 14) -> Dict[str, Any]:
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": int(zoom), "addressdetails": 1}
        data, _ = await request_json(
            self.reverse_url,
            params=params,
            headers=self.headers,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            trust_env=self.trust_env,
            verify_tls=self.verify_tls,
            debug=self.debug,
        )
        return data if isinstance(data, dict) else {}
