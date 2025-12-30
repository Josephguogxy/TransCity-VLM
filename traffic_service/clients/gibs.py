# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import List, Optional, Tuple

from traffic_service.clients.http import request_bytes
from traffic_service.config import settings


class GIBSWMSClient:
    WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        user_agent: str = "traffic-multiagent-template/0.1 (research)",
        debug: bool = False,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.headers = {"User-Agent": user_agent}
        self.debug = bool(debug)

    async def get_map_png(
        self,
        *,
        layer: str,
        time: Optional[str],
        bbox: List[float],
        width: int,
        height: int,
    ) -> Tuple[bytes, str]:
        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.1.1",
            "layers": layer,
            "styles": "default",
            "format": "image/png",
            "transparent": "true",
            "width": str(int(width)),
            "height": str(int(height)),
            "srs": "EPSG:4326",
            "bbox": ",".join(str(x) for x in bbox),
        }
        if time:
            params["time"] = time

        body, hdrs, final_url = await request_bytes(
            self.WMS_URL,
            params=params,
            headers=self.headers,
            timeout_s=max(settings.http_timeout_s, self.timeout_s),
            max_retries=self.max_retries,
            debug=self.debug,
        )

        ct = (hdrs.get("content-type") or "").lower()
        if "image" not in ct:
            raise RuntimeError(f"GIBS returned non-image content-type={ct}, url={final_url}")

        return body, final_url
