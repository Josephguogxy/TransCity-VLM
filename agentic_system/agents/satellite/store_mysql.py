# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict

from agentic_system.agents.base import ActionAgent
from agentic_system.core.context import ExecutionContext
from agentic_system.db.satellite_images_repo import upsert_satellite_png


class SatelliteImageStoreMySQLAgent(ActionAgent):
    name = "SatelliteImageStoreMySQLAgent"

    async def store(
        self,
        ctx: ExecutionContext,
        *,
        input_key: str = "satellite_image",
        soft_fail: bool = True,
    ) -> Dict[str, object]:
        sat = ctx.get(input_key, None)
        if not isinstance(sat, dict):
            return {"stored": False, "reason": "satellite_image_missing_or_not_dict", "image_id": None, "image_ref": None}

        cache_path = sat.get("cache_path")
        if not cache_path or not isinstance(cache_path, str) or not Path(cache_path).exists():
            return {"stored": False, "reason": "cache_path_missing", "image_id": None, "image_ref": None}

        try:
            img_bytes: bytes = await asyncio.to_thread(Path(cache_path).read_bytes)

            lat = float(sat.get("lat", ctx.request.lat or 0.0))
            lon = float(sat.get("lon", ctx.request.lon or 0.0))
            layer = str(sat.get("layer") or "")
            image_date = sat.get("time")
            bbox = sat.get("bbox") or []
            width = int(sat.get("width") or 512)
            height = int(sat.get("height") or 512)
            mime_type = str(sat.get("mime_type") or "image/png")
            source = str(sat.get("source") or "NASA GIBS WMS")
            request_url = sat.get("request_url")

            res = await asyncio.to_thread(
                upsert_satellite_png,
                lat=lat,
                lon=lon,
                image_date=image_date,
                layer=layer,
                bbox=bbox,
                width=width,
                height=height,
                mime_type=mime_type,
                png_bytes=img_bytes,
                source=source,
                request_url=request_url,
            )

            image_id = int(res["image_id"])
            sha256 = str(res["sha256"])

            try:
                from agentic_system.config import settings
                dbname = getattr(getattr(settings, "mysql", None), "database", None) or "mysql"
            except Exception:
                dbname = "mysql"

            image_ref = f"mysql:{dbname}:satellite_images:{image_id}"

            sat["image_id"] = image_id
            sat["image_ref"] = image_ref
            sat["sha256"] = sha256

            return {
                "stored": True,
                "image_id": image_id,
                "image_ref": image_ref,
                "sha256": sha256,
                "mime_type": mime_type,
                "width": width,
                "height": height,
            }

        except Exception as e:
            if soft_fail:
                return {"stored": False, "image_id": None, "image_ref": None, "error": f"{type(e).__name__}: {e}"}
            raise
