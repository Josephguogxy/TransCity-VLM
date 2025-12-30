# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image, ImageStat

from traffic_service.agents.base import ActionAgent
from traffic_service.clients.gibs import GIBSWMSClient
from traffic_service.core.context import ExecutionContext
from traffic_service.utils.time import parse_local_datetime


class SatelliteImageAgent(ActionAgent):
    name = "SatelliteImageAgent"

    def __init__(
        self,
        *,
        client: Optional[GIBSWMSClient] = None,
        default_layer: str = "HLS_L30_Nadir_BRDF_Adjusted_Reflectance",
        radius_km: float = 10.0,
        width: int = 512,
        height: int = 512,
        cache_dir: str = ".cache/sat_images",
        store: str = "file",  # base64/file/none
        max_lookback_days: int = 365,
        sleep_s: float = 0.1,
        debug: bool = False,
    ) -> None:
        self.client = client or GIBSWMSClient()
        self.default_layer = str(default_layer)
        self.radius_km = float(radius_km)
        self.width = int(width)
        self.height = int(height)
        self.cache_dir = Path(cache_dir)
        self.store = str(store)
        self.max_lookback_days = int(max_lookback_days)
        self.sleep_s = float(sleep_s)
        self.debug = bool(debug)

    @staticmethod
    def _bbox_around(lat: float, lon: float, radius_km: float) -> List[float]:
        lat_rad = math.radians(lat)
        dlat = radius_km / 110.574
        dlon = radius_km / (111.320 * max(1e-6, math.cos(lat_rad)))
        return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

    def _cache_path(self, *, lat: float, lon: float, time: str, layer: str, bbox: List[float]) -> Path:
        key = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "time": time,
            "layer": layer,
            "bbox": [round(x, 6) for x in bbox],
            "w": self.width,
            "h": self.height,
        }
        h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"gibs_{h}.png"

    @staticmethod
    def _is_bad_png(png: bytes) -> bool:
        if len(png) < 4000:
            try:
                _ = Image.open(BytesIO(png)).convert("RGBA")
            except Exception:
                return True

        try:
            img = Image.open(BytesIO(png)).convert("RGBA")

            alpha = img.split()[-1]
            if alpha.getbbox() is None:
                return True

            gray = img.convert("L")
            st = ImageStat.Stat(gray)
            mean = float(st.mean[0])
            std = float(st.stddev[0])

            if std < 1.0 and (mean < 5.0 or mean > 250.0):
                return True

            hist = gray.histogram()
            total = max(1, sum(hist))
            dark_ratio = sum(hist[:6]) / total
            bright_ratio = sum(hist[250:]) / total
            if dark_ratio > 0.985 or bright_ratio > 0.985:
                return True

            return False
        except Exception:
            return len(png) < 6000

    def _build_out(
        self,
        *,
        png: bytes,
        lat: float,
        lon: float,
        bbox: List[float],
        layer: str,
        chosen_date: str,
        requested_date: Optional[str],
        cache_path: str,
        request_url: Optional[str],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": "NASA GIBS WMS (epsg4326/best)",
            "layer": layer,
            "time": chosen_date,
            "requested_time": requested_date,
            "lat": lat,
            "lon": lon,
            "bbox": bbox,
            "width": self.width,
            "height": self.height,
            "mime_type": "image/png",
            "cache_path": cache_path,
            "request_url": request_url,
        }

        if self.store == "base64":
            out["encoding"] = "base64"
            out["data"] = base64.b64encode(png).decode("ascii")
        elif self.store == "file":
            out["encoding"] = "file"
            out["data"] = None
        else:
            out["encoding"] = "none"
            out["data"] = None

        return out

    async def fetch(
        self,
        ctx: ExecutionContext,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: Optional[float] = None,
        layer: Optional[str] = None,
        max_lookback_days: Optional[int] = None,
        soft_fail: bool = True,
        debug: Optional[bool] = None,
    ) -> dict[str, Any]:
        try:
            lat2 = float(lat if lat is not None else ctx.request.lat)
            lon2 = float(lon if lon is not None else ctx.request.lon)

            radius = float(radius_km if radius_km is not None else self.radius_km)
            lay = str(layer if layer is not None else self.default_layer)

            dbg = self.debug if debug is None else bool(debug)
            lookback = int(max_lookback_days if max_lookback_days is not None else self.max_lookback_days)

            requested_date: Optional[str] = None
            ls = getattr(ctx.request, "local_start", None)
            if ls:
                try:
                    requested_date = parse_local_datetime(str(ls)).date().isoformat()
                except Exception:
                    requested_date = None

            try:
                base_day = datetime.fromisoformat(requested_date).date() if requested_date else datetime.utcnow().date()
            except Exception:
                base_day = datetime.utcnow().date()

            bbox = self._bbox_around(lat2, lon2, radius)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            for k in range(0, lookback + 1):
                dt = (base_day - timedelta(days=k)).isoformat()
                cache_path = self._cache_path(lat=lat2, lon=lon2, time=dt, layer=lay, bbox=bbox)

                if cache_path.exists():
                    png = cache_path.read_bytes()
                    if self._is_bad_png(png):
                        try:
                            cache_path.unlink()
                        except Exception:
                            pass
                        continue

                    if dbg:
                        print(f"[SatelliteImageAgent] cache hit OK {cache_path}")
                    return self._build_out(
                        png=png, lat=lat2, lon=lon2, bbox=bbox,
                        layer=lay, chosen_date=dt, requested_date=requested_date,
                        cache_path=str(cache_path), request_url=None
                    )

                png, req_url = await self.client.get_map_png(
                    layer=lay, time=dt, bbox=bbox, width=self.width, height=self.height
                )

                if self._is_bad_png(png):
                    if dbg:
                        print(f"[SatelliteImageAgent] bad png skip dt={dt} bytes={len(png)} url={req_url}")
                    if self.sleep_s > 0:
                        await asyncio.sleep(self.sleep_s)
                    continue

                cache_path.write_bytes(png)
                if dbg:
                    print(f"[SatelliteImageAgent] downloaded OK dt={dt} path={cache_path} url={req_url}")

                return self._build_out(
                    png=png, lat=lat2, lon=lon2, bbox=bbox,
                    layer=lay, chosen_date=dt, requested_date=requested_date,
                    cache_path=str(cache_path), request_url=req_url
                )

            raise RuntimeError(f"No non-bad image found within lookback_days={lookback} requested_date={requested_date}")

        except Exception as e:
            if soft_fail:
                err = f"{type(e).__name__}: {e}"
                print(f"[WARN] SatelliteImageAgent failed -> empty. {err}")
                return {"ok": False, "error": err}
            raise
