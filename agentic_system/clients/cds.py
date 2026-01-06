# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from agentic_system.config import settings

try:
    import cdsapi  # type: ignore
except Exception:  # pragma: no cover
    cdsapi = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


@dataclass
class CdsEra5Client:
    """
    CDS/ERA5 downloader + cache + point reader.
    We request a small bbox and select nearest grid point.
    """
    dataset: str = settings.cds.dataset
    cache_dir: str = settings.cds.cache_dir

    def _cache_path(self, key_obj: dict) -> Path:
        h = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return Path(self.cache_dir) / f"era5_{h}.nc"

    @staticmethod
    def _utc_hours_between(start_utc: datetime, end_utc: datetime) -> list[datetime]:
        s = start_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        e = end_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        if e < s:
            s, e = e, s
        out = []
        cur = s
        while cur <= e:
            out.append(cur)
            cur = cur + (datetime.min.replace(tzinfo=timezone.utc) - datetime.min.replace(tzinfo=timezone.utc))  # no-op guard
            cur = cur.replace(tzinfo=timezone.utc)  # keep aware
            cur = cur + __import__("datetime").timedelta(hours=1)  # avoid local import loops
        return out

    def download_era5_single_levels(
        self,
        *,
        lat: float,
        lon: float,
        start_utc: datetime,  # aware utc
        end_utc: datetime,    # aware utc
        target_nc: Path,
    ) -> None:
        """
        Uses cdsapi to retrieve netcdf. The request keys follow CDS API patterns (data_format, area, year/month/day/time).
        :contentReference[oaicite:2]{index=2}
        """
        if cdsapi is None:
            raise RuntimeError("cdsapi not installed. pip install cdsapi")

        # tiny bbox around point (ERA5 grid is 0.25°; bbox helps avoid corner cases)
        north, south = lat + 0.25, lat - 0.25
        west, east = lon - 0.25, lon + 0.25

        hours = []
        cur = start_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end0 = end_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        while cur <= end0:
            hours.append(cur)
            cur = cur + __import__("datetime").timedelta(hours=1)

        years = sorted({f"{h.year:04d}" for h in hours})
        months = sorted({f"{h.month:02d}" for h in hours})
        days = sorted({f"{h.day:02d}" for h in hours})
        times = sorted({f"{h.hour:02d}:00" for h in hours})

        # CDS key sometimes "UID:APIKEY" — cdsapi supports it, your config keeps it.
        client = cdsapi.Client(url=settings.cds.url, key=settings.cds.key, timeout=settings.cds.request_timeout_s)

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "area": [north, west, south, east],  # N,W,S,E  :contentReference[oaicite:3]{index=3}
            "data_format": "netcdf",             # netcdf/grib  :contentReference[oaicite:4]{index=4}
            "download_format": "unarchived",
        }

        target_nc.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_nc.with_suffix(".download")

        # cdsapi versions differ: support both retrieve(..., target) and retrieve(...).download(target)
        try:
            client.retrieve(self.dataset, request, str(tmp))
        except TypeError:
            res = client.retrieve(self.dataset, request)
            if hasattr(res, "download"):
                res.download(str(tmp))
            else:
                raise

        # Sometimes it's a zip
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z:
                nc_names = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise RuntimeError(f"Downloaded zip has no .nc: {z.namelist()[:20]}")
                name = nc_names[0]
                z.extract(name, path=target_nc.parent)
                extracted = target_nc.parent / name
                if target_nc.exists():
                    target_nc.unlink()
                extracted.replace(target_nc)
            tmp.unlink(missing_ok=True)
        else:
            if target_nc.exists():
                target_nc.unlink()
            tmp.replace(target_nc)

    def read_point_hourly(
        self,
        *,
        nc_path: Path,
        lat: float,
        lon: float,
        start_utc: datetime,
        end_utc: datetime,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Returns (temperature_c, wind_speed_ms, wind_direction_deg) aligned to utc hours in [start_utc, end_utc].
        """
        if xr is None or np is None:
            raise RuntimeError("xarray + numpy required to parse netcdf. pip install xarray netCDF4 numpy")

        ds = xr.open_dataset(nc_path)

        # coordinate names vary
        lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
        lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
        if lat_name is None or lon_name is None:
            raise KeyError(f"No lat/lon coords in netcdf. coords={list(ds.coords)}")

        # time coord
        time_name = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
        if time_name is None:
            raise KeyError(f"No time coord. coords={list(ds.coords)}")

        pt = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")

        def pick(*names: str):
            for n in names:
                if n in pt:
                    return pt[n].values
            raise KeyError(f"None of {names} in dataset. vars={list(pt.data_vars)}")

        t2m = pick("t2m", "2m_temperature")
        u10 = pick("u10", "10m_u_component_of_wind")
        v10 = pick("v10", "10m_v_component_of_wind")

        times = pt[time_name].values

        # convert to python utc datetimes
        def to_py_utc(x) -> datetime:
            # xarray gives numpy datetime64
            ts = (x - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)

        times_py = [to_py_utc(t) for t in times]

        s = start_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        e = end_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        idx = [i for i, tt in enumerate(times_py) if s <= tt <= e]
        if not idx:
            # fallback: return all
            idx = list(range(len(times_py)))

        t2m = t2m[idx]
        u10 = u10[idx]
        v10 = v10[idx]

        temperature_c = (t2m - 273.15).astype(float).tolist()
        wind_speed = (np.sqrt(u10**2 + v10**2)).astype(float).tolist()
        wind_dir = ((np.degrees(np.arctan2(u10, v10)) + 180.0) % 360.0).astype(float).tolist()

        return temperature_c, wind_speed, wind_dir

    def fetch_hourly_cached(
        self,
        *,
        lat: float,
        lon: float,
        start_utc: datetime,
        end_utc: datetime,
    ) -> tuple[list[float], list[float], list[float]]:
        key_obj = {
            "dataset": self.dataset,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "vars": ["t2m", "u10", "v10"],
        }
        path = self._cache_path(key_obj)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            self.download_era5_single_levels(lat=lat, lon=lon, start_utc=start_utc, end_utc=end_utc, target_nc=path)

        return self.read_point_hourly(nc_path=path, lat=lat, lon=lon, start_utc=start_utc, end_utc=end_utc)
