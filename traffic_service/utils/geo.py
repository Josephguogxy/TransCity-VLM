# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import math
from typing import Tuple

EARTH_RADIUS_M = 6371008.8  # mean Earth radius (meters)


def circle_area_km2(radius_km: float) -> float:
    if radius_km <= 0:
        return 0.0
    return math.pi * (radius_km ** 2)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bbox_around(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """Return (north, west, south, east) bbox in degrees around point."""
    lat_deg = radius_km / 111.32
    lon_deg = radius_km / (111.32 * math.cos(math.radians(lat)) + 1e-12)
    north = lat + lat_deg
    south = lat - lat_deg
    west = lon - lon_deg
    east = lon + lon_deg
    return (north, west, south, east)
