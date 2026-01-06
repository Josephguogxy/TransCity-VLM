# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from timezonefinder import TimezoneFinder  # type: ignore
except Exception:  # pragma: no cover
    TimezoneFinder = None  # type: ignore


@dataclass(frozen=True)
class TimeWindow:
    tz_name: str
    local_start: datetime
    local_end: datetime
    utc_start: datetime
    utc_end: datetime


def parse_local_datetime(s: str) -> datetime:
    """
    Accept "YYYY-MM-DD HH:MM" or ISO-like strings.
    Return naive datetime interpreted as local time.
    """
    ss = s.strip().replace("T", " ")
    # datetime.fromisoformat supports "YYYY-MM-DD HH:MM"
    return datetime.fromisoformat(ss)


def timezone_name_from_latlon(lat: float, lon: float) -> str:
    if TimezoneFinder is None:
        return "UTC"
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=lat, lng=lon)
    return tz or "UTC"


def local_to_utc(local_dt: datetime, tz_name: str) -> datetime:
    """
    Convert naive local datetime -> aware UTC datetime.
    """
    if tz_name == "UTC" or ZoneInfo is None:
        return local_dt.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    aware_local = local_dt.replace(tzinfo=tz)
    return aware_local.astimezone(timezone.utc)


def hourly_points_covering(start_local: datetime, end_local: datetime) -> List[datetime]:
    """
    Produce hourly points from floor(start) to floor(end) inclusive.
    This covers the interval at hour granularity.
    """
    start = start_local.replace(minute=0, second=0, microsecond=0)
    end = end_local.replace(minute=0, second=0, microsecond=0)
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def parse_local_dt(s: str) -> datetime:
    """
    For postprocess scoring: parse ISO or "YYYY-MM-DD HH:MM" into datetime (naive).
    """
    return parse_local_datetime(s)
