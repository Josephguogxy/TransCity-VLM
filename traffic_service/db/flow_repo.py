# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traffic_service.db.mysql_pool import get_conn


def fetch_all_sensors() -> List[Dict[str, Any]]:
    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)
        cur.execute(
            """
            SELECT sensor_idx, sensor_code, lat, lon, fwy, lanes, direction
            FROM sensors
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            """
        )
        rows = cur.fetchall() or []
        cur.close()
        return rows
    finally:
        cnx.close()


def fetch_flow_year_one(sensor_idx: int, year: int) -> Optional[Dict[str, Any]]:
    rows = fetch_flow_year_many([sensor_idx], year)
    return rows.get(int(sensor_idx))


def fetch_flow_year_many(sensor_idxs: List[int], year: int) -> Dict[int, Dict[str, Any]]:
    """
    Fetch all rows for (year, sensor_idx in [...]).
    Returns: {sensor_idx: row}
    """
    if not sensor_idxs:
        return {}

    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(sensor_idxs))
        sql = f"""
            SELECT sensor_idx, year, freq_min, start_local, n_steps, values_zlib
            FROM sensor_flow_year
            WHERE year=%s AND sensor_idx IN ({placeholders})
        """
        cur.execute(sql, [int(year)] + [int(x) for x in sensor_idxs])
        rows = cur.fetchall() or []
        cur.close()

        out: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            out[int(r["sensor_idx"])] = r
        return out
    finally:
        cnx.close()
