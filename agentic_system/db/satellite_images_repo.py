# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import hashlib
from typing import Optional, Any, Dict, List

from agentic_system.db.mysql_pool import get_conn


def upsert_satellite_png(
    *,
    lat: float,
    lon: float,
    image_date: Optional[str],
    layer: str,
    bbox: List[float],  # [min_lon, min_lat, max_lon, max_lat]
    width: int,
    height: int,
    mime_type: str,
    png_bytes: bytes,
    source: Optional[str] = None,
    request_url: Optional[str] = None,
) -> Dict[str, Any]:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError("bbox must be [min_lon,min_lat,max_lon,max_lat]")

    sha = hashlib.sha256(png_bytes).hexdigest()

    cnx = get_conn()
    try:
        cur = cnx.cursor()
        sql = """
        INSERT INTO satellite_images
          (lat, lon, image_date, layer,
           bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
           width, height, mime_type, sha256, png, source, request_url)
        VALUES
          (%s,%s,%s,%s,
           %s,%s,%s,%s,
           %s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          image_id = LAST_INSERT_ID(image_id)
        """
        cur.execute(
            sql,
            (
                float(lat), float(lon), image_date, layer,
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                int(width), int(height),
                mime_type, sha, png_bytes,
                source, request_url,
            ),
        )
        image_id = int(cur.lastrowid)
        cur.close()
        return {"image_id": image_id, "sha256": sha}
    finally:
        cnx.close()


def fetch_satellite_png(image_id: int) -> Dict[str, Any]:
    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)
        cur.execute(
            """
            SELECT image_id, mime_type, png, lat, lon, image_date, layer, width, height
            FROM satellite_images
            WHERE image_id=%s
            """,
            (int(image_id),),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise KeyError("satellite_images not found: image_id=%s" % int(image_id))
        return row
    finally:
        cnx.close()
