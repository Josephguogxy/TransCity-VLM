# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Optional

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

_pool: Optional[MySQLConnectionPool] = None


def get_pool() -> MySQLConnectionPool:
    global _pool
    if _pool is not None:
        return _pool

    from traffic_service.config import get_settings
    cfg = get_settings().mysql

    _pool = MySQLConnectionPool(
        pool_name="traffic_pool",
        pool_size=int(getattr(cfg, "pool_size", 8)),
        pool_reset_session=True,
        host=cfg.host,
        port=int(cfg.port),
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        connection_timeout=int(getattr(cfg, "connect_timeout_s", 5)),
        autocommit=True,
    )
    return _pool


def get_conn() -> mysql.connector.MySQLConnection:
    
    return get_pool().get_connection()
