# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Optional

from traffic_service.clients.http import request_json


class OverpassClient:
    def __init__(
        self,
        endpoints: list[str],
        *,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        user_agent: str = "traffic-multiagent-template/0.1 (research)",
        debug: bool = False,
    ) -> None:
        self.endpoints = endpoints
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.headers = {"User-Agent": user_agent}
        self.debug = bool(debug)

    async def post(self, query: str) -> dict:
        data, _ = await request_json(
            self.endpoints,
            method="POST",
            data=query,
            headers=self.headers,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            debug=self.debug,
        )
        return data if isinstance(data, dict) else {}
