# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup

from agentic_system.clients.http import request_bytes


class ContentFetcher:
    async def fetch_text(self, url: str) -> Optional[str]:
        try:
            body, hdrs, _final = await request_bytes(url, timeout_s=20, max_retries=1)
        except Exception:
            return None

        try:
            html = body.decode("utf-8", errors="ignore")
        except Exception:
            return None

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:20000]
