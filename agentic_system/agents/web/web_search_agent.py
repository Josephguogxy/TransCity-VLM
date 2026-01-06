# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from agentic_system.agents.base import ActionAgent
from agentic_system.clients.web_search import WebSearchClient, client_from_settings
from agentic_system.core.context import ExecutionContext
from agentic_system.schemas import SearchResult


class WebSearchAgent(ActionAgent):
    name = "WebSearchAgent"

    def __init__(self, client: Optional[WebSearchClient] = None, *, debug: bool = False) -> None:
        self.client = client or client_from_settings()
        self.debug = bool(debug)

    @staticmethod
    def _brief_item(it: SearchResult) -> str:
        t = (it.title or "").replace("\n", " ").strip()
        if len(t) > 120:
            t = t[:117] + "..."
        pub = it.published or ""
        src = it.source or ""
        return f"- {t} | pub={pub} | src={src} | url={it.url}"

    async def search(
        self,
        ctx: ExecutionContext,
        *,
        queries: list[str],
        per_query: int = 5,
        debug: Optional[bool] = None,
        soft_fail: bool = True,
    ) -> list[SearchResult]:
        dbg = self.debug if debug is None else bool(debug)
        all_items: list[SearchResult] = []

        try:
            for i, q in enumerate(list(queries)):
                if dbg:
                    print(f"\n[WebSearch] ({i+1}/{len(queries)}) q={q!r}")
                try:
                    items = await self.client.search(q, num_results=int(per_query))
                    if dbg:
                        print(f"[WebSearch]   -> got {len(items)} results")
                        for it in items[:3]:
                            print("[WebSearch]   ", self._brief_item(it))
                    all_items.extend(items)
                except Exception as e:
                    if dbg:
                        print(f"[WebSearch][ERR] {type(e).__name__}: {e} (q={q!r})")
                    continue

            if dbg:
                print(f"\n[WebSearch] DONE: total_results={len(all_items)}")
            return all_items

        except Exception as e:
            if soft_fail:
                print(f"[WARN] WebSearchAgent failed -> []. {type(e).__name__}: {e}")
                return []
            raise
