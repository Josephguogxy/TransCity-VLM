# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
from typing import Any, Optional

from agentic_system.agents.base import ActionAgent
from agentic_system.core.context import ExecutionContext
from agentic_system.llm.clients import OpenAICompatibleClient, LLMChatMessage, client_for
from agentic_system.llm.prompts import search_query_prompt


class QueryGeneratorAgent(ActionAgent):
    name = "QueryGeneratorAgent"

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None) -> None:
        self.llm = llm or client_for("extractor")

    async def generate(
        self,
        ctx: ExecutionContext,
        *,
        kind: str,
        place_key: str = "place",
        roads_key: str = "roads",
        debug: bool = True,
    ) -> list[str]:
        if kind not in {"news", "accidents"}:
            raise ValueError("QueryGeneratorAgent requires kind in {'news','accidents'}")

        place: Any = ctx.get(place_key, None)
        place_hint = getattr(place, "display_name", None) or str(place)

        roads: Any = ctx.get(roads_key, [])
        road_hint = None
        if isinstance(roads, list) and roads:
            r0 = roads[0] or {}
            if isinstance(r0, dict):
                road_hint = (r0.get("ref") or r0.get("name"))

        sys = "You are a search query generation model."

        user = search_query_prompt(
            kind=kind,
            place_hint=place_hint,
            road_hint=road_hint,
            local_start=str(ctx.request.local_start),
            local_end=str(ctx.request.local_end),
            question=str(ctx.request.question),
        )

        data = await asyncio.to_thread(
            self.llm.chat_json,
            [LLMChatMessage("system", sys), LLMChatMessage("user", user)],
            model=None,
            temperature=0.0,
            max_tokens=900,
        )
        queries = data.get("queries", []) or []
        out = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:12]

        if debug:
            print(f"\n[QueryGenerator][{kind}] window={ctx.request.local_start}~{ctx.request.local_end}")
            print(f"[QueryGenerator][{kind}] place_hint={place_hint!r} road_hint={road_hint!r}")
            for i, q in enumerate(out, 1):
                print(f"[QueryGenerator][{kind}] {i:02d}. {q}")

        return out
