# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import abc
from typing import Optional
from urllib.parse import urlparse

from traffic_service.clients.http import request_json
from traffic_service.config import settings
from traffic_service.schemas import SearchResult


class WebSearchClient(abc.ABC):
    @abc.abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        raise NotImplementedError


class SerperWebSearchClient(WebSearchClient):
    def __init__(self, api_key: str, endpoint: str) -> None:
        self.api_key = api_key
        self.endpoint = endpoint

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if not self.api_key or "REPLACE_WITH" in self.api_key:
            raise RuntimeError("Serper API key not configured")

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": int(num_results)}

        data, _ = await request_json(
            self.endpoint,
            method="POST",
            json_body=payload,
            headers=headers,
            timeout_s=settings.web_search.timeout_s,
            max_retries=2,
        )

        out: list[SearchResult] = []
        if isinstance(data, dict):
            for it in (data.get("organic") or []):
                title = it.get("title") or ""
                link = it.get("link") or ""
                snippet = it.get("snippet") or ""
                out.append(
                    SearchResult(
                        title=title,
                        snippet=snippet,
                        url=link,
                        source=urlparse(link).netloc if link else None,
                        published=it.get("date") or None,
                    )
                )
        return out


class BingWebSearchClient(WebSearchClient):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if not self.api_key or "REPLACE_WITH" in self.api_key:
            raise RuntimeError("Bing API key not configured")

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": int(num_results), "textDecorations": False, "textFormat": "Raw"}

        data, _ = await request_json(
            url,
            params=params,
            headers=headers,
            timeout_s=settings.web_search.timeout_s,
            max_retries=2,
        )

        out: list[SearchResult] = []
        if isinstance(data, dict):
            for w in (data.get("webPages", {}) or {}).get("value", []) or []:
                out.append(
                    SearchResult(
                        title=w.get("name", ""),
                        snippet=w.get("snippet", ""),
                        url=w.get("url", ""),
                        source=(w.get("provider", [{}])[0] or {}).get("name"),
                        published=w.get("dateLastCrawled"),
                    )
                )
        return out


def client_from_settings() -> WebSearchClient:
    if settings.web_search.provider == "serper":
        return SerperWebSearchClient(api_key=settings.web_search.api_key, endpoint=settings.web_search.endpoint)
    if settings.web_search.provider == "bing":
        return BingWebSearchClient(api_key=settings.web_search.api_key)
    raise ValueError(f"Unknown web_search.provider: {settings.web_search.provider}")
