# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import httpx


RetryableURLs = Union[str, list[str]]


def _normalize_urls(urls: RetryableURLs) -> list[str]:
    if isinstance(urls, str):
        return [urls]
    out = []
    seen = set()
    for u in urls:
        u2 = (u or "").strip()
        if not u2:
            continue
        k = u2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(u2)
    return out


def _should_retry_status(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


async def request_json(
    urls: RetryableURLs,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_body: Any = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    follow_redirects: bool = True,
    max_retries: int = 2,
    backoff_base: float = 0.8,
    trust_env: bool = True,
    verify_tls: bool = True,
    debug: bool = False,
) -> Tuple[Any, str]:
    url_list = _normalize_urls(urls)
    if not url_list:
        raise ValueError("request_json: urls is empty")

    last_err: Optional[Exception] = None
    backoff = float(backoff_base)

    async with httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=follow_redirects,
        trust_env=trust_env,
        verify=verify_tls,
    ) as client:
        for attempt in range(max_retries + 1):
            cand = list(url_list)
            random.shuffle(cand)

            for url in cand:
                try:
                    r = await client.request(
                        method.upper(),
                        url,
                        params=params,
                        json=json_body,
                        data=data,
                        headers=headers,
                    )

                    # Treat 204/empty as no-data and return without retry.
                    if r.status_code == 204 or not r.content:
                        return None, str(r.url)

                    # Standard error handling.
                    if r.status_code >= 400:
                        if _should_retry_status(r.status_code) and attempt < max_retries:
                            if debug:
                                print(f"[http] retryable HTTP {r.status_code} url={url}")
                            last_err = httpx.HTTPStatusError("retryable", request=r.request, response=r)
                            continue
                        r.raise_for_status()

                    # JSON decode failure: retry to handle transient issues.
                    try:
                        return r.json(), str(r.url)
                    except ValueError as e:
                        last_err = e
                        if attempt < max_retries:
                            if debug:
                                print(f"[http] JSON decode error, retrying url={url}")
                            continue
                        raise

                except httpx.HTTPStatusError as e:
                    last_err = e
                except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                    last_err = e
                    if debug:
                        print(f"[http] net error {type(e).__name__} url={url}")
                except Exception as e:
                    last_err = e
                    if debug:
                        print(f"[http] unknown error {type(e).__name__} url={url}")

            if attempt < max_retries:
                await asyncio.sleep(backoff + random.random() * 0.3)
                backoff *= 2.0

    raise RuntimeError(f"request_json failed. last_err={type(last_err).__name__}: {last_err}")


async def request_bytes(
    urls: RetryableURLs,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    follow_redirects: bool = True,
    max_retries: int = 2,
    backoff_base: float = 0.8,
    trust_env: bool = True,
    verify_tls: bool = True,
    debug: bool = False,
) -> Tuple[bytes, Dict[str, str], str]:
    url_list = _normalize_urls(urls)
    if not url_list:
        raise ValueError("request_bytes: urls is empty")

    last_err: Optional[Exception] = None
    backoff = float(backoff_base)

    for attempt in range(max_retries + 1):
        cand = list(url_list)
        random.shuffle(cand)

        for url in cand:
            try:
                async with httpx.AsyncClient(
                    timeout=timeout_s,
                    follow_redirects=follow_redirects,
                    trust_env=trust_env,
                    verify=verify_tls,
                ) as client:
                    r = await client.request(method.upper(), url, params=params, headers=headers)

                if r.status_code >= 400:
                    if _should_retry_status(r.status_code) and attempt < max_retries:
                        if debug:
                            print(f"[http] retryable HTTP {r.status_code} url={url}")
                        last_err = httpx.HTTPStatusError("retryable", request=r.request, response=r)
                        continue
                    r.raise_for_status()

                hdrs = {k.lower(): v for k, v in r.headers.items()}
                return r.content, hdrs, str(r.url)

            except Exception as e:
                last_err = e
                if debug:
                    print(f"[http] err {type(e).__name__} url={url}")

        if attempt < max_retries:
            await asyncio.sleep(backoff + random.random() * 0.3)
            backoff *= 2.0

    raise RuntimeError(f"request_bytes failed. last_err={type(last_err).__name__}: {last_err}")
