# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import json
import textwrap
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from traffic_service.agents.base import ActionAgent
from traffic_service.agents.web.content_fetcher import ContentFetcher
from traffic_service.agents.web.scoring import (
    WebDocument,
    ExtractedEvent,
    VerifiedEvent,
    rule_score_doc,
    cluster_events,
    rank_clusters,
    _domain_from_url,
)
from traffic_service.core.context import ExecutionContext
from traffic_service.llm.clients import OpenAICompatibleClient, LLMChatMessage, client_for
from traffic_service.clients.web_search import WebSearchClient, client_from_settings
import traffic_service.llm.prompts as prompts

from traffic_service.schemas import SearchResult, PlaceInfo, NewsEvent, AccidentEvent


def _as_place_keywords(place: Any) -> List[str]:
    if place is None:
        return []
    if isinstance(place, PlaceInfo):
        kws = [place.city, place.county, place.state, place.country, place.display_name]
        return [k for k in kws if k]
    if isinstance(place, dict):
        kws = [
            place.get("city"),
            place.get("county"),
            place.get("state"),
            place.get("country"),
            place.get("display_name"),
        ]
        return [k for k in kws if k]
    return [str(place)]


def _items_to_docs(items: List[Any]) -> List[WebDocument]:
    docs: List[WebDocument] = []
    for it in items:
        if isinstance(it, SearchResult):
            title = it.title
            snippet = it.snippet
            url = it.url
            published = it.published
        elif isinstance(it, dict):
            title = str(it.get("title", ""))
            snippet = str(it.get("snippet", ""))
            url = str(it.get("url", ""))
            published = it.get("published")
        else:
            continue

        if not url:
            continue

        docs.append(
            WebDocument(
                title=title,
                url=url,
                domain=_domain_from_url(url),
                published=published if isinstance(published, str) else None,
                content=snippet or "",
                snippet=snippet or None,
            )
        )
    return docs


def _norm_query(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def _unique_domains(urls: List[str]) -> List[str]:
    doms = []
    seen = set()
    for u in urls or []:
        d = _domain_from_url(u)
        if not d:
            continue
        if d in seen:
            continue
        seen.add(d)
        doms.append(d)
    return doms


class EventExtractorAgent:
    def __init__(self, llm: OpenAICompatibleClient, *, model: Optional[str] = None) -> None:
        self.llm = llm
        self.model = model

    def extract_event(self, doc: WebDocument, *, kind: str, rule_score: float = 0.0) -> ExtractedEvent:
        user = prompts.EVENT_EXTRACT_USER_TEMPLATE.format(
            kind=kind,
            title=doc.title,
            url=doc.url,
            published=doc.published or "",
            content=(doc.content or doc.snippet or "")[:9000],
        )
        j = self.llm.chat_json(
            [LLMChatMessage("system", prompts.EVENT_EXTRACT_SYSTEM), LLMChatMessage("user", user)],
            model=self.model,
            temperature=0.0,
            max_tokens=1200,
        )

        return ExtractedEvent(
            is_relevant=bool(j.get("is_relevant", False)),
            event_type=str(j.get("event_type", kind) or kind),
            title=str(j.get("title", doc.title) or doc.title),
            summary=str(j.get("summary", "") or ""),
            location_text=str(j.get("location_text", "") or ""),
            time_text=str(j.get("time_text", "") or ""),
            start_time_local=(j.get("start_time_local") or None),
            end_time_local=(j.get("end_time_local") or None),
            severity=str(j.get("severity", "unknown") or "unknown"),
            roads=list(j.get("roads") or []),
            entities=list(j.get("entities") or []),
            evidence=list(j.get("evidence") or []),
            notes=str(j.get("notes", "") or ""),
            source_url=doc.url,
            source_domain=doc.domain,
            published=doc.published,
            rule_score=float(rule_score),
            llm_confidence=float(j.get("confidence", 0.5) or 0.5),
        )


class EventVerifierAgent:
    def __init__(self, llm: OpenAICompatibleClient, *, model: Optional[str] = None) -> None:
        self.llm = llm
        self.model = model

    def verify_cluster(
        self,
        cluster: List[ExtractedEvent],
        docs_by_url: Dict[str, WebDocument],
        *,
        kind: str,
        target_context: str,
        max_sources: int = 3,
    ) -> VerifiedEvent:
        cluster_sorted = sorted(cluster, key=lambda e: e.rule_score, reverse=True)[:max_sources]

        blocks: List[str] = []
        for idx, ev in enumerate(cluster_sorted):
            doc = docs_by_url.get(ev.source_url)
            content = (doc.content or doc.snippet or "") if doc else ""
            blocks.append(
                textwrap.dedent(
                    f"""\
                    [SOURCE {idx}]
                    URL: {ev.source_url}
                    DOMAIN: {ev.source_domain}
                    PUBLISHED: {ev.published or ""}
                    TITLE: {ev.title}
                    EXTRACTED_EVENT_JSON: {asdict(ev)}
                    TEXT:
                    {content[:7000]}
                    """
                ).strip()
            )

        sources_block = "\n\n".join(blocks)
        user = prompts.EVENT_VERIFY_USER_TEMPLATE.format(
            kind=kind, target_context=target_context, sources_block=sources_block
        )

        j = self.llm.chat_json(
            [LLMChatMessage("system", prompts.EVENT_VERIFY_SYSTEM), LLMChatMessage("user", user)],
            model=self.model,
            temperature=0.0,
            max_tokens=1400,
        )

        verdict = str(j.get("verdict", "uncertain"))
        confidence = float(j.get("confidence", 0.5))
        consolidated = dict(j.get("consolidated") or {})
        key_evidence = list(j.get("key_evidence") or [])
        contradictions = list(j.get("contradictions") or [])

        verdict_bonus = {"supported": 0.25, "uncertain": 0.0, "contradicted": -0.3}.get(verdict, 0.0)
        best_rule = max((e.rule_score for e in cluster), default=0.0)
        corroboration = 0.12 * max(0, len({e.source_domain for e in cluster}) - 1)

        score = best_rule + corroboration + verdict_bonus + 0.5 * confidence

        # Include contradictions in key_evidence for traceability.
        if contradictions:
            key_evidence = key_evidence + [{"note": c} for c in contradictions]

        return VerifiedEvent(
            verdict=verdict,
            confidence=confidence,
            consolidated=consolidated,
            key_evidence=key_evidence,
            sources=[e.source_url for e in cluster_sorted],
            score=score,
        )


class _BaseWebEventAgent(ActionAgent):
    """
    Adds a bounded refine loop on top of search -> extract -> cluster -> verify.
    """

    def __init__(
        self,
        *,
        kind: str,
        extractor_llm: Optional[OpenAICompatibleClient] = None,
        verifier_llm: Optional[OpenAICompatibleClient] = None,
        refiner_llm: Optional[OpenAICompatibleClient] = None,
        search_client: Optional[WebSearchClient] = None,
        fetch_fulltext: bool = False,
        max_docs_for_llm: int = 20,
        doc_score_threshold: float = -0.25,
        topk: int = 5,
        concurrent_fetch: int = 6,
        concurrent_llm: int = 4,
        refine_enabled: bool = True,
        refine_max_iters: int = 1,
        refine_total_query_budget: int = 24,
        refine_per_query: int = 5,
        refine_min_confidence: float = 0.60,
        refine_debug: bool = False,
        concurrent_search: int = 6,
    ) -> None:
        self.kind = kind
        self.fetch_fulltext = bool(fetch_fulltext)
        self.max_docs_for_llm = int(max_docs_for_llm)
        self.doc_score_threshold = float(doc_score_threshold)
        self.topk = int(topk)

        self.fetcher = ContentFetcher()
        self.extractor_llm = extractor_llm or client_for("extractor")
        self.verifier_llm = verifier_llm or client_for("verifier")
        self.refiner_llm = refiner_llm or client_for("extractor")

        self.extractor = EventExtractorAgent(self.extractor_llm)
        self.verifier = EventVerifierAgent(self.verifier_llm)

        self.search_client: WebSearchClient = search_client or client_from_settings()

        self._fetch_sem = asyncio.Semaphore(int(concurrent_fetch))
        self._llm_sem = asyncio.Semaphore(int(concurrent_llm))
        self._search_sem = asyncio.Semaphore(int(concurrent_search))

        self.refine_enabled = bool(refine_enabled)
        self.refine_max_iters = int(max(0, refine_max_iters))
        self.refine_total_query_budget = int(max(0, refine_total_query_budget))
        self.refine_per_query = int(max(1, refine_per_query))
        self.refine_min_confidence = float(refine_min_confidence)
        self.refine_debug = bool(refine_debug)

    async def _maybe_fetch(self, doc: WebDocument) -> WebDocument:
        if not self.fetch_fulltext:
            return doc
        async with self._fetch_sem:
            txt = await self.fetcher.fetch_text(doc.url)
        if txt:
            doc.content = txt
        return doc

    def _needs_refine(self, best: Optional[VerifiedEvent]) -> bool:
        """
        Refine when evidence is weak, missing fields, or sources are concentrated.
        """
        if best is None:
            return True

        verdict = (best.verdict or "").strip().lower()
        if verdict != "supported":
            return True

        if float(best.confidence or 0.0) < self.refine_min_confidence:
            return True

        cons = best.consolidated or {}
        if not (cons.get("event_time") or ""):
            return True
        if not (cons.get("location_text") or ""):
            return True

        doms = _unique_domains(best.sources or [])
        if len(doms) < 2:
            return True

        return False

    def _derive_gaps(
        self,
        *,
        best: Optional[VerifiedEvent],
        best_cluster: List[ExtractedEvent],
    ) -> str:
        gaps: List[str] = []

        if best is None:
            gaps.append("No verified event produced (no cluster passed extraction+verification).")
            gaps.append("Need more authoritative / corroborating sources.")
            return "\n".join(gaps)

        cons = best.consolidated or {}

        if not cons.get("event_time"):
            gaps.append("Missing event_time (consolidated.event_time is null/empty).")

        if not cons.get("location_text"):
            gaps.append("Missing/unclear location_text (consolidated.location_text is null/empty).")

        locs = {(" ".join((e.location_text or "").lower().split())) for e in best_cluster if (e.location_text or "").strip()}
        times = {(" ".join((e.time_text or "").lower().split())) for e in best_cluster if (e.time_text or "").strip()}

        if len(locs) >= 3:
            gaps.append(f"Divergent location descriptions across sources (distinct_location_texts={len(locs)}).")
        if len(times) >= 3:
            gaps.append(f"Divergent time descriptions across sources (distinct_time_texts={len(times)}).")

        roads: List[str] = []
        seen_r = set()
        for e in best_cluster:
            for r in (e.roads or []):
                r2 = r.strip()
                if not r2:
                    continue
                k = r2.lower()
                if k in seen_r:
                    continue
                seen_r.add(k)
                roads.append(r2)

        if self.kind in {"accident", "accidents", "crash", "collision"} and not roads:
            gaps.append("Road segment unclear (no road ref; need interchange/exit/direction/closure details).")
        elif not roads:
            gaps.append("Road reference missing/weak (consider adding highway/route/interchange terms).")

        doms = _unique_domains(best.sources or [])
        if len(doms) < 2:
            gaps.append(f"Source diversity low (domains={doms}).")

        verdict = (best.verdict or "").lower().strip()
        if verdict == "uncertain":
            gaps.append("Verifier verdict is uncertain; need stronger cross-source corroboration.")

        if not gaps:
            gaps.append("Evidence still weak/underspecified; try narrower queries with authoritative sources.")

        return "\n".join(gaps)

    async def _generate_refined_queries(
        self,
        *,
        target_context: str,
        best: Optional[VerifiedEvent],
        gaps: str,
        previous_queries: List[str],
        remaining_budget: int,
    ) -> List[str]:
        verdict = "uncertain"
        consolidated_summary = "{}"
        if best is not None:
            verdict = str(best.verdict or "uncertain")
            try:
                consolidated_summary = json.dumps(best.consolidated or {}, ensure_ascii=False)
            except Exception:
                consolidated_summary = str(best.consolidated or {})

        user = prompts.refine_search_query_prompt(
            target_context=target_context,
            verdict=verdict,
            consolidated_summary=consolidated_summary,
            gaps=gaps,
            previous_queries=list(previous_queries or []),
        )

        def _call_llm() -> Dict[str, Any]:
            return self.refiner_llm.chat_json(
                [LLMChatMessage("system", "You are a web search query refinement model."), LLMChatMessage("user", user)],
                model=None,
                temperature=0.0,
                max_tokens=900,
            )

        async with self._llm_sem:
            data = await asyncio.to_thread(_call_llm)

        queries = data.get("queries", []) or []
        out = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

        out2: List[str] = []
        seen = set(_norm_query(q) for q in (previous_queries or []))
        for q in out:
            nq = _norm_query(q)
            if not nq:
                continue
            if nq in seen:
                continue
            seen.add(nq)
            out2.append(q.strip())
            if len(out2) >= min(12, max(0, remaining_budget)):
                break

        if not out2:
            out2 = out[: min(12, max(0, remaining_budget))]

        return out2

    async def _search_queries(self, queries: List[str], *, per_query: int) -> List[SearchResult]:
        all_items: List[SearchResult] = []

        async def _one(q: str) -> None:
            async with self._search_sem:
                try:
                    items = await self.search_client.search(q, num_results=int(per_query))
                    all_items.extend(items or [])
                except Exception:
                    return

        tasks = [asyncio.create_task(_one(q)) for q in (queries or []) if isinstance(q, str) and q.strip()]
        if tasks:
            await asyncio.gather(*tasks)

        return all_items

    async def _run_round(
        self,
        *,
        docs: List[WebDocument],
        place_keywords: List[str],
        window_start: str,
        window_end: str,
        target_context: str,
        extracted_cache: Dict[str, ExtractedEvent],
    ) -> Tuple[List[VerifiedEvent], Optional[VerifiedEvent], List[ExtractedEvent]]:
        """
        Single pass: score+filter -> (optional fetch fulltext) -> extract -> cluster -> verify
        Returns: verified_list, best_verified, best_cluster_extracted
        """
        scored = [
            (d, rule_score_doc(d, place_keywords=place_keywords, window_start=window_start, window_end=window_end))
            for d in docs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [(d, s) for (d, s) in scored if s >= self.doc_score_threshold][: self.max_docs_for_llm]
        if not filtered:
            return ([], None, [])

        # maybe fetch full text (only for filtered docs)
        docs2: List[WebDocument] = []
        for d, _ in filtered:
            docs2.append(await self._maybe_fetch(d))
        docs_by_url = {d.url: d for d in docs2}

        extracted: List[ExtractedEvent] = []

        async def _extract_one(d: WebDocument, rs: float) -> None:
            # cache hit
            if d.url in extracted_cache:
                ev = extracted_cache[d.url]
                ev.rule_score = float(rs)
                extracted.append(ev)
                return

            async with self._llm_sem:
                ev = await asyncio.to_thread(self.extractor.extract_event, d, kind=self.kind, rule_score=rs)

            extracted_cache[d.url] = ev
            extracted.append(ev)

        tasks = []
        for d, rs in filtered:
            tasks.append(asyncio.create_task(_extract_one(docs_by_url[d.url], rs)))
        if tasks:
            await asyncio.gather(*tasks)

        extracted = [e for e in extracted if e.is_relevant and e.llm_confidence >= 0.35]
        if not extracted:
            return ([], None, [])

        clusters = rank_clusters(cluster_events(extracted))

        verified: List[VerifiedEvent] = []
        verified_pairs: List[Tuple[VerifiedEvent, List[ExtractedEvent]]] = []

        async def _verify_one(c: List[ExtractedEvent]) -> None:
            async with self._llm_sem:
                ve = await asyncio.to_thread(
                    self.verifier.verify_cluster,
                    c,
                    docs_by_url,
                    kind=self.kind,
                    target_context=target_context,
                    max_sources=3,
                )
            verified.append(ve)
            verified_pairs.append((ve, c))

        # Verify sequentially to control token usage / concurrency
        for c in clusters[: max(self.topk * 2, 10)]:
            await _verify_one(c)

        # Drop contradicted
        verified = [v for v in verified if (v.verdict or "").lower().strip() != "contradicted"]
        verified.sort(key=lambda x: float(x.score or 0.0), reverse=True)

        best = verified[0] if verified else None
        best_cluster: List[ExtractedEvent] = []
        if best is not None:
            for ve, c in verified_pairs:
                if ve is best:
                    best_cluster = c
                    break

        return (verified[: self.topk], best, best_cluster)

    async def _extract_and_verify_with_refine(
        self,
        *,
        docs: List[WebDocument],
        place_keywords: List[str],
        window_start: str,
        window_end: str,
        target_context: str,
        previous_queries: List[str],
    ) -> List[VerifiedEvent]:
        """
        # Refinement loop: run round0 first; if refinement is needed:
           regenerate queries -> search -> merge docs -> rerun round
        """
        # docs dedup by url
        doc_map: Dict[str, WebDocument] = {}
        for d in docs or []:
            if not d.url:
                continue
            if d.url not in doc_map:
                doc_map[d.url] = d

        # extracted cache to avoid re-extracting same url in multiple rounds
        extracted_cache: Dict[str, ExtractedEvent] = {}

        # query history
        query_history: List[str] = [q for q in (previous_queries or []) if isinstance(q, str) and q.strip()]

        # ---- round 0 ----
        verified, best, best_cluster = await self._run_round(
            docs=list(doc_map.values()),
            place_keywords=place_keywords,
            window_start=window_start,
            window_end=window_end,
            target_context=target_context,
            extracted_cache=extracted_cache,
        )

        best_overall_list = list(verified)
        best_overall_score = float(best.score) if best is not None else float("-inf")

        if not self.refine_enabled or self.refine_max_iters <= 0:
            return verified

        remaining_budget = int(self.refine_total_query_budget)

        # ---- refine loop ----
        for it in range(self.refine_max_iters):
            if not self._needs_refine(best):
                break
            if remaining_budget <= 0:
                break

            gaps = self._derive_gaps(best=best, best_cluster=best_cluster)

            try:
                refined_queries = await self._generate_refined_queries(
                    target_context=target_context,
                    best=best,
                    gaps=gaps,
                    previous_queries=query_history,
                    remaining_budget=remaining_budget,
                )
            except Exception as e:
                # refine LLM failure -> stop refine, return best so far
                if self.refine_debug:
                    print(f"[Refine] LLM failed: {type(e).__name__}: {e}")
                break

            refined_queries = [q for q in refined_queries if isinstance(q, str) and q.strip()]
            if not refined_queries:
                break

            remaining_budget -= len(refined_queries)
            query_history.extend(refined_queries)

            if self.refine_debug:
                print(f"\n[Refine][{self.kind}] iter={it+1} remaining_budget={remaining_budget}")
                print("[Refine] gaps:\n", gaps)
                for i, q in enumerate(refined_queries, 1):
                    print(f"[Refine] {i:02d}. {q}")

            # retrieve
            try:
                new_items = await self._search_queries(refined_queries, per_query=self.refine_per_query)
            except Exception as e:
                if self.refine_debug:
                    print(f"[Refine] search failed: {type(e).__name__}: {e}")
                break

            new_docs = _items_to_docs(list(new_items))
            # merge
            added = 0
            for d in new_docs:
                if not d.url:
                    continue
                if d.url in doc_map:
                    continue
                doc_map[d.url] = d
                added += 1

            if self.refine_debug:
                print(f"[Refine] retrieved_docs={len(new_docs)} added_new_urls={added} total_urls={len(doc_map)}")

            # rerun round
            verified2, best2, best_cluster2 = await self._run_round(
                docs=list(doc_map.values()),
                place_keywords=place_keywords,
                window_start=window_start,
                window_end=window_end,
                target_context=target_context,
                extracted_cache=extracted_cache,
            )

            # update "current"
            verified, best, best_cluster = verified2, best2, best_cluster2

            # track best overall
            cur_score = float(best.score) if best is not None else float("-inf")
            if cur_score > best_overall_score:
                best_overall_score = cur_score
                best_overall_list = list(verified)

        # If the final round has no result, fall back to best_overall
        return verified if verified else best_overall_list


class NewsExtractorAgent(_BaseWebEventAgent):
    name = "NewsExtractorAgent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kind="news", **kwargs)

    async def extract_news_events(
        self,
        ctx: ExecutionContext,
        *,
        items_key: str,
        place_key: str,
        queries_key: Optional[str] = None,
    ) -> List[VerifiedEvent]:
        items = ctx.get(items_key, [])
        place = ctx.get(place_key, None)
        docs = _items_to_docs(list(items))

        # previous queries for refine (prefer the provided queries_key; otherwise fallback automatically)
        prev_q: List[str] = []
        if queries_key:
            prev_q = ctx.get(queries_key, []) or []
        else:
            prev_q = ctx.get("news_queries", []) or []

        place_keywords = _as_place_keywords(place)
        target_context = (
            f"News near {getattr(place,'display_name', place)}; "
            f"window(local): {ctx.request.local_start} - {ctx.request.local_end}; "
            f"Q: {ctx.request.question}"
        )

        return await self._extract_and_verify_with_refine(
            docs=docs,
            place_keywords=place_keywords,
            window_start=str(ctx.request.local_start),
            window_end=str(ctx.request.local_end),
            target_context=target_context,
            previous_queries=list(prev_q) if isinstance(prev_q, list) else [],
        )


class AccidentExtractorAgent(_BaseWebEventAgent):
    name = "AccidentExtractorAgent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kind="accident", **kwargs)

    async def extract_accident_events(
        self,
        ctx: ExecutionContext,
        *,
        items_key: str,
        place_key: str,
        queries_key: Optional[str] = None,
    ) -> List[VerifiedEvent]:
        items = ctx.get(items_key, [])
        place = ctx.get(place_key, None)
        docs = _items_to_docs(list(items))

        # previous queries for refine (prefer the provided queries_key; otherwise fallback automatically)
        prev_q: List[str] = []
        if queries_key:
            prev_q = ctx.get(queries_key, []) or []
        else:
            prev_q = ctx.get("accident_queries", []) or []

        place_keywords = _as_place_keywords(place)
        target_context = (
            f"Accidents near {getattr(place,'display_name', place)}; "
            f"window(local): {ctx.request.local_start} - {ctx.request.local_end}; "
            f"Q: {ctx.request.question}"
        )

        return await self._extract_and_verify_with_refine(
            docs=docs,
            place_keywords=place_keywords,
            window_start=str(ctx.request.local_start),
            window_end=str(ctx.request.local_end),
            target_context=target_context,
            previous_queries=list(prev_q) if isinstance(prev_q, list) else [],
        )


class VerifiedEventConverterAgent(ActionAgent):
    name = "VerifiedEventConverterAgent"

    async def convert(self, ctx: ExecutionContext, *, kind: str, input_key: str, topk: int = 5) -> list[Any]:
        kind2 = str(kind).lower()
        verified = ctx.get(input_key, []) or []
        if not isinstance(verified, list):
            verified = [verified]

        def _score(x: Any) -> float:
            try:
                return float(getattr(x, "score", None) or (x.get("score") if isinstance(x, dict) else 0.0) or 0.0)
            except Exception:
                return 0.0

        verified = sorted(verified, key=_score, reverse=True)[: int(topk)]

        def _get(obj: Any, k: str, default=None):
            if isinstance(obj, dict):
                return obj.get(k, default)
            return getattr(obj, k, default)

        if kind2 == "news":
            out: list[NewsEvent] = []
            for v in verified:
                d = _get(v, "consolidated", {}) or {}
                srcs = _get(v, "sources", []) or []
                out.append(
                    NewsEvent(
                        event_time=d.get("event_time"),
                        location_text=d.get("location_text"),
                        summary=str(d.get("summary") or ""),
                        category=d.get("category"),
                        tone=d.get("tone"),
                        source_url=(srcs[0] if srcs else None),
                        confidence=float(_get(v, "confidence", 0.5) or 0.5),
                    )
                )
            return out

        out2: list[AccidentEvent] = []
        for v in verified:
            d = _get(v, "consolidated", {}) or {}
            srcs = _get(v, "sources", []) or []
            out2.append(
                AccidentEvent(
                    event_time=d.get("event_time"),
                    location_text=d.get("location_text"),
                    collision_type=d.get("collision_type"),
                    fatalities=d.get("fatalities"),
                    injuries=d.get("injuries"),
                    conditions=d.get("conditions"),
                    summary=str(d.get("summary") or ""),
                    source_url=(srcs[0] if srcs else None),
                    confidence=float(_get(v, "confidence", 0.5) or 0.5),
                )
            )
        return out2
