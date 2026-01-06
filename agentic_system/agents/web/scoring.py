# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from agentic_system.utils.time import parse_local_dt

try:
    from rapidfuzz.fuzz import token_set_ratio  # type: ignore
except Exception:
    token_set_ratio = None  # type: ignore


@dataclass
class WebDocument:
    title: str
    url: str
    domain: str
    published: Optional[str]
    content: str
    snippet: Optional[str] = None


@dataclass
class ExtractedEvent:
    is_relevant: bool
    event_type: str
    title: str
    summary: str
    location_text: str
    time_text: str
    start_time_local: Optional[str]
    end_time_local: Optional[str]
    severity: str
    roads: List[str]
    entities: List[str]
    evidence: List[Dict[str, str]]
    notes: str
    source_url: str
    source_domain: str
    published: Optional[str]
    rule_score: float = 0.0
    llm_confidence: float = 0.5


@dataclass
class VerifiedEvent:
    verdict: str  # supported/uncertain/contradicted
    confidence: float
    consolidated: Dict[str, Any]
    key_evidence: List[Dict[str, Any]]
    sources: List[str]
    score: float
    contradictions: List[str] = field(default_factory=list) 


def _domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc or url
    except Exception:
        return url


def _root_domain(domain: str) -> str:
    d = (domain or "").lower().strip()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/")[0]
    parts = d.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return d


TRUSTED_SUFFIXES = (".gov", ".edu", ".mil")
HIGH_QUALITY_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "washingtonpost.com",
    "latimes.com", "theguardian.com", "npr.org", "wsj.com", "bloomberg.com",
}
LOW_QUALITY_HINTS = ("blogspot.", "wordpress.", "medium.com")
LOW_QUALITY_DOMAINS = {"facebook.com", "x.com", "twitter.com", "instagram.com", "tiktok.com", "youtube.com", "reddit.com"}


def rule_score_doc(
    doc: WebDocument,
    *,
    place_keywords: List[str],
    window_start: Optional[str],
    window_end: Optional[str],
) -> float:
    score = 0.0
    root = _root_domain(doc.domain or doc.url)

    if any(root.endswith(suf.lstrip(".")) for suf in TRUSTED_SUFFIXES) or root in HIGH_QUALITY_DOMAINS:
        score += 0.35
    if root in LOW_QUALITY_DOMAINS:
        score -= 0.30
    if any(h in root for h in LOW_QUALITY_HINTS):
        score -= 0.15

    text = (doc.title + " " + (doc.snippet or "") + " " + (doc.content or "")).lower()
    if any(k.lower() in text for k in place_keywords if k):
        score += 0.2

    if doc.published and window_start and window_end:
        try:
            pub = parse_local_dt(doc.published)
            ws = parse_local_dt(window_start)
            we = parse_local_dt(window_end)
            if ws <= pub <= we:
                score += 0.2
            else:
                delta_days = min(abs((pub - ws).days), abs((pub - we).days))
                score += 0.05 if delta_days <= 7 else -0.1
        except Exception:
            pass

    if doc.content and len(doc.content) > 1500:
        score += 0.1
    elif doc.content and len(doc.content) < 200:
        score -= 0.1

    return max(-1.0, min(1.0, score))


def cluster_events(events: List[ExtractedEvent], *, threshold: int = 75) -> List[List[ExtractedEvent]]:
    clusters: List[List[ExtractedEvent]] = []
    for ev in events:
        if not ev.is_relevant:
            continue

        key = f"{ev.event_type} | {ev.location_text} | {ev.time_text}".strip().lower()
        placed = False

        for c in clusters:
            rep = c[0]
            rep_key = f"{rep.event_type} | {rep.location_text} | {rep.time_text}".strip().lower()

            if token_set_ratio is None:
                if key == rep_key:
                    c.append(ev)
                    placed = True
                    break
            else:
                if token_set_ratio(key, rep_key) >= threshold:
                    c.append(ev)
                    placed = True
                    break

        if not placed:
            clusters.append([ev])

    return clusters


def rank_clusters(clusters: List[List[ExtractedEvent]]) -> List[List[ExtractedEvent]]:
    def score_cluster(c: List[ExtractedEvent]) -> float:
        best = max((e.rule_score for e in c), default=0.0)
        n_sources = len({e.source_domain for e in c})
        boost = 0.12 * max(0, n_sources - 1)
        return best + boost

    return sorted(clusters, key=score_cluster, reverse=True)
