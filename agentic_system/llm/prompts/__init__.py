# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

# ---------------------------------------------------------------------
# Centralized prompts for the whole project.
# Agents should import prompts ONLY from: agentic_system.llm.prompts
# ---------------------------------------------------------------------

PROMPT_VERSION = "v1.0"

# =========================================================
# NL parse (canonical implementation: *_prompt)
# =========================================================
def nl_parse_system() -> str:
    return (
        "You are a parser for traffic/logistics questions.\n"
        "Extract USER location (where the user/company is based or currently located),\n"
        "route key area (optional), destination (optional), and time constraints.\n"
        "Return STRICT JSON only. No extra text.\n"
        "Never guess missing facts.\n"
    )


def nl_parse_user(
    *,
    question: str,
    request_time_utc: Optional[str] = None,
    refine: bool = False,
) -> str:
    rt = request_time_utc or ""

    base = f"""
Question:
{question}

Request time (UTC, reference only; use it ONLY when the question contains relative time like "today/this evening"):
{rt}

Return STRICT JSON ONLY with this schema:
{{
  "user_location_query": "string or null (USER/COMPANY base or current location; geocodable; may include organization name)",
  "route_location_query": "string or null (route corridor / key area; e.g., 'I-5 North / Newhall Pass area')",
  "destination_location_query": "string or null (destination only)",
  "road_ref": "string or null",
  "direction_hint": "string or null",

  "time_kind": "none" | "point" | "range",
  "time_point_local": "YYYY-MM-DD HH:MM" or null,
  "time_start_local": "YYYY-MM-DD HH:MM" or null,
  "time_end_local": "YYYY-MM-DD HH:MM" or null,
  "time_text": "string or null",
  "time_confidence": 0.0,

  "confidence": 0.0,
  "notes": "string or null",

  "primary_location_query": "string or null (MUST equal user_location_query; backward compatibility)"
}}

CRITICAL DEFINITIONS:
- user_location_query MUST be where the USER/COMPANY is based or located.
  If the question says "We are Allied Beverages Inc., based in Los Angeles":
  => user_location_query = "Allied Beverages Inc., Los Angeles, California, United States"
- route_location_query is the route key area / incident corridor mentioned for travel impact (optional).
- destination_location_query is ONLY the destination (optional).
- primary_location_query MUST equal user_location_query (do not invent different values).

LOCATION RULES (NO MERGING):
- Do NOT merge user location, route area, and destination into one field.
- Keep each field semantically clean:
  - user_location_query: user/company location only
  - route_location_query: route key area only
  - destination_location_query: destination only

ORGANIZATION RULES:
- If an organization/company name is explicitly given, include it in user_location_query
  together with city/state/country if present.

DIRECTION RULES:
- Normalize direction to: northbound/southbound/eastbound/westbound when possible.

TIME RULES:
- Interpret times as LOCAL time near the user_location_query (if available), otherwise use the most relevant mentioned place.
- If the question states a single time -> time_kind="point" and fill time_point_local.
- If the question states a time range -> time_kind="range" and fill time_start_local/time_end_local.
- If no explicit time -> time_kind="none" and keep time_* null; keep time_text if there is vague time like "this evening".

NO GUESSING:
- Do NOT invent locations or times.
- If a field is not stated, use null and explain briefly in notes.

EXAMPLE (DO NOT COPY VALUES IF NOT PRESENT):
Input: "We are Allied Beverages Inc., based in Los Angeles... route includes I-5 North / Newhall Pass... to Downtown LA this evening"
Output:
{{
  "user_location_query": "Allied Beverages Inc., Los Angeles, California, United States",
  "route_location_query": "I-5 North / Newhall Pass area, California, United States",
  "destination_location_query": "Downtown Los Angeles, California, United States",
  "road_ref": "I-5",
  "direction_hint": "northbound",
  "time_kind": "none",
  "time_point_local": null,
  "time_start_local": null,
  "time_end_local": null,
  "time_text": "this evening",
  "time_confidence": 0.4,
  "confidence": 0.8,
  "notes": null,
  "primary_location_query": "Allied Beverages Inc., Los Angeles, California, United States"
}}

Now return the JSON only.
""".strip()
    return base

# =========================================================
# Query generation for web search (QueryGeneratorAgent uses this)
# =========================================================
def search_query_prompt(
    *,
    kind: str,
    place_hint: str,
    road_hint: Optional[str],
    local_start: str,
    local_end: str,
    question: str,
) -> str:
    """
    Must induce model output:
      {"queries": ["...", "...", ...]}
    """
    kind_norm = (kind or "").strip().lower()
    if kind_norm in {"accident", "accidents", "crash", "collision"}:
        kind_label = "traffic accidents / crashes"
    else:
        kind_label = "traffic-related news"

    road_part = f"Road hint: {road_hint}\n" if road_hint else "Road hint: (none)\n"

    return (
        "You are a web search query generation model.\n"
        "Your job: generate high-quality search queries to find relevant sources.\n"
        "\n"
        "Output format:\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "queries": ["string", "string", "..."]\n'
        "}\n"
        "\n"
        "Context:\n"
        f"Kind: {kind_label}\n"
        f"Place hint: {place_hint}\n"
        f"{road_part}"
        f"Local time window: {local_start} to {local_end}\n"
        f"User question: {question}\n"
        "\n"
        "Rules:\n"
        "- Output JSON ONLY. No markdown.\n"
        "- Provide 8 to 12 queries.\n"
        "- Mix broad queries (high recall) and specific queries (high precision).\n"
        "- Include at least 3 queries that contain the place hint.\n"
        "- If road hint exists, include it in at least 3 queries.\n"
        "- Include time window in multiple styles:\n"
        "  - exact date (YYYY-MM-DD)\n"
        "  - natural date (e.g., Oct 23 2017)\n"
        "  - time anchors (e.g., \"6 PM\") when present in the question\n"
        "- Avoid quotes unless necessary.\n"
        "- Prefer English queries.\n"
        "\n"
        "Now return the JSON.\n"
    )


# =========================================================
# Event extraction + verification (agents/web/events.py uses these constants)
# =========================================================
EVENT_EXTRACT_SYSTEM = (
    "You are an information extraction model.\n"
    "You extract a single event from ONE web document.\n"
    "Return STRICT JSON ONLY. No markdown, no explanation.\n"
    "Do not hallucinate facts not present in the document.\n"
    "If uncertain, set fields to null or empty lists.\n"
)

# NOTE: used with .format(...), so all JSON braces must be doubled {{ }}
EVENT_EXTRACT_USER_TEMPLATE = """
Task: Extract ONE {kind} event from the following web document.
If the document is not about a real-world event relevant to the task, set "is_relevant": false.

Document:
TITLE: {title}
URL: {url}
PUBLISHED: {published}
CONTENT:
{content}

Return STRICT JSON ONLY with this schema:
{{
  "is_relevant": true,
  "event_type": "string (use a short label; if unknown use '{kind}')",
  "title": "string (event headline; prefer document title if it matches)",
  "summary": "string (2-4 sentences, factual)",
  "location_text": "string (where it happened; may be city/road/interchange) or ''",
  "time_text": "string (as written in the doc) or ''",
  "start_time_local": "YYYY-MM-DD HH:MM" or null,
  "end_time_local": "YYYY-MM-DD HH:MM" or null,
  "severity": "low|medium|high|unknown",
  "roads": ["list of road/highway names/ref like I-5, US-101"],
  "entities": ["key organizations/locations/people if explicitly mentioned"],
  "evidence": [
    {{"field": "summary|location_text|time_text|...", "quote": "short quote (<=25 words)"}}
  ],
  "notes": "string or ''",
  "confidence": 0.0
}}

Rules:
- Output JSON ONLY.
- "confidence" is your confidence in the extraction (0.0-1.0).
- Use null for unknown start/end times. Do NOT guess times.
- If you can infer only a date but not time, keep start_time_local/end_time_local null and store the date in time_text.
- Keep evidence quotes short and directly copied from the content when possible.
""".strip()


EVENT_VERIFY_SYSTEM = (
    "You are a cross-source event verifier.\n"
    "You will be given multiple extracted events from different sources about the same suspected event.\n"
    "Your job: decide if the event is supported, uncertain, or contradicted.\n"
    "Return STRICT JSON ONLY. No markdown.\n"
    "Do not invent details not supported by sources.\n"
)

# NOTE: used with .format(...), escape braces with double {{ }}
EVENT_VERIFY_USER_TEMPLATE = """
You are verifying a {kind} event cluster for this target context:
{target_context}

You are given up to several sources (extracted JSON + text). Use them to verify.

SOURCES:
{sources_block}

Return STRICT JSON ONLY with this schema:
{{
  "verdict": "supported|uncertain|contradicted",
  "confidence": 0.0,
  "consolidated": {{
    "event_time": "YYYY-MM-DD HH:MM" or null,
    "location_text": "string or null",
    "summary": "string (2-4 sentences)",
    "category": "string or null",
    "tone": 0.0 or null,

    "collision_type": "string or null",
    "fatalities": 0 or null,
    "injuries": 0 or null,
    "conditions": "string or null"
  }},
  "key_evidence": [
    {{"source_url": "string", "note": "short reason", "quote": "optional short quote <=25 words"}}
  ],
  "contradictions": [
    "string"
  ]
}}

Rules:
- Output JSON ONLY.
- "supported": at least 2 independent sources corroborate core facts (time/location/what happened),
  OR 1 highly credible source with strong details.
- "uncertain": some hints but not enough corroboration.
- "contradicted": sources clearly conflict (e.g., different locations/events) or indicate the event did not happen.
- consolidated.summary must be factual and match the sources.
- If time is unclear, set consolidated.event_time = null (do NOT guess).
- Fill accident-specific fields only if {kind} is accident/crash related and the sources contain the info; otherwise keep them null.
- Keep key_evidence short and cite URLs from the sources.
""".strip()


# =========================================================
# Dataset Record builder (RecordBuilder uses this function)
# =========================================================
def record_builder_system_prompt() -> str:
    return (
        "You are a traffic-impact reasoning assistant.\n"
        "You will receive a USER message that contains:\n"
        "1) A header with coordinates/time window and optional roadway attributes.\n"
        "2) The phrase: [chunk token] [image token]\n"
        "   - [chunk token] stands for structured context chunks (POI/News/Accident/HopSensor).\n"
        "   - [image token] stands for an optional satellite image near the location.\n"
        "3) A question to answer.\n"
        "\n"
        "Your task:\n"
        "- Answer the user question using ONLY the provided context.\n"
        "- If evidence is missing or uncertain, say what is unknown and give a cautious answer.\n"
        "- Prefer concrete, operational guidance.\n"
        "- Do not mention tokens like '[chunk token]' or '[image token]' in your final answer.\n"
        "- Do not fabricate numbers, closures, fatalities, or exact delay durations unless stated.\n"
        "\n"
        "Output:\n"
        "- Return a helpful answer in plain text.\n"
    )


def refine_search_query_prompt(
    *,
    target_context: str,
    verdict: str,
    consolidated_summary: str,
    gaps: str,
    previous_queries: list[str],
) -> str:
    return f"""
    You are a web search query refinement model. You will be given a target context and the current verification result for an event cluster. Generate improved search queries that can resolve the missing/uncertain parts (time, location, road segment, closure details). Return STRICT JSON ONLY.

    Target context:
    {target_context}

    Current verification verdict:
    {verdict}

    Known consolidated summary (may be incomplete):
    {consolidated_summary}

    Observed gaps / uncertainties:
    {gaps}

    Previous queries (for reference; avoid repeating them verbatim):
    {previous_queries}

    Return STRICT JSON ONLY:
    {{
      "queries": ["string", "string", "..."]
    }}

    Rules:
    - Output JSON ONLY. No markdown.
    - Provide 8 to 12 queries.
    - Prefer queries that are likely to return primary or authoritative sources (e.g., CHP reports, DOT road closures, local news).
    - Include alternative phrasings for the same location (interchange name, nearby city, county).
    - Include road ref and direction if known.
    - Include date/time anchors in multiple styles when possible.
    - Do NOT fabricate facts; only reformulate and broaden/narrow queries.
    """



__all__ = [
    "PROMPT_VERSION",
    # nl parse
    "nl_parse_system",
    "nl_parse_user",
    "nl_parse",
    # query gen
    "search_query_prompt",
    # events
    "EVENT_EXTRACT_SYSTEM",
    "EVENT_EXTRACT_USER_TEMPLATE",
    "EVENT_VERIFY_SYSTEM",
    "EVENT_VERIFY_USER_TEMPLATE",
    # record builder
    "record_builder_system_prompt",
    "refine_search_query_prompt"
]
