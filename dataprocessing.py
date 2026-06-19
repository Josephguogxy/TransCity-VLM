# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import os
import json
import re

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import Dataset as HFDataset, Features, Value, Sequence
from transformers import PreTrainedTokenizerBase
from collections.abc import Mapping


# =========================
# env switches
# =========================
SMARTCITY_DEBUG = os.environ.get("SMARTCITY_DEBUG", "0") == "1"
SMARTCITY_MAX_BAD_LINES = int(os.environ.get("SMARTCITY_MAX_BAD_LINES", "0"))
SMARTCITY_LOG_EVERY = int(os.environ.get("SMARTCITY_LOG_EVERY", "100000"))

# reason 样本默认强制 prompt 里同时有 [chunk token] 和 [image token]
# 因为 Stage IV evidence-grounded QA 依赖这两个 marker 的位置稳定。
SMARTCITY_REASON_ALWAYS_MARKERS = os.environ.get("SMARTCITY_REASON_ALWAYS_MARKERS", "1") == "1"

# StageD SFT warmup:
# - keep reason answer as freeform target instead of converting it back to JSON
# - optionally rewrite reason prompt to freeform evidence format at load time
SMARTCITY_REASON_KEEP_FREEFORM_ANSWER = os.environ.get("SMARTCITY_REASON_KEEP_FREEFORM_ANSWER", "0").strip().lower() in {"1", "true", "yes", "on"}
SMARTCITY_REASON_SFT_JSON_GOLD_TO_FREEFORM = os.environ.get("SMARTCITY_REASON_SFT_JSON_GOLD_TO_FREEFORM", "0").strip().lower() in {"1", "true", "yes", "on"}
SMARTCITY_REASON_SFT_MAX_CITATIONS = int(os.environ.get("SMARTCITY_REASON_SFT_MAX_CITATIONS", "4"))
SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE = os.environ.get("SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE", "0").strip().lower() in {"1", "true", "yes", "on"}
SMARTCITY_REASON_ALWAYS_THINK_TAGS = os.environ.get("SMARTCITY_REASON_ALWAYS_THINK_TAGS", "0").strip().lower() in {"1", "true", "yes", "on"}
SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX = os.environ.get("SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX", "0").strip().lower() in {"1", "true", "yes", "on"}
SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE = os.environ.get("SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE", "qwen").strip().lower()
SMARTCITY_STAGE_D_FREEFORM_OUTPUT_FORMAT = "qwen3_freeform_evidence_v1"
SMARTCITY_STAGE_D_SFT_OUTPUT_FORMAT = "qwen3_freeform_evidence_sft_warmup_v1"


# =========================
# constants / schema
# =========================
FIELD_ORDER = ["POI", "News", "Accident", "HopSensor", "HopBA"]
CHUNK_MARKER = "[chunk token]"
IMAGE_MARKER = "[image token]"

TASK2ROUTER = {"understand": 0, "prediction": 1, "reason": 2}

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

PROMPT_STYLE_QWEN3 = "qwen3"
PROMPT_STYLE_PLAIN = "plain"
PROMPT_STYLES = {PROMPT_STYLE_QWEN3, PROMPT_STYLE_PLAIN}

NO_THINK_MODES = {"no_think", "nothink", "no-think", "instant", "none", "no"}
THINK_MODES = {"think", "thinking", "cot"}

# 当前 Stage IV evidence-grounded GRPO 的 canonical schema。
# 注意：answer JSON 不再要求 confidence。
SMARTCITY_FEATURES = Features({
    "task": Value("string"),

    # 方便追踪和评估，不直接参与模型输入。
    "sample_id": Value("string"),
    "dataset_source": Value("string"),
    "question_type": Value("string"),

    "chunks": {
        "POI":       Sequence(Value("string")),
        "News":      Sequence(Value("string")),
        "Accident":  Sequence(Value("string")),
        "HopSensor": Sequence(Value("string")),
        "HopBA":     Sequence(Value("string")),
    },
    "images":   Sequence(Value("string")),
    "prompt":   Value("string"),
    "thinking": Sequence(Value("string")),
    "target":   Value("string"),
    "answer":   Value("string"),

    # Stage IV evidence-grounded GRPO metadata.
    # 全部可选；旧数据会自动补空或轻量生成。
    "reasoning_mode": Value("string"),
    "gold_decision": Value("string"),
    "evidence_catalog": Value("string"),
    "key_evidence_ids": Sequence(Value("string")),
    "key_source_types": Sequence(Value("string")),
    "as_of_time": Value("string"),
    "perturbation_type": Value("string"),
    "expected_behavior": Value("string"),
    "reward_meta": Value("string"),
})


# =========================
# rank/debug
# =========================
def _is_rank0() -> bool:
    rk = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "-1"
    try:
        return int(rk) in (-1, 0)
    except Exception:
        return True


def _dprint(*args, **kwargs):
    if _is_rank0() and SMARTCITY_DEBUG:
        print(*args, **kwargs)


# =========================
# generic helpers
# =========================
def _compact_text(x: Any, max_chars: int = 300) -> str:
    s = str(x or "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)].rstrip() + "…"


def _safe_id(x: Any) -> str:
    s = str(x or "sample").strip()
    s = re.sub(r"[^A-Za-z0-9_.:@\-]+", "_", s)
    return s[:180] or "sample"


def _json_string_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(x)


def _parse_json_like(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_reward_meta(x: Any) -> Dict[str, Any]:
    y = _parse_json_like(x)
    return y if isinstance(y, dict) else {}


def _as_string_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        y = _parse_json_like(s)
        if isinstance(y, list):
            return [str(v).strip() for v in y if str(v).strip()]
        return [v.strip() for v in re.split(r"[,;\n]", s) if v.strip()]
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [str(x).strip()] if str(x).strip() else []


def _get_ci(d: Dict[str, Any] | None, key: str, default=None, aliases: tuple[str, ...] = ()) -> Any:
    """大小写不敏感取值；支持别名。"""
    if d is None:
        return default
    if not isinstance(d, Mapping):
        try:
            d = dict(d)
        except Exception:
            return default
    if key in d:
        return d[key]
    for a in aliases:
        if a in d:
            return d[a]
    wanted = {key.lower(), *(a.lower() for a in aliases)}
    for k, v in d.items():
        try:
            if str(k).lower() in wanted:
                return v
        except Exception:
            continue
    return default


def _get_meta_field(ex: Dict[str, Any], rm: Dict[str, Any], key: str, default=None):
    v = _get_ci(ex, key, default=None)
    if v is None or v == "":
        v = rm.get(key, default)
    if v is None:
        v = default
    return v


def _normalize_chunks_ci(raw: Dict[str, Any] | None) -> Dict[str, list[str]]:
    """按 FIELD_ORDER 规范化 chunks，大小写不敏感；缺失字段补空列表。"""
    out: Dict[str, list[str]] = {}
    src = raw if isinstance(raw, dict) else {}
    for canon in FIELD_ORDER:
        val = _get_ci(src, canon, default=[])
        if val is None:
            arr = []
        elif isinstance(val, list):
            arr = val
        else:
            arr = [val]
        out[canon] = [str(s) for s in arr if s is not None and str(s).strip()]
    return out


def _as_list_text(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                out.append(json.dumps(v, ensure_ascii=False))
            else:
                t = str(v).strip()
                if t:
                    out.append(t)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    s = str(x).strip()
    return [s] if s else []


# =========================
# messages helpers
# =========================
def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for seg in content:
            if isinstance(seg, dict):
                if seg.get("type") == "text" or "text" in seg:
                    t = str(seg.get("text", "") or "").strip()
                    if t:
                        parts.append(t)
                elif "content" in seg:
                    t = _content_to_text(seg.get("content"))
                    if t.strip():
                        parts.append(t.strip())
            else:
                t = str(seg).strip()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return _content_to_text(content.get("content"))
    return str(content)


def _extract_message_text(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    out: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() == role.lower():
            txt = _content_to_text(msg.get("content"))
            if txt.strip():
                out.append(txt.strip())
    return "\n".join(out)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    obj = _parse_json_like(s)
    if isinstance(obj, dict):
        return obj

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = _parse_json_like(s[start:i + 1])
                return obj if isinstance(obj, dict) else None
    return None


# =========================
# evidence catalog fallback helpers
# =========================
def _make_link(sample_id: str, path: str) -> str:
    return f"dataset://sample/{_safe_id(sample_id)}/{path.strip('/')}"


def _add_catalog_item(
    catalog: List[Dict[str, Any]],
    *,
    eid: str,
    source_type: str,
    modality: str,
    handle: str,
    link: str,
    present: bool,
    role: str,
    should_use_as_support: bool,
    should_mention_in_limitations: bool,
    legacy_converted: bool,
) -> None:
    catalog.append({
        "id": eid,
        "source_type": source_type,
        "modality": modality,
        "status": "present" if present else "missing",
        "role": role,
        "handle": _compact_text(handle, 320),
        "link": link,
        "present": bool(present),
        "should_use_as_support": bool(should_use_as_support),
        "should_mention_in_limitations": bool(should_mention_in_limitations),
        "legacy_converted": bool(legacy_converted),
        "verified": True,
    })


def _density_summary(data: Dict[str, Any]) -> str:
    parts: List[str] = []

    if isinstance(data.get("population_density"), dict):
        raw = data["population_density"].get("raw")
        parts.append(f"population_density={raw}" if raw else "population_density=" + json.dumps(data["population_density"], ensure_ascii=False))
    elif data.get("population_density") is not None:
        parts.append(f"population_density={data.get('population_density')}")

    if isinstance(data.get("housing_density"), dict):
        raw = data["housing_density"].get("raw")
        parts.append(f"housing_density={raw}" if raw else "housing_density=" + json.dumps(data["housing_density"], ensure_ascii=False))
    elif data.get("housing_density") is not None:
        parts.append(f"housing_density={data.get('housing_density')}")

    if data.get("population_density_km2") is not None:
        parts.append(f"population_density_km2={data.get('population_density_km2')}")
    if data.get("housing_density_km2") is not None:
        parts.append(f"housing_density_km2={data.get('housing_density_km2')}")

    return "; ".join([p for p in parts if p])


def _site_meta_summary(data: Dict[str, Any]) -> Tuple[str, str]:
    sensor = data.get("sensor") if isinstance(data.get("sensor"), dict) else {}
    if data.get("sensor_id") or sensor:
        sid = data.get("sensor_id") or sensor.get("sensor_id") or ""
        lon = sensor.get("longitude", "")
        lat = sensor.get("latitude", "")
        return "sensor_meta", _compact_text(f"sensor_id={sid}, lon={lon}, lat={lat}", 240)

    if data.get("utility_name"):
        return "utility_meta", _compact_text(f"utility_name={data.get('utility_name')}", 240)

    return "site_meta", "site metadata available"


def _build_catalog_from_new_data(sample_id: str, data: Dict[str, Any], images: List[str]) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []

    time_present = bool(data.get("forecast_window") or data.get("historical_window"))
    _add_catalog_item(
        catalog,
        eid="E_TimeContext",
        source_type="time_context",
        modality="structured_text",
        handle=f"forecast_window={data.get('forecast_window','')}; historical_window={data.get('historical_window','')}",
        link=_make_link(sample_id, "data/time_context"),
        present=time_present,
        role="primary" if time_present else "unavailable",
        should_use_as_support=time_present,
        should_mention_in_limitations=not time_present,
        legacy_converted=False,
    )

    hist = data.get("historical_series")
    hist_present = isinstance(hist, list) and len(hist) > 0
    _add_catalog_item(
        catalog,
        eid="E_HistoricalSeries",
        source_type="historical_series",
        modality="time_series",
        handle=f"{len(hist) if isinstance(hist, list) else 0}-step historical {data.get('target_name','series')} available",
        link=_make_link(sample_id, "data/historical_series"),
        present=hist_present,
        role="primary" if hist_present else "unavailable",
        should_use_as_support=hist_present,
        should_mention_in_limitations=not hist_present,
        legacy_converted=False,
    )

    meta_type, meta_handle = _site_meta_summary(data)
    _add_catalog_item(
        catalog,
        eid="E_SiteMeta",
        source_type=meta_type,
        modality="structured_text",
        handle=meta_handle,
        link=_make_link(sample_id, f"data/{meta_type}"),
        present=True,
        role="context",
        should_use_as_support=True,
        should_mention_in_limitations=False,
        legacy_converted=False,
    )

    weather = str(data.get("weather") or "").strip()
    _add_catalog_item(
        catalog,
        eid="E_Weather",
        source_type="weather",
        modality="structured_text",
        handle=_compact_text(weather, 260) if weather else "No weather evidence provided",
        link=_make_link(sample_id, "data/weather"),
        present=bool(weather),
        role="context" if weather else "unavailable",
        should_use_as_support=bool(weather),
        should_mention_in_limitations=not bool(weather),
        legacy_converted=False,
    )

    dens = _density_summary(data)
    _add_catalog_item(
        catalog,
        eid="E_Demographics",
        source_type="density_or_demographic_context",
        modality="structured_text",
        handle=dens or "No density/demographic evidence provided",
        link=_make_link(sample_id, "data/density_or_demographic_context"),
        present=bool(dens),
        role="context" if dens else "unavailable",
        should_use_as_support=bool(dens),
        should_mention_in_limitations=not bool(dens),
        legacy_converted=False,
    )

    poi = _as_list_text(data.get("poi"))
    _add_catalog_item(
        catalog,
        eid="E_POI",
        source_type="poi",
        modality="text",
        handle="; ".join([_compact_text(x, 120) for x in poi[:3]]) if poi else "No POI evidence provided",
        link=_make_link(sample_id, "data/poi"),
        present=bool(poi),
        role="context" if poi else "unavailable",
        should_use_as_support=bool(poi),
        should_mention_in_limitations=not bool(poi),
        legacy_converted=False,
    )

    # 第一版不做地理/时间过滤：news 非空即 present。
    news = _as_list_text(data.get("news_or_event"))
    if news:
        for i, item in enumerate(news[:20], 1):
            _add_catalog_item(
                catalog,
                eid=f"E_News_{i:03d}",
                source_type="news_or_event",
                modality="text",
                handle=_compact_text(item, 260),
                link=_make_link(sample_id, f"data/news_or_event/{i-1}"),
                present=True,
                role="context",
                should_use_as_support=True,
                should_mention_in_limitations=False,
                legacy_converted=False,
            )
    else:
        _add_catalog_item(
            catalog,
            eid="E_News",
            source_type="news_or_event",
            modality="text",
            handle="No news/event evidence provided",
            link=_make_link(sample_id, "data/news_or_event"),
            present=False,
            role="unavailable",
            should_use_as_support=False,
            should_mention_in_limitations=True,
            legacy_converted=False,
        )

    accidents = _as_list_text(data.get("accident"))
    if accidents:
        for i, item in enumerate(accidents[:20], 1):
            _add_catalog_item(
                catalog,
                eid=f"E_Accident_{i:03d}",
                source_type="accident",
                modality="text",
                handle=_compact_text(item, 260),
                link=_make_link(sample_id, f"data/accident/{i-1}"),
                present=True,
                role="context",
                should_use_as_support=True,
                should_mention_in_limitations=False,
                legacy_converted=False,
            )
    else:
        _add_catalog_item(
            catalog,
            eid="E_Accident",
            source_type="accident",
            modality="text",
            handle="No accident evidence provided",
            link=_make_link(sample_id, "data/accident"),
            present=False,
            role="unavailable",
            should_use_as_support=False,
            should_mention_in_limitations=True,
            legacy_converted=False,
        )

    neighbors = _as_list_text(data.get("neighbor_sensors_or_neighbor_regions"))
    if neighbors:
        for i, item in enumerate(neighbors[:20], 1):
            _add_catalog_item(
                catalog,
                eid=f"E_Neighbor_{i:03d}",
                source_type="neighbor_sensors_or_neighbor_regions",
                modality="time_series_text",
                handle=_compact_text(item, 260),
                link=_make_link(sample_id, f"data/neighbor_sensors_or_neighbor_regions/{i-1}"),
                present=True,
                role="context",
                should_use_as_support=True,
                should_mention_in_limitations=False,
                legacy_converted=False,
            )
    else:
        _add_catalog_item(
            catalog,
            eid="E_Neighbor",
            source_type="neighbor_sensors_or_neighbor_regions",
            modality="time_series_text",
            handle="No neighbor sensor/region evidence provided",
            link=_make_link(sample_id, "data/neighbor_sensors_or_neighbor_regions"),
            present=False,
            role="unavailable",
            should_use_as_support=False,
            should_mention_in_limitations=True,
            legacy_converted=False,
        )

    if images:
        for i, _ in enumerate(images[:20], 1):
            _add_catalog_item(
                catalog,
                eid=f"E_Image_{i:03d}",
                source_type="image",
                modality="image",
                handle=f"Image reference {i} available",
                link=_make_link(sample_id, f"images/{i-1}"),
                present=True,
                role="context",
                should_use_as_support=True,
                should_mention_in_limitations=False,
                legacy_converted=False,
            )
    else:
        _add_catalog_item(
            catalog,
            eid="E_Image",
            source_type="image",
            modality="image",
            handle="No image evidence provided",
            link=_make_link(sample_id, "images"),
            present=False,
            role="unavailable",
            should_use_as_support=False,
            should_mention_in_limitations=True,
            legacy_converted=False,
        )

    return catalog


def _build_catalog_from_legacy(sample_id: str, prompt: str, chunks: Dict[str, List[str]], images: List[str]) -> List[Dict[str, Any]]:
    data = {
        "forecast_window": "",
        "historical_window": "",
        "historical_series": [],
        "weather": "",
        "poi": chunks.get("POI", []),
        "news_or_event": chunks.get("News", []),
        "accident": chunks.get("Accident", []),
        "neighbor_sensors_or_neighbor_regions": chunks.get("HopSensor", []),
    }

    if re.search(r"traffic\s+flow\s*:", prompt, flags=re.I) or re.search(r"History\s+average", prompt, flags=re.I):
        data["historical_series"] = [1]

    if re.search(r"Weather", prompt, flags=re.I):
        m = re.search(r"Weather.*?(?=traffic flow:|History average|Please answer|$)", prompt, flags=re.I | re.S)
        data["weather"] = m.group(0) if m else "Weather evidence available"

    if re.search(r"Population density|Housing density", prompt, flags=re.I):
        data["population_density"] = {"raw": "population density available from prompt"}
        data["housing_density"] = {"raw": "housing density available from prompt"}

    m = re.search(r"Sensor\s+ID\s*:\s*([^\s]+)", prompt, flags=re.I)
    if m:
        data["sensor_id"] = m.group(1)

    catalog = _build_catalog_from_new_data(sample_id, data, images)
    for item in catalog:
        item["legacy_converted"] = True
        item["link"] = item["link"].replace("/data/", "/legacy/")
    return catalog


def _catalog_present_ids(catalog: List[Dict[str, Any]]) -> List[str]:
    return [
        str(x.get("id"))
        for x in catalog
        if x.get("present") and x.get("should_use_as_support") and x.get("id")
    ]


def _catalog_present_source_types(catalog: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for x in catalog:
        if x.get("present") and x.get("source_type") and x.get("source_type") not in out:
            out.append(str(x.get("source_type")))
    return out


def _catalog_missing_source_types(catalog: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for x in catalog:
        if not x.get("present") and x.get("source_type") and x.get("source_type") not in out:
            out.append(str(x.get("source_type")))
    return out


def _select_answer_evidence_ids(catalog: List[Dict[str, Any]], max_per_source: int = 3) -> List[str]:
    selected: List[str] = []
    per_src: Dict[str, int] = {}

    for item in catalog:
        if not item.get("present") or not item.get("should_use_as_support"):
            continue
        if item.get("role") == "primary":
            eid = str(item.get("id"))
            selected.append(eid)
            src = str(item.get("source_type"))
            per_src[src] = per_src.get(src, 0) + 1

    for item in catalog:
        if not item.get("present") or not item.get("should_use_as_support"):
            continue
        eid = str(item.get("id"))
        if eid in selected:
            continue
        src = str(item.get("source_type"))
        if per_src.get(src, 0) >= max_per_source:
            continue
        selected.append(eid)
        per_src[src] = per_src.get(src, 0) + 1

    return selected


def _catalog_item_by_id(catalog: List[Dict[str, Any]], eid: str) -> Optional[Dict[str, Any]]:
    for item in catalog:
        if str(item.get("id")) == str(eid):
            return item
    return None


def _evidence_links_from_ids(catalog: List[Dict[str, Any]], ids: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for eid in ids:
        item = _catalog_item_by_id(catalog, eid)
        if not item:
            continue
        out.append({
            "id": str(item.get("id", "")),
            "source_type": str(item.get("source_type", "")),
            "label": _compact_text(item.get("handle", ""), 120),
            "link": str(item.get("link", "")),
        })
    return out


def _infer_decision(question: str, answer: str) -> str:
    q = (question or "").lower()
    a = (answer or "").lower()

    if re.search(r"\bno\b|not likely|unlikely|unnecessary|not necessary|no major|no significant|does not appear", a):
        if "stress" in q or "shortage" in q:
            return "no_high_stress"
        if "crowd" in q or "congestion" in q or "disruption" in q:
            return "low_or_no_disruption"
        return "negative_or_low_risk"

    if re.search(r"\byes\b|likely|should expect|significant|high|crowded|heavy|stressed|surge", a):
        if "crowd" in q:
            return "crowding_expected"
        if "stress" in q or "shortage" in q:
            return "grid_stress_possible"
        if "congestion" in q or "disruption" in q:
            return "disruption_or_congestion_possible"
        return "positive_or_elevated_risk"

    if "which" in q or "most relevant" in q or "relevant" in q:
        return "identify_relevant_signals"

    return "general_reasoned_answer"


def _build_limitations(catalog: List[Dict[str, Any]]) -> List[str]:
    missing = _catalog_missing_source_types(catalog)
    out: List[str] = []
    for src in missing:
        if src == "accident":
            out.append("No accident evidence was provided, so accident-related explanations should not be asserted as supporting evidence.")
        elif src == "news_or_event":
            out.append("No news/event evidence was provided, so news-driven explanations should not be asserted as supporting evidence.")
        elif src == "weather":
            out.append("No weather evidence was provided, so weather-driven explanations should not be asserted as supporting evidence.")
        elif src == "neighbor_sensors_or_neighbor_regions":
            out.append("No neighbor sensor or regional evidence was provided.")
        elif src == "image":
            out.append("No image evidence was provided.")
        elif src == "poi":
            out.append("No POI evidence was provided.")
    return out[:8]


def _build_supporting_facts(catalog: List[Dict[str, Any]], selected_ids: List[str]) -> List[Dict[str, Any]]:
    by_src: Dict[str, List[str]] = {}
    for eid in selected_ids:
        item = _catalog_item_by_id(catalog, eid)
        if not item:
            continue
        src = str(item.get("source_type", ""))
        by_src.setdefault(src, []).append(eid)

    facts: List[Dict[str, Any]] = []

    def add(srcs: List[str], claim: str):
        ids: List[str] = []
        for src in srcs:
            ids.extend(by_src.get(src, []))
        ids = list(dict.fromkeys(ids))
        if ids:
            facts.append({"claim": claim, "evidence_ids": ids})

    add(["historical_series", "time_context"], "The historical series and time context provide the primary temporal basis for the answer.")
    add(["weather"], "Weather evidence provides environmental context for the queried urban condition.")
    add(["poi"], "POI evidence provides local land-use and activity context.")
    add(["news_or_event"], "News or event evidence provides external urban-event context available to the model.")
    add(["accident"], "Accident evidence provides incident context available to the model.")
    add(["neighbor_sensors_or_neighbor_regions"], "Neighbor sensor or regional evidence provides spatial context around the target node.")
    add(["density_or_demographic_context", "sensor_meta", "utility_meta", "site_meta"], "Site, density, or demographic context helps ground the answer to the queried urban system.")
    add(["image"], "Image evidence provides visual or map context when available.")

    if not facts and selected_ids:
        facts.append({
            "claim": "The answer is grounded in the available evidence handles listed for this sample.",
            "evidence_ids": selected_ids[:4],
        })

    return facts[:8]


def _ensure_structured_answer_json(
    answer: str,
    *,
    question: str,
    catalog: List[Dict[str, Any]],
) -> str:
    """
    Stage IV canonical answer schema:
      decision / answer / evidence_links / supporting_facts / final_summary / limitations

    不再生成或保留 confidence。
    """
    obj = _extract_first_json_object(answer)
    selected_ids = _select_answer_evidence_ids(catalog, max_per_source=3)

    if isinstance(obj, dict):
        obj = dict(obj)
        obj.pop("confidence", None)

        # 兼容旧字段名。
        if "answer" not in obj:
            for k in ("final_answer", "response", "direct_answer"):
                if isinstance(obj.get(k), str) and obj.get(k).strip():
                    obj["answer"] = obj.get(k).strip()
                    break
        if "answer" not in obj:
            obj["answer"] = _compact_text(answer, 1200) or "The answer should be based on the available smart-city evidence."

        if "decision" not in obj or not str(obj.get("decision", "")).strip():
            obj["decision"] = _infer_decision(question, str(obj.get("answer", "")))

        if not isinstance(obj.get("evidence_links"), list) or not obj.get("evidence_links"):
            obj["evidence_links"] = _evidence_links_from_ids(catalog, selected_ids)

        if not isinstance(obj.get("supporting_facts"), list) or not obj.get("supporting_facts"):
            obj["supporting_facts"] = _build_supporting_facts(catalog, selected_ids)

        if "final_summary" not in obj or not str(obj.get("final_summary", "")).strip():
            obj["final_summary"] = str(obj.get("answer", ""))

        if not isinstance(obj.get("limitations"), list):
            obj["limitations"] = _build_limitations(catalog)

        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    answer_text = _compact_text(answer, 1200) or "The answer should be based on the available smart-city evidence."
    obj = {
        "decision": _infer_decision(question, answer_text),
        "answer": answer_text,
        "evidence_links": _evidence_links_from_ids(catalog, selected_ids),
        "supporting_facts": _build_supporting_facts(catalog, selected_ids),
        "final_summary": answer_text,
        "limitations": _build_limitations(catalog),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _format_evidence_handles(catalog: List[Dict[str, Any]]) -> str:
    if not catalog:
        return "Evidence handles:\n- none"
    lines = ["Evidence handles:"]
    for item in catalog:
        status = "present" if item.get("present") else "missing"
        support = "support" if item.get("should_use_as_support") else "do_not_use_as_support"
        lines.append(
            f"- {item.get('id','')} | {item.get('source_type','')} | {status} | {support} | {_compact_text(item.get('handle',''), 180)}"
        )
    return "\n".join(lines)



def _reason_output_format_from_example(ex: Dict[str, Any], reward_meta: Dict[str, Any]) -> str:
    vals = [
        _get_ci(ex, "output_format", default=""),
        _get_ci(ex, "stageD_output_format", default=""),
        reward_meta.get("output_format", ""),
        reward_meta.get("stageD_output_format", ""),
        reward_meta.get("policy_output_format", ""),
    ]
    for v in vals:
        s = str(v or "").strip()
        if s:
            return s
    return ""


def _stageD_parse_json_maybe(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _stageD_norm_label(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _stageD_decision_header(decision: Any) -> str:
    d = _stageD_norm_label(decision)
    positive = {
        "positive",
        "positive_or_high_risk",
        "positive_or_elevated_risk",
        "high_risk",
        "elevated_risk",
        "significant_change",
        "high_stress",
        "elevated_stress",
    }
    negative = {
        "negative",
        "negative_or_low_risk",
        "low_risk",
        "no_high_risk",
        "no_high_stress",
        "no_stress",
        "low_stress",
        "no_significant_change",
        "normal_or_low_risk",
        "stable_or_low_risk",
    }
    if d in positive or any(k in d for k in ("positive", "elevated", "high_risk", "high_stress", "significant_change")):
        return "Significant change / high risk"
    if d in negative or any(k in d for k in ("negative", "low_risk", "no_high", "no_stress", "normal", "stable")):
        return "No significant change / low risk"
    return "Reasoned answer"


def _stageD_catalog_support_and_missing(catalog: Any) -> Tuple[set, List[str]]:
    support_ids = set()
    missing_msgs: List[str] = []
    if not isinstance(catalog, list):
        return support_ids, missing_msgs

    for item in catalog:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id", "") or "").strip()
        source_type = str(item.get("source_type", "") or item.get("type", "") or "").strip()
        status = str(item.get("status", "") or "").strip().lower()
        role = str(item.get("role", "") or "").strip().lower()
        present = bool(item.get("present", False)) or status == "present"
        support = bool(item.get("should_use_as_support", False)) or role == "support"

        if eid and present and support:
            support_ids.add(eid)

        if (not present) or status == "missing" or "do_not_use" in role:
            if source_type:
                pretty = source_type.replace("_", " ")
                missing_msgs.append(f"No {pretty} evidence was provided.")
            else:
                missing_msgs.append("Some evidence was not provided.")

    # de-duplicate while preserving order
    out = []
    seen = set()
    for msg in missing_msgs:
        key = msg.lower()
        if key not in seen:
            seen.add(key)
            out.append(msg)
    return support_ids, out



def _stageD_strip_answer_header_text(s: Any) -> str:
    """Remove decision-style answer prefixes from supervised public answers."""
    x = re.sub(r"\s+", " ", str(s or "")).strip()
    x = re.sub(r"^\s*Significant change\s*/\s*high risk\s*:\s*", "", x, flags=re.I)
    x = re.sub(r"^\s*No significant change\s*/\s*low risk\s*:\s*", "", x, flags=re.I)
    x = re.sub(r"^\s*Reasoned answer\s*:\s*", "", x, flags=re.I)
    x = re.sub(r"^\s*Decision\s*:\s*", "", x, flags=re.I)
    x = re.sub(r"^\s*Answer\s*:\s*", "", x, flags=re.I)
    return re.sub(r"\s+", " ", x).strip()

def _stageD_freeform_answer_from_json_gold(
    answer_json: Any,
    catalog: Any = None,
    *,
    preferred_decision: Any = None,
    preferred_answer: Any = None,
) -> str:
    # Convert hidden JSON gold into a supervised freeform final answer.
    # No-header answer-fit route:
    #   final answer = original_answer / answer_core + citations + optional limitations.
    # This avoids contradictory headers such as:
    #   "Significant change / high risk: ... not under stress ..."
    obj = _stageD_parse_json_maybe(answer_json)
    if not isinstance(obj, dict):
        return ""

    body = (
        preferred_answer
        or obj.get("answer")
        or obj.get("final_summary")
        or obj.get("summary")
        or obj.get("response")
        or ""
    )
    body = re.sub(r"\s+", " ", str(body or "")).strip()
    body = _stageD_strip_answer_header_text(body)
    if not body:
        body = "The answer is based on the available present/support evidence."

    support_ids, missing_msgs = _stageD_catalog_support_and_missing(catalog)

    ids: List[str] = []
    links = obj.get("evidence_links", []) or []
    if isinstance(links, list):
        for item in links:
            if isinstance(item, dict):
                eid = str(item.get("id", "") or "").strip()
            else:
                eid = str(item or "").strip()

            if not eid or not eid.startswith("E_"):
                continue
            if support_ids and eid not in support_ids:
                continue
            if eid not in ids:
                ids.append(eid)

    if not ids and support_ids:
        ids = list(support_ids)

    try:
        max_cites = int(os.environ.get("SMARTCITY_REASON_SFT_MAX_CITATIONS", "4") or "4")
    except Exception:
        max_cites = 4
    max_cites = max(1, max_cites)
    ids = ids[:max_cites]

    citations = " ".join(f"[{eid}]" for eid in ids)
    final = body
    if citations:
        final = f"{final} {citations}"

    include_lims = os.environ.get("SMARTCITY_REASON_SFT_INCLUDE_LIMITATIONS", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }

    if include_lims:
        limitations = obj.get("limitations", []) or []
        lim_texts: List[str] = []

        if isinstance(limitations, list):
            for x in limitations:
                if isinstance(x, dict):
                    txt = str(
                        x.get("text", "")
                        or x.get("label", "")
                        or x.get("reason", "")
                        or ""
                    ).strip()
                else:
                    txt = str(x or "").strip()
                if txt:
                    lim_texts.append(txt)
        elif isinstance(limitations, str) and limitations.strip():
            lim_texts.append(limitations.strip())

        lim_texts.extend(missing_msgs)

        dedup_lims: List[str] = []
        seen = set()

        for txt in lim_texts:
            txt = re.sub(r"\s+", " ", txt).strip()
            if not txt:
                continue

            key = txt.lower()

            # Collapse common short/long variants of the same missing evidence.
            if "no accident" in key and "evidence" in key:
                key = "missing_accident"
            elif "no image" in key and "evidence" in key:
                key = "missing_image"
            elif "no weather" in key and "evidence" in key:
                key = "missing_weather"
            elif (("no neighbor" in key) or ("no nearby" in key) or ("no regional" in key)) and "evidence" in key:
                key = "missing_neighbor"
            elif (("no news" in key) or ("no event" in key)) and "evidence" in key:
                key = "missing_news_event"

            if key not in seen:
                seen.add(key)
                dedup_lims.append(txt)

        if dedup_lims:
            final = final.rstrip() + "\n\nLimitations: " + " ".join(dedup_lims[:4])

    final = re.sub(r"<think>.*?</think>", "", final, flags=re.I | re.S)
    final = re.sub(r"</?think>", "", final, flags=re.I).strip()
    final = _stageD_strip_answer_header_text(final)
    return final

def _is_stageD_sft_freeform_answer(ex: Dict[str, Any], reward_meta: Dict[str, Any]) -> bool:
    if SMARTCITY_REASON_KEEP_FREEFORM_ANSWER:
        return True
    fmt = _reason_output_format_from_example(ex, reward_meta).lower()
    return ("sft_warmup" in fmt) or ("freeform_sft" in fmt)


def _is_stageD_freeform_prompt(ex: Dict[str, Any], reward_meta: Dict[str, Any]) -> bool:
    if SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE:
        return True
    fmt = _reason_output_format_from_example(ex, reward_meta).lower()
    if "freeform" in fmt:
        return True
    v = reward_meta.get("model_should_output_json", None)
    if isinstance(v, bool) and (not v):
        return True
    return False


def _rewrite_reason_prompt_for_freeform_evidence(prompt: str, reasoning_mode: str) -> str:
    """
    Remove old JSON-output instructions and replace them with StageD freeform
    evidence-citation instructions. Keep Evidence handles and multimodal markers.
    """
    s = str(prompt or "")

    # Preserve whether markers existed, then remove all marker text and append a clean marker block later.
    had_chunk = CHUNK_MARKER in s
    had_image = IMAGE_MARKER in s
    s = s.replace(CHUNK_MARKER, "").replace(IMAGE_MARKER, "")

    # If there is an Output requirements block, remove from the first such block onward.
    # This handles prompts that contain both old JSON requirements and new freeform requirements.
    m = re.search(r"(?is)\n?\n?Output requirements\s*:", s)
    if m:
        s = s[:m.start()].rstrip()

    # Remove older JSON instruction forms that may not have the "Output requirements:" header.
    old_patterns = [
        r"(?is)\n*\s*Output exactly one single-line JSON object.*?(?=\n\s*(?:Only cite|This sample|$))",
        r"(?is)\n*\s*Only cite present evidence IDs from Evidence handles\..*?(?=\n\s*(?:This sample|$))",
        r"(?im)^\s*This sample is think-mode:.*public JSON\.\s*$",
        r"(?im)^\s*This sample is no-think[- ]mode:.*JSON.*$",
        r"(?im)^\s*-\s*Output exactly one single-line JSON object.*$",
        r"(?im)^\s*-\s*The JSON must contain:.*$",
        r"(?im)^\s*-\s*evidence_links\[\]\.id.*$",
        r"(?im)^\s*-\s*Do not use missing evidence as supporting facts.*$",
        r"(?im)^\s*-\s*This sample is think-mode:.*public JSON.*$",
        r"(?im)^\s*-\s*This sample is no-think[- ]mode:.*public JSON.*$",
    ]
    for pat in old_patterns:
        s = re.sub(pat, "", s).rstrip()

    mode = str(reasoning_mode or "").strip().lower().replace("-", "_")
    if SMARTCITY_REASON_ALWAYS_THINK_TAGS:
        if mode == "think":
            mode_line = "- This sample is think-mode: start with a <think>...</think> block containing brief evidence planning, then write the public final answer in natural language."
        else:
            mode_line = "- This sample is no-think mode: still start with an empty <think></think> block, then write only the public final answer in natural language."
    else:
        if mode == "think":
            mode_line = "- This sample is think-mode: start with a <think>...</think> block containing brief evidence planning, then write the public final answer in natural language."
        else:
            mode_line = "- This sample is no-think mode: start with an empty <think></think> block, then write only the public final answer in natural language. Do not put reasoning inside the think block."

    req = f"""Output requirements:
- Output format id: {SMARTCITY_STAGE_D_FREEFORM_OUTPUT_FORMAT}
{mode_line}
- Do not output JSON. Do not output markdown. Do not output a confidence field.
- In the final answer, cite evidence handles exactly in square brackets when you rely on them, for example [E_HistoricalSeries] or [E_Weather].
- Only cite present/support evidence IDs listed in Evidence handles.
- Never cite missing or do_not_use_as_support evidence as support.
- Mention missing evidence only as a limitation, for example: "No accident evidence was provided."
- State the decision clearly in words, such as "no significant change / low risk" or "significant change / high risk", and explain why."""

    s = s.rstrip() + "\n\n" + req.strip()

    # Keep marker block stable for the multimodal collator.
    if had_chunk or had_image or SMARTCITY_REASON_ALWAYS_MARKERS:
        s = s.rstrip() + "\n" + CHUNK_MARKER + "\n" + IMAGE_MARKER

    return s.strip()


def _strip_old_answer_instruction(prompt: str) -> str:
    s = str(prompt or "").strip()
    patterns = [
        r"\s*Please\s+provide\s+the\s+result\s+in\s+JSON\s+format.*$",
        r"\s*Please\s+first\s+write\s+your\s+detailed\s+reasoning\s+process.*$",
        r"\s*Please\s+first\s+write.*?<answer>.*$",
    ]
    for pat in patterns:
        s2 = re.sub(pat, "", s, flags=re.I | re.S).strip()
        if len(s2) < len(s):
            s = s2
    return s


def _build_prompt_from_new_data(question: str, data: Dict[str, Any], catalog: List[Dict[str, Any]], reasoning_mode: str) -> str:
    lines = [
        "You are TransCity, an evidence-grounded smart-city reasoning model.",
        "",
        "Question:",
        str(question or "").strip(),
        "",
        "Task context:",
        f"- task_family: {data.get('task_family', '')}",
        f"- target_name: {data.get('target_name', '')}",
        f"- forecast_window: {data.get('forecast_window', '')}",
        f"- historical_window: {data.get('historical_window', '')}",
    ]

    if data.get("historical_series") is not None:
        lines.append(f"- historical_series: {json.dumps(data.get('historical_series'), ensure_ascii=False)}")

    meta_type, meta_handle = _site_meta_summary(data)
    lines.append(f"- {meta_type}: {meta_handle}")

    dens = _density_summary(data)
    if dens:
        lines.append(f"- density_or_demographic_context: {dens}")

    weather = str(data.get("weather") or "").strip()
    if weather:
        lines.append(f"- weather_summary: {_compact_text(weather, 800)}")

    lines.extend([
        "",
        "Long multimodal evidence is provided through the following special tokens:",
        CHUNK_MARKER,
        IMAGE_MARKER,
        "",
        _format_evidence_handles(catalog),
        "",
        "Output requirements:",
        "- Output exactly one single-line JSON object after any required thinking section.",
        "- The JSON must contain: decision, answer, evidence_links, supporting_facts, final_summary, limitations.",
        "- evidence_links[].id and supporting_facts[].evidence_ids must only cite present evidence IDs from Evidence handles.",
        "- Do not use missing evidence as supporting facts. Missing evidence can only be mentioned in limitations.",
    ])

    if reasoning_mode == "think":
        lines.append("- This sample is think-mode: brief internal planning may appear inside <think>...</think>, then output the public JSON.")
    else:
        lines.append("- This sample is no-think mode: do not output <think> tags; output only the public JSON.")

    return "\n".join(lines).strip()


def _append_evidence_handles_to_prompt(prompt: str, catalog: List[Dict[str, Any]], reasoning_mode: str) -> str:
    p = _strip_old_answer_instruction(prompt)
    parts = [
        p,
        _format_evidence_handles(catalog),
        "Output exactly one single-line JSON object with: decision, answer, evidence_links, supporting_facts, final_summary, limitations.",
        "Only cite present evidence IDs from Evidence handles. Do not use missing evidence as supporting facts.",
    ]
    if reasoning_mode == "think":
        parts.append("This sample is think-mode: use brief <think> planning, then public JSON.")
    else:
        parts.append("This sample is no-think mode: do not output <think> tags.")
    return "\n\n".join([x.strip() for x in parts if x and str(x).strip()])


def _infer_reasoning_mode(ex: Dict[str, Any], thinking: List[str]) -> str:
    mode = str(_get_ci(ex, "reasoning_mode", default="") or "").strip().lower().replace("-", "_")
    if mode in THINK_MODES:
        return "think"
    if mode in NO_THINK_MODES:
        return "no_think"

    # 有 cot/cot_steps 或 thinking 的样本走 think；否则 no-think。
    if str(_get_ci(ex, "cot", default="") or "").strip():
        return "think"
    steps = _get_ci(ex, "cot_steps", default=None)
    if isinstance(steps, list) and len(steps) > 0:
        return "think"
    if thinking:
        return "think"

    return "no_think"


def _thinking_from_cot(ex: Dict[str, Any]) -> List[str]:
    steps = _get_ci(ex, "cot_steps", default=None)
    if isinstance(steps, list) and steps:
        out: List[str] = []
        for st in steps:
            if isinstance(st, dict):
                mod = _compact_text(st.get("modality", ""), 80)
                ev = _compact_text(st.get("evidence", ""), 400)
                inf = _compact_text(st.get("inference", ""), 400)
                txt = f"{mod}: {ev} -> {inf}".strip()
                if txt:
                    out.append(txt)
            else:
                txt = _compact_text(st, 500)
                if txt:
                    out.append(txt)
        if out:
            return out[:8]

    cot = str(_get_ci(ex, "cot", default="") or "").strip()
    if cot:
        parts = re.split(r"(?<=[.!?。！？])\s+", cot)
        parts = [_compact_text(x, 500) for x in parts if x.strip()]
        return parts[:8] if parts else [_compact_text(cot, 1000)]

    return []


# =========================
# prompt builder (Qwen3 style)
# =========================
def _validate_prompt_style(prompt_style: str) -> str:
    s = (prompt_style or PROMPT_STYLE_QWEN3).strip().lower()
    if s not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {sorted(PROMPT_STYLES)}; got {prompt_style!r}")
    return s


def _format_message(role: str, text: str, *, prompt_style: str) -> str:
    prompt_style = _validate_prompt_style(prompt_style)
    role = (role or "").strip().lower()
    text = (text or "").strip()

    if prompt_style == PROMPT_STYLE_PLAIN:
        if role == "system":
            return f"System: {text}".strip()
        if role == "user":
            return f"User: {text}".strip()
        if role == "assistant":
            return f"Assistant: {text}".strip()
        return text.strip()

    if role not in ("system", "user", "assistant"):
        return ""
    return f"<|im_start|>{role}\n{text}<|im_end|>"


def _build_prompt_from_messages(
    msgs: list,
    *,
    prompt_style: str,
    drop_last_assistant: bool = True,
    add_generation_prompt: bool = True,
) -> Tuple[str, str, List[str]]:
    """
    多轮 messages -> prompt（按顺序拼接）
    - drop_last_assistant=True：把最后一个 assistant 作为 label，不放进 prompt history
    - add_generation_prompt=True：末尾追加一个“打开的 assistant”段
    返回：(prompt, last_assistant_text, images_from_user_turns)
    """
    prompt_style = _validate_prompt_style(prompt_style)

    def _collect_images(content) -> List[str]:
        paths: List[str] = []
        if isinstance(content, list):
            for seg in content:
                if not isinstance(seg, dict):
                    continue
                t = seg.get("type", None)
                if t == "image":
                    uri = seg.get("image", None)
                elif t == "image_url":
                    iu = seg.get("image_url", None)
                    uri = iu.get("url", None) if isinstance(iu, dict) else None
                else:
                    uri = None
                if not (isinstance(uri, str) and uri.strip()):
                    uri = seg.get("url", None) or seg.get("path", None)
                if isinstance(uri, str) and uri.strip():
                    paths.append(uri.strip())
        return paths

    parsed: List[Dict[str, Any]] = []
    images: List[str] = []

    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "") or "").lower()
            content = m.get("content", "")
            text = _content_to_text(content)
            if role == "user":
                images.extend(_collect_images(content))
            if role in ("system", "user", "assistant"):
                parsed.append({"role": role, "text": text})

    last_asst_text = ""
    last_asst_idx = None
    for i in range(len(parsed) - 1, -1, -1):
        if parsed[i]["role"] == "assistant" and (parsed[i]["text"] or "").strip():
            last_asst_text = parsed[i]["text"].strip()
            last_asst_idx = i
            break

    if drop_last_assistant and last_asst_idx is not None:
        history = parsed[:last_asst_idx]
    else:
        history = parsed

    parts: List[str] = []
    if prompt_style == PROMPT_STYLE_PLAIN:
        for msg in history:
            line = _format_message(msg["role"], msg["text"], prompt_style=prompt_style)
            if line:
                parts.append(line)
        if add_generation_prompt:
            parts.append("Assistant:")
        prompt = "\n".join([p for p in parts if p]).strip()
    else:
        for msg in history:
            seg = _format_message(msg["role"], msg["text"], prompt_style=prompt_style)
            if seg:
                parts.append(seg)
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

    seen = set()
    images_dedup: List[str] = []
    for p in images:
        if p not in seen:
            seen.add(p)
            images_dedup.append(p)

    return prompt, last_asst_text, images_dedup


def _inject_markers_before_open_assistant(prompt: str, markers: List[str]) -> str:
    """
    把 marker 块插入到最后一个“打开的 assistant”之前（如果存在）。
    目标：让 collator 的“单次插入块”稳定落在末尾。
    """
    prompt = prompt or ""
    markers = [m for m in markers if m]
    if not markers:
        return prompt

    injection = "\n".join(markers).strip() + "\n"
    anchor = "<|im_start|>assistant\n"
    idx = prompt.rfind(anchor)
    if idx >= 0:
        return prompt[:idx] + injection + prompt[idx:]
    return (prompt + "\n" + injection).strip()


def _ensure_reason_markers(prompt: str) -> str:
    need: List[str] = []
    if CHUNK_MARKER not in prompt:
        need.append(CHUNK_MARKER)
    if IMAGE_MARKER not in prompt:
        need.append(IMAGE_MARKER)
    return _inject_markers_before_open_assistant(prompt, need) if need else prompt


# =========================
# I/O
# =========================
def _is_remote_uri(p: str) -> bool:
    p = (p or "").strip().lower()
    return p.startswith(("http://", "https://", "s3://", "gs://", "file://"))


def _resolve_paths_under(base_dir: str, paths: list[str]) -> list[str]:
    base_dir = os.path.abspath(base_dir)
    out = []
    for p in paths or []:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if not p:
            continue
        if os.path.isabs(p) or _is_remote_uri(p):
            out.append(p)
        else:
            out.append(os.path.normpath(os.path.join(base_dir, p)))
    return out


def load_jsonl_dataset(
    path: str | Path,
    split_name: str = "train",
    *,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
) -> HFDataset:
    return _load_via_generator_flat(str(path), split_name, prompt_style=prompt_style, ensure_markers=ensure_markers)


def save_dataset_jsonl(ds: HFDataset, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_via_generator_flat(path: str, split_name: str, *, prompt_style: str, ensure_markers: bool) -> HFDataset:
    prompt_style = _validate_prompt_style(prompt_style)
    base_dir = os.path.abspath(os.path.dirname(path))

    def gen():
        bad = 0
        use_orjson = False
        try:
            import orjson  # type: ignore
            use_orjson = True
        except Exception:
            pass

        with open(path, "r", encoding="utf-8", buffering=1 << 20) as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = (orjson.loads(s) if use_orjson else json.loads(s))
                except Exception as e:
                    bad += 1
                    if _is_rank0() and SMARTCITY_DEBUG and (bad <= 10 or bad % 100 == 0):
                        print(f"[JSONL] parse error line {ln}: {e} (bad={bad})")
                    if SMARTCITY_MAX_BAD_LINES and bad > SMARTCITY_MAX_BAD_LINES:
                        raise RuntimeError(f"Too many bad lines: {bad} (limit={SMARTCITY_MAX_BAD_LINES}). Last at {ln}")
                    continue

                row = _normalize_row_core(obj, prompt_style=prompt_style, ensure_markers=ensure_markers)
                row["images"] = _resolve_paths_under(base_dir, row.get("images", []) or [])
                yield row

                if _is_rank0() and SMARTCITY_DEBUG and (ln % SMARTCITY_LOG_EVERY == 0):
                    _dprint(f"[JSONL] parsed {ln:,} lines ... (bad={bad})")

    ds = HFDataset.from_generator(gen, features=SMARTCITY_FEATURES)
    return ds


# =========================
# row normalize (core)
# =========================
def _normalize_row_core(
    ex: Dict[str, Any],
    *,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
) -> Dict[str, Any]:
    """
    统一为 SMARTCITY_FEATURES。

    支持三种输入：
    1. 已转换的 evidence-aware 样本；
    2. 新格式 raw 样本：data + question + answer + cot/cot_steps；
    3. 旧格式 raw 样本：messages + chunks + thinking。
    """
    prompt_style = _validate_prompt_style(prompt_style)

    try:
        ex = dict(ex)
    except Exception:
        pass

    out: Dict[str, Any] = {}

    sample_id = str(_get_ci(ex, "sample_id", default="") or "")
    data_obj = _get_ci(ex, "data", default={})
    if not isinstance(data_obj, dict):
        data_obj = {}

    if not sample_id:
        sample_id = str(data_obj.get("sample_id") or "")

    dataset_source = str(_get_ci(ex, "dataset_source", default="") or data_obj.get("source_name", "") or "")
    question_type = str(_get_ci(ex, "question_type", default="") or "")

    out["sample_id"] = sample_id
    out["dataset_source"] = dataset_source
    out["question_type"] = question_type

    # task: 新 raw QA 没有 task 时默认 reason。
    task_in = _get_ci(ex, "task", default=None)
    task = str(task_in).strip().lower() if task_in is not None else ""
    if not task:
        if data_obj or _get_ci(ex, "question", default=None) is not None or _get_ci(ex, "evidence_catalog", default=None) is not None:
            task = "reason"
    out["task"] = task

    # images
    images_top = _get_ci(ex, "images", default=[]) or _get_ci(ex, "image", default=[]) or _get_ci(ex, "image_paths", default=[]) or []
    if not isinstance(images_top, list):
        images_top = [images_top]
    images_top = [str(x).strip() for x in images_top if isinstance(x, str) and str(x).strip()]

    # chunks
    raw_chunks = _get_ci(ex, "chunks", default=None)
    if isinstance(raw_chunks, dict):
        chunks = _normalize_chunks_ci(raw_chunks)
    elif data_obj:
        chunks = {
            "POI": _as_list_text(data_obj.get("poi")),
            "News": _as_list_text(data_obj.get("news_or_event")),
            "Accident": _as_list_text(data_obj.get("accident")),
            "HopSensor": _as_list_text(data_obj.get("neighbor_sensors_or_neighbor_regions")),
            "HopBA": [],
        }
    else:
        chunks = _normalize_chunks_ci({})
    out["chunks"] = chunks

    # thinking
    tk = _get_ci(ex, "thinking", default=[])
    if tk is None:
        tk = []
    if not isinstance(tk, list):
        tk = [str(tk)]
    else:
        tk = [str(t) for t in tk if t is not None and str(t).strip()]

    # top-level prompt/answer/target/images
    prompt = str(_get_ci(ex, "prompt", default="") or "")
    answer = str(_get_ci(ex, "answer", default="") or "")
    target = str(_get_ci(ex, "target", default="") or "")
    question = str(_get_ci(ex, "question", default="") or "")

    msgs = _get_ci(ex, "messages", default=None)
    images = images_top
    last_asst_text = ""

    if isinstance(msgs, list) and msgs:
        prompt_from_msgs, last_asst_text, imgs_from_msgs = _build_prompt_from_messages(
            msgs,
            prompt_style=prompt_style,
            drop_last_assistant=True,
            add_generation_prompt=True,
        )

        if not prompt and prompt_from_msgs:
            prompt = prompt_from_msgs

        seen = set()
        merged: List[str] = []
        for p in (images_top + imgs_from_msgs):
            if p and (p not in seen):
                seen.add(p)
                merged.append(p)
        images = merged

        if task == "understand":
            if (not target) and last_asst_text:
                target = last_asst_text
        else:
            if (not answer) and last_asst_text:
                answer = last_asst_text

    # raw 新格式没有 prompt 时，生成 prompt。
    # 已转换数据一般已经有 prompt，不会走这里。
    if data_obj and question and not prompt:
        # reasoning_mode 先用 cot 判断，后面会再标准化。
        tmp_thinking = _thinking_from_cot(ex) or tk
        tmp_mode = _infer_reasoning_mode(ex, tmp_thinking)
        tmp_catalog = _build_catalog_from_new_data(sample_id or "sample", data_obj, images)
        prompt = _build_prompt_from_new_data(question, data_obj, tmp_catalog, tmp_mode)

    # reward meta / evidence catalog
    raw_reward_meta = _get_ci(ex, "reward_meta", default={})
    reward_meta = _parse_reward_meta(raw_reward_meta)

    evidence_catalog_val = _get_meta_field(ex, reward_meta, "evidence_catalog", "")
    catalog_obj = _parse_json_like(evidence_catalog_val)

    if not isinstance(catalog_obj, list):
        if data_obj:
            catalog_obj = _build_catalog_from_new_data(sample_id or "sample", data_obj, images)
        else:
            user_prompt_for_legacy = question or prompt or _extract_message_text(msgs, "user")
            catalog_obj = _build_catalog_from_legacy(sample_id or "sample", user_prompt_for_legacy, chunks, images)

    evidence_catalog_str = json.dumps(catalog_obj, ensure_ascii=False, separators=(",", ":"))

    # reasoning_mode
    if not tk:
        tk = _thinking_from_cot(ex)

    reasoning_mode = str(_get_meta_field(ex, reward_meta, "reasoning_mode", "") or "").strip().lower().replace("-", "_")
    if reasoning_mode in THINK_MODES:
        reasoning_mode = "think"
    elif reasoning_mode in NO_THINK_MODES:
        reasoning_mode = "no_think"
    else:
        reasoning_mode = _infer_reasoning_mode(ex, tk)

    # no-think 样本清空 thinking，避免 collator 又训练 <think>。
    if reasoning_mode == "no_think":
        tk = []

    # old prompt / raw legacy prompt 的输出要求需要补 evidence handles。
    # 已转换 prompt 通常已经有 Evidence handles，不重复追加。
    if task == "reason" and prompt and "Evidence handles:" not in prompt:
        prompt = _append_evidence_handles_to_prompt(prompt, catalog_obj, reasoning_mode)

    # StageD freeform safety override: remove old JSON-output instructions.
    if task == "reason" and prompt and _is_stageD_freeform_prompt(ex, reward_meta):
        prompt = _rewrite_reason_prompt_for_freeform_evidence(prompt, reasoning_mode)

    if task == "reason" and ensure_markers and SMARTCITY_REASON_ALWAYS_MARKERS:
        prompt = _ensure_reason_markers(prompt)
    elif ensure_markers and prompt:
        has_chunks = any((len(v) > 0) for v in (chunks or {}).values())
        has_images = len(images) > 0
        need: List[str] = []
        if has_chunks and (CHUNK_MARKER not in prompt):
            need.append(CHUNK_MARKER)
        if has_images and (IMAGE_MARKER not in prompt):
            need.append(IMAGE_MARKER)
        if need:
            prompt = _inject_markers_before_open_assistant(prompt, need)

    # reason answer canonicalization:
    # - GRPO hidden gold can stay JSON.
    # - StageD SFT warmup must keep freeform answer labels.
    if task == "reason":
        if _is_stageD_sft_freeform_answer(ex, reward_meta):
            _sft_freeform = ""
            if SMARTCITY_REASON_SFT_JSON_GOLD_TO_FREEFORM:
                _preferred_decision = str(ex.get("gold_decision", "") or reward_meta.get("gold_decision", "") or "").strip()
                _preferred_answer = str(
                    ex.get("answer_core", "")
                    or reward_meta.get("original_answer", "")
                    or ""
                ).strip()
                _sft_freeform = _stageD_freeform_answer_from_json_gold(
                    answer or target,
                    catalog_obj,
                    preferred_decision=_preferred_decision,
                    preferred_answer=_preferred_answer,
                )
            if _sft_freeform:
                answer = _sft_freeform
            else:
                answer = str(answer or target or "")
            # The answer field is only the public final answer. Think tags are added by the collator.
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.I | re.S)
            answer = re.sub(r"</?think>", "", answer, flags=re.I).strip()

        else:
            answer = _ensure_structured_answer_json(
                answer,
                question=question or prompt,
                catalog=catalog_obj,
            )

    out["thinking"] = tk
    out["prompt"] = prompt or ""
    out["answer"] = answer or ""
    out["target"] = target or ""
    out["images"] = images or []

    # metadata
    out["reward_meta"] = _json_string_or_empty(raw_reward_meta)
    out["reasoning_mode"] = reasoning_mode
    out["gold_decision"] = str(_get_meta_field(ex, reward_meta, "gold_decision", "") or "")
    if not out["gold_decision"] and task == "reason":
        ans_obj = _parse_json_like(answer)
        if isinstance(ans_obj, dict):
            out["gold_decision"] = str(ans_obj.get("decision", "") or "")

    out["evidence_catalog"] = evidence_catalog_str

    key_ids = _as_string_list(_get_meta_field(ex, reward_meta, "key_evidence_ids", []))
    if not key_ids:
        key_ids = _select_answer_evidence_ids(catalog_obj, max_per_source=3)
    out["key_evidence_ids"] = key_ids

    key_srcs = _as_string_list(_get_meta_field(ex, reward_meta, "key_source_types", []))
    if not key_srcs:
        key_srcs = _catalog_present_source_types(catalog_obj)
    out["key_source_types"] = key_srcs

    out["as_of_time"] = str(_get_meta_field(ex, reward_meta, "as_of_time", "") or "")
    out["perturbation_type"] = str(_get_meta_field(ex, reward_meta, "perturbation_type", "") or "full")
    out["expected_behavior"] = str(_get_meta_field(ex, reward_meta, "expected_behavior", "") or "")

    return out


# =========================
# ensure on existing dataset
# =========================
def ensure_smartcity_features(
    ds: HFDataset,
    *,
    num_proc: int = 16,
    prompt_style: str = PROMPT_STYLE_QWEN3,
    ensure_markers: bool = True,
    messages_policy: str = "to_prompt",   # 兼容保留
    fold_messages_to: str = "News",        # 兼容保留
) -> HFDataset:
    prompt_style = _validate_prompt_style(prompt_style)
    expected_cols = set(SMARTCITY_FEATURES.keys())
    if set(ds.column_names) == expected_cols:
        return ds.cast(SMARTCITY_FEATURES)

    def _normalize_row(ex: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_row_core(ex, prompt_style=prompt_style, ensure_markers=ensure_markers)

    ds = ds.map(_normalize_row, num_proc=num_proc, desc="smartcity:normalize(messages/data→prompt/evidence)")
    to_drop = [c for c in ds.column_names if c not in expected_cols]
    if to_drop:
        ds = ds.remove_columns(to_drop)
    ds = ds.cast(SMARTCITY_FEATURES)
    return ds


# =========================
# chunk helper: 510 tokens/chunk
# =========================
def _chunk_by_token_ids(
    text: str,
    tok: PreTrainedTokenizerBase,
    *,
    chunk_size_tokens: int = 510,
    overlap: int = 0,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Tokenize the full raw field only to split it into <= chunk_size_tokens pieces.
    # This raw field can be > encoder model_max_length, which would trigger a
    # HuggingFace warning even though the long sequence is NOT fed to RoBERTa.
    # verbose=False suppresses that pre-chunk warning without changing behavior.
    try:
        ids = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
    except TypeError:
        # Older tokenizer fallback: temporarily lift model_max_length only for
        # this pre-chunk tokenization pass.
        old_model_max_length = getattr(tok, "model_max_length", None)
        try:
            tok.model_max_length = int(1e12)
            ids = tok.encode(text, add_special_tokens=False)
        finally:
            if old_model_max_length is not None:
                tok.model_max_length = old_model_max_length

    if not ids:
        return []

    overlap = max(0, int(overlap))
    chunk_size_tokens = max(1, int(chunk_size_tokens))
    step = max(1, chunk_size_tokens - overlap)

    out: List[str] = []
    for st in range(0, len(ids), step):
        piece = ids[st: st + chunk_size_tokens]
        if not piece:
            break
        s = tok.decode(piece, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if s:
            out.append(s)
        if st + chunk_size_tokens >= len(ids):
            break
    return out


# =========================
# prompt split: only consume first contiguous marker block
# =========================
def _split_prompt_by_first_marker_block(prompt: str) -> Tuple[str, str]:
    """
    切法：
    - 找到第一个 marker（chunk 或 image）出现的位置 l
    - 从 l 开始向右吞掉 marker/空白/marker/空白... 形成的连续块
    - prefix=prompt[:l], suffix=prompt[r:]
    - suffix 内残留 marker 全部清掉，避免模型看到 marker 文本
    """
    prompt = prompt or ""
    idx_chunk = prompt.find(CHUNK_MARKER)
    idx_img = prompt.find(IMAGE_MARKER)
    idxs = [i for i in (idx_chunk, idx_img) if i >= 0]
    if not idxs:
        return prompt, ""

    l = min(idxs)
    r = l
    while True:
        advanced = False
        while r < len(prompt) and prompt[r].isspace():
            r += 1
            advanced = True
        if prompt.startswith(CHUNK_MARKER, r):
            r += len(CHUNK_MARKER)
            advanced = True
            continue
        if prompt.startswith(IMAGE_MARKER, r):
            r += len(IMAGE_MARKER)
            advanced = True
            continue
        if not advanced:
            break

    pre = prompt[:l]
    suf = prompt[r:]
    suf = suf.replace(CHUNK_MARKER, "").replace(IMAGE_MARKER, "")
    return pre, suf


# =========================
# collator (multimodal kept)
# =========================
class DataCollatorSmartCity:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        # chunk
        chunk_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        chunk_size_tokens: int = 510,
        chunk_overlap: int = 0,
        join_fields_before_chunk: bool = True,
        add_field_header: bool = True,

        # image
        max_images_per_sample: int = 1,
        image_size: int = 384,

        # length/pad
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,

        # seg masks
        use_seg_masks: bool = True,

        # IMPORTANT: must match model config
        chunk_token_factor: int = 4,
        image_token_factor: int = 64,

        max_chunks_per_sample: Optional[int] = None,

        add_eos_to_labels: bool = True,
        wrap_reason_with_tags: bool = True,
    ):
        self.tok = tokenizer
        self.chunk_tok = chunk_tokenizer if chunk_tokenizer is not None else tokenizer
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.chunk_overlap = int(chunk_overlap)
        self.join_fields_before_chunk = bool(join_fields_before_chunk)
        self.add_field_header = bool(add_field_header)

        self.max_images_per_sample = max(0, int(max_images_per_sample))
        self.image_size = int(image_size)

        self.max_chunks_per_sample = int(max_chunks_per_sample) if max_chunks_per_sample is not None else None

        self.max_len = max_length or getattr(tokenizer, "model_max_length", 2048)
        if self.max_len is None or self.max_len > 10_000_000:
            self.max_len = 2048

        self.pad_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        assert self.pad_id is not None and self.eos_id is not None, "pad/eos 不能为空"

        self.lab_pad_id = int(label_pad_token_id)
        self.use_seg_masks = bool(use_seg_masks)

        self.chunk_token_factor = int(chunk_token_factor)
        self.image_token_factor = int(image_token_factor)

        self.add_eos_to_labels = bool(add_eos_to_labels)
        self.wrap_reason_with_tags = bool(wrap_reason_with_tags)

        import torchvision.transforms as T
        self._img_tf = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.max_missing_image_ratio = 0.20
        self.min_images_to_check = 200
        self.hard_fail_on_missing = True

        self._img_total = 0
        self._img_missing = 0
        self._img_missing_examples: List[str] = []
        self._img_missing_examples_cap = 20

    def _tok_ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]

    @staticmethod
    def _left_trim(seq, mask, need):
        if need <= 0:
            return seq, mask
        if need >= len(seq):
            return [], []
        return seq[need:], mask[need:]

    def _trim_labels_keep_eos(self, ids, seg_think, seg_label, seg_target, need_drop):
        if need_drop <= 0 or len(ids) <= 1:
            return ids, seg_think, seg_label, seg_target
        has_eos = self.add_eos_to_labels and (ids[-1] == self.eos_id)
        if has_eos:
            new_len = max(1, len(ids) - need_drop)
            if new_len == 1:
                return [self.eos_id], [seg_think[-1]], [seg_label[-1]], [seg_target[-1]]
            return (
                ids[: new_len - 1] + [self.eos_id],
                seg_think[: new_len - 1] + [seg_think[-1]],
                seg_label[: new_len - 1] + [seg_label[-1]],
                seg_target[: new_len - 1] + [seg_target[-1]],
            )
        new_len = max(1, len(ids) - need_drop)
        return ids[:new_len], seg_think[:new_len], seg_label[:new_len], seg_target[:new_len]

    def _to_pil(self, obj) -> Image.Image:
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(obj)).convert("RGB")
        if isinstance(obj, dict) and "path" in obj:
            return Image.open(obj["path"]).convert("RGB")
        if isinstance(obj, str):
            return Image.open(obj).convert("RGB")
        return Image.open(obj).convert("RGB")

    def _gather_image_paths(self, ex: Dict[str, Any]) -> List[str]:
        src = ex.get("images", []) or []
        if not isinstance(src, list):
            src = [src]
        paths = [str(x).strip() for x in src if isinstance(x, str) and str(x).strip()]
        if self.max_images_per_sample > 0:
            paths = paths[: self.max_images_per_sample]
        else:
            paths = []
        return paths

    @staticmethod
    def _normalize_reasoning_mode(mode: Any) -> str:
        m = str(mode or "").strip().lower().replace("-", "_")
        if m in {"think", "thinking", "cot"}:
            return "think"
        if m in {"no_think", "nothink", "no", "none", "instant"}:
            return "no_think"
        return ""

    @staticmethod
    def _strip_think_tags_for_nothink(text: str) -> str:
        # 只做保险清理；正常情况下转换脚本已经保证 no-think answer 没有 <think>。
        s = str(text or "")
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.I | re.S)
        s = re.sub(r"</?think>", "", s, flags=re.I)
        return s.strip()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)

        # ========== A) images -> pixel_values / n_images ==========
        img_paths_per_ex: List[List[str]] = []
        n_images_list: List[int] = []
        for ex in batch:
            paths = self._gather_image_paths(ex)
            img_paths_per_ex.append(paths)
            n_images_list.append(len(paths))

        batch_max_images = max(n_images_list) if (self.max_images_per_sample > 0 and n_images_list) else 0
        pixel_values: Optional[torch.Tensor] = None
        n_images: Optional[torch.Tensor] = None

        if batch_max_images > 0:
            pv_list: List[torch.Tensor] = []
            for i in range(B):
                paths = img_paths_per_ex[i]
                tensors: List[torch.Tensor] = []

                for p in paths[:batch_max_images]:
                    self._img_total += 1
                    try:
                        if (not _is_remote_uri(p)) and (not os.path.exists(p)):
                            raise FileNotFoundError(p)
                        im = self._to_pil(p)
                        tensors.append(self._img_tf(im))
                    except Exception as e:
                        self._img_missing += 1
                        if len(self._img_missing_examples) < self._img_missing_examples_cap:
                            self._img_missing_examples.append(f"{p} | {type(e).__name__}: {e}")
                        tensors.append(torch.zeros(3, self.image_size, self.image_size))

                if len(tensors) < batch_max_images:
                    tensors += [torch.zeros(3, self.image_size, self.image_size)] * (batch_max_images - len(tensors))

                pv_list.append(torch.stack(tensors, dim=0))

            pixel_values = torch.stack(pv_list, dim=0)
            n_images = torch.tensor(n_images_list, dtype=torch.long)

            if self.hard_fail_on_missing and self._img_total >= self.min_images_to_check:
                ratio = self._img_missing / max(1, self._img_total)
                if ratio > self.max_missing_image_ratio:
                    examples = "\n".join(self._img_missing_examples[:10])
                    raise RuntimeError(
                        f"[SmartCity] missing/corrupt images too many: "
                        f"{self._img_missing}/{self._img_total} ({ratio:.2%}) "
                        f"> {self.max_missing_image_ratio:.2%}\n"
                        f"Examples:\n{examples}"
                    )

        # ========== B) chunks: join field -> chunk by encoder tokens ==========
        chunk_txts: List[List[str]] = []
        for ex in batch:
            rows = ex.get("chunks", {}) or {}
            merged: List[str] = []
            if self.join_fields_before_chunk:
                for f in FIELD_ORDER:
                    arr = rows.get(f, None) or []
                    lines = [str(s).strip() for s in arr if isinstance(s, str) and str(s).strip()]
                    if not lines:
                        continue
                    big = "\n".join(lines)
                    if self.add_field_header:
                        big = f"{f}:\n{big}"
                    merged.extend(_chunk_by_token_ids(
                        big,
                        self.chunk_tok,
                        chunk_size_tokens=self.chunk_size_tokens,
                        overlap=self.chunk_overlap,
                    ))
            else:
                for f in FIELD_ORDER:
                    arr = rows.get(f, None)
                    if not arr:
                        continue
                    for s in arr:
                        if isinstance(s, str) and s.strip():
                            merged.append(s.strip())
            chunk_txts.append(merged)

        # ========== C) prompt -> prefix/suffix ==========
        prefixes, suffixes = [], []
        for ex in batch:
            prompt = ex.get("prompt", "") or ""
            pre, suf = _split_prompt_by_first_marker_block(prompt)
            prefixes.append(pre)
            suffixes.append(suf)

        # ========== D) tokenize + labels ==========
        task_ids: List[int] = []
        pre_ids, pre_m = [], []
        suf_ids, suf_m = [], []
        ans_ids: List[List[int]] = []
        seg_think, seg_label, seg_target = [], [], []

        for i, ex in enumerate(batch):
            tname = (ex.get("task", "") or "").lower()
            task_ids.append(TASK2ROUTER.get(tname, 0))

            # understand / prediction 仍然保持旧逻辑。
            if tname in ("understand", "prediction"):
                anchor = "<|im_start|>assistant\n"
                empty_think = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

                suf_text = suffixes[i] or ""
                if anchor in suf_text:
                    pos = suf_text.rfind(anchor) + len(anchor)
                    after = suf_text[pos:]
                    if not after.lstrip().startswith(THINK_OPEN):
                        suf_text = suf_text[:pos] + empty_think + after
                    suffixes[i] = suf_text
                else:
                    pre_text = prefixes[i] or ""
                    if anchor in pre_text:
                        pos = pre_text.rfind(anchor) + len(anchor)
                        after = pre_text[pos:]
                        if not after.lstrip().startswith(THINK_OPEN):
                            pre_text = pre_text[:pos] + empty_think + after
                        prefixes[i] = pre_text

            # Qwen3-style reason prefix for StageD SFT.
            #
            # think:
            #   context gets "<think>\n"
            #   label starts from reasoning body and still generates "</think>" + final answer.
            #
            # no_think:
            #   context gets Qwen non-thinking empty block:
            #   "<think>\n\n</think>\n\n"
            #   label is final answer only.
            if tname == "reason" and SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX:
                mode_for_prefix = self._normalize_reasoning_mode(ex.get("reasoning_mode", ""))

                if mode_for_prefix == "no_think":
                    if SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE == "compact":
                        qwen_reason_prefill = f"{THINK_OPEN}{THINK_CLOSE}\n\n"
                    else:
                        qwen_reason_prefill = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
                else:
                    qwen_reason_prefill = f"{THINK_OPEN}\n"

                anchor = "<|im_start|>assistant\n"

                def _inject_qwen_reason_prefill(txt: str) -> str:
                    if not txt:
                        return txt
                    pos = txt.rfind(anchor)
                    if pos < 0:
                        return txt
                    insert_at = pos + len(anchor)
                    after = txt[insert_at:]
                    if after.lstrip().startswith(THINK_OPEN):
                        return txt
                    return txt[:insert_at] + qwen_reason_prefill + after

                suf_text = suffixes[i] or ""
                pre_text = prefixes[i] or ""

                if anchor in suf_text:
                    suffixes[i] = _inject_qwen_reason_prefill(suf_text)
                elif anchor in pre_text:
                    prefixes[i] = _inject_qwen_reason_prefill(pre_text)
                else:
                    # fallback: append qwen_reason_prefill to suffixes.
                    # Some SmartCity prompts are already split without the literal
                    # <|im_start|>assistant\n anchor in prefix/suffix text. In that
                    # case, suffix_ids are still generation context, so put the
                    # Qwen reason control prefix there.
                    if not suf_text.lstrip().startswith(THINK_OPEN):
                        suffixes[i] = (suf_text or "") + qwen_reason_prefill

            p = self._tok_ids(prefixes[i])
            s = self._tok_ids(suffixes[i])
            pre_ids.append(p)
            pre_m.append([1] * len(p))
            suf_ids.append(s)
            suf_m.append([1] * len(s))

            if tname == "reason":
                mode = self._normalize_reasoning_mode(ex.get("reasoning_mode", ""))
                think_list = ex.get("thinking", []) or []

                if mode == "no_think":
                    think_body = ""
                else:
                    think_body = "\n".join([x for x in think_list if isinstance(x, str) and x.strip()])

                ans_text = ex.get("answer", "") or ""
                if mode == "no_think":
                    ans_text = self._strip_think_tags_for_nothink(ans_text)

                think_body_clean = think_body.strip()

                if SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX:
                    if mode == "no_think":
                        # Empty think block is already in context.
                        # Model generates final answer only.
                        think_text = ""
                    elif think_body_clean:
                        # Opening <think> is already in context.
                        # Model generates reasoning body, </think>, then final answer.
                        think_text = f"{think_body_clean}\n{THINK_CLOSE}\n\n"
                    else:
                        think_text = f"{THINK_CLOSE}\n\n"

                elif self.wrap_reason_with_tags and (think_body_clean or SMARTCITY_REASON_ALWAYS_THINK_TAGS):
                    if think_body_clean:
                        think_text = f"{THINK_OPEN}\n{think_body_clean}\n{THINK_CLOSE}\n\n"
                    else:
                        # no_think target: exact empty think block, then public answer.
                        think_text = f"{THINK_OPEN}{THINK_CLOSE}\n\n"
                else:
                    think_text = ""

                think_ids = self._tok_ids(think_text) if think_text else []
                ans_only_ids = self._tok_ids(ans_text)

                lab = think_ids + ans_only_ids
                st = [True] * len(think_ids) + [False] * len(ans_only_ids)
                sl = [False] * len(think_ids) + [True] * len(ans_only_ids)
                su = [False] * len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id)
                    st.append(False)
                    sl.append(True)
                    su.append(False)
                if len(lab) == 0:
                    lab = [self.eos_id]
                    st, sl, su = [False], [True], [False]

                ans_ids.append(lab)
                seg_think.append(st)
                seg_label.append(sl)
                seg_target.append(su)

            elif tname == "understand":
                tgt = ex.get("target", "") or ""
                lab = self._tok_ids(tgt)
                st = [False] * len(lab)
                sl = [False] * len(lab)
                su = [True] * len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id)
                    st.append(False)
                    sl.append(False)
                    su.append(True)
                if len(lab) == 0:
                    lab = [self.eos_id]
                    st, sl, su = [False], [False], [True]

                ans_ids.append(lab)
                seg_think.append(st)
                seg_label.append(sl)
                seg_target.append(su)

            else:
                ans_text = ex.get("answer", "") or ""
                lab = self._tok_ids(ans_text)
                st = [False] * len(lab)
                sl = [True] * len(lab)
                su = [False] * len(lab)

                if self.add_eos_to_labels and (len(lab) == 0 or lab[-1] != self.eos_id):
                    lab.append(self.eos_id)
                    st.append(False)
                    sl.append(True)
                    su.append(False)
                if len(lab) == 0:
                    lab = [self.eos_id]
                    st, sl, su = [False], [True], [False]

                ans_ids.append(lab)
                seg_think.append(st)
                seg_label.append(sl)
                seg_target.append(su)

        # ========== E) dynamic batch_max_chunks with image tokens in budget ==========
        max_pre_len = max((len(x) for x in pre_ids), default=0)
        max_suf_len = max((len(x) for x in suf_ids), default=0)
        max_ans_len = max((len(x) for x in ans_ids), default=1)

        # GRPO trains on generated completions, not on reference labels.
        # Stage D reference JSON can be long, so do not let it steal prompt/chunk budget.
        ignore_label_budget = os.environ.get("SMARTCITY_GRPO_IGNORE_LABEL_BUDGET", "0").strip().lower() in {"1", "true", "yes", "on"}
        budget_ans_len = 1 if ignore_label_budget else max_ans_len

        raw_max_chunks = max((len(x) for x in chunk_txts), default=0)
        if self.max_chunks_per_sample is not None and self.max_chunks_per_sample > 0:
            raw_max_chunks = min(raw_max_chunks, self.max_chunks_per_sample)
        raw_max_chunks = max(1, int(raw_max_chunks))

        img_block_len = (batch_max_images * self.image_token_factor) if batch_max_images > 0 else 0

        reserved = max_pre_len + max_suf_len + budget_ans_len + img_block_len
        budget = self.max_len - reserved
        if budget <= 0:
            batch_max_chunks = 1
        else:
            batch_max_chunks = max(1, int(budget // self.chunk_token_factor))
            batch_max_chunks = min(batch_max_chunks, raw_max_chunks)

        for i in range(B):
            if len(chunk_txts[i]) > batch_max_chunks:
                chunk_txts[i] = chunk_txts[i][:batch_max_chunks]

        # ========== F) overflow trim (prefix -> suffix -> labels) ==========
        chunk_block_len = batch_max_chunks * self.chunk_token_factor
        for i in range(B):
            p_len, s_len, a_len = len(pre_ids[i]), len(suf_ids[i]), len(ans_ids[i])
            a_budget_len = 1 if ignore_label_budget else a_len
            total = p_len + chunk_block_len + img_block_len + s_len + a_budget_len
            overflow = max(0, total - self.max_len)
            if overflow > 0:
                cut = min(overflow, p_len)
                if cut > 0:
                    pre_ids[i], pre_m[i] = self._left_trim(pre_ids[i], pre_m[i], cut)
                    overflow -= cut
                if overflow > 0:
                    cut = min(overflow, s_len)
                    if cut > 0:
                        suf_ids[i], suf_m[i] = self._left_trim(suf_ids[i], suf_m[i], cut)
                        overflow -= cut
                if overflow > 0 and not ignore_label_budget:
                    ans_ids[i], seg_think[i], seg_label[i], seg_target[i] = self._trim_labels_keep_eos(
                        ans_ids[i], seg_think[i], seg_label[i], seg_target[i], overflow
                    )

        # ========== G) padding ==========
        max_pre = max(1, max(len(x) for x in pre_ids)) if pre_ids else 1
        max_suf = max(1, max(len(x) for x in suf_ids)) if suf_ids else 1
        max_ans = max(1, max(len(x) for x in ans_ids)) if ans_ids else 1

        for i in range(B):
            dp = max_pre - len(pre_ids[i])
            pre_ids[i] = [self.pad_id] * dp + pre_ids[i]
            pre_m[i] = [0] * dp + pre_m[i]

            ds = max_suf - len(suf_ids[i])
            suf_ids[i] = [self.pad_id] * ds + suf_ids[i]
            suf_m[i] = [0] * ds + suf_m[i]

            da = max_ans - len(ans_ids[i])
            ans_ids[i] = ans_ids[i] + [self.lab_pad_id] * da
            seg_think[i] = seg_think[i] + [False] * da
            seg_label[i] = seg_label[i] + [False] * da
            seg_target[i] = seg_target[i] + [False] * da

        out: Dict[str, Any] = {
            "chunk_txts": chunk_txts,
            "prefix_ids": torch.tensor(pre_ids, dtype=torch.long),
            "prefix_mask": torch.tensor(pre_m, dtype=torch.long),
            "suffix_ids": torch.tensor(suf_ids, dtype=torch.long),
            "suffix_mask": torch.tensor(suf_m, dtype=torch.long),
            "labels": torch.tensor(ans_ids, dtype=torch.long),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
        }

        if pixel_values is not None and n_images is not None:
            out["pixel_values"] = pixel_values
            out["n_images"] = n_images

        if self.use_seg_masks:
            out.update({
                "seg_think": torch.tensor(seg_think, dtype=torch.bool),
                "seg_label": torch.tensor(seg_label, dtype=torch.bool),
                "seg_target": torch.tensor(seg_target, dtype=torch.bool),
            })

        out["meta"] = [
            {
                "task": (ex.get("task", "") or ""),
                "sample_id": (ex.get("sample_id", "") or ""),
                "dataset_source": (ex.get("dataset_source", "") or ""),
                "question_type": (ex.get("question_type", "") or ""),
                "target": (ex.get("target", "") or ""),
                "answer": (ex.get("answer", "") or ""),
                "reasoning_mode": (ex.get("reasoning_mode", "") or ""),
                "gold_decision": (ex.get("gold_decision", "") or ""),
                "evidence_catalog": (ex.get("evidence_catalog", "") or ""),
                "key_evidence_ids": (ex.get("key_evidence_ids", []) or []),
                "key_source_types": (ex.get("key_source_types", []) or []),
                "as_of_time": (ex.get("as_of_time", "") or ""),
                "perturbation_type": (ex.get("perturbation_type", "") or ""),
                "expected_behavior": (ex.get("expected_behavior", "") or ""),
                "reward_meta": (ex.get("reward_meta", "") or ""),
            }
            for ex in batch
        ]
        return out


DataCollatorTraffic = DataCollatorSmartCity

__all__ = [
    "load_jsonl_dataset",
    "save_dataset_jsonl",
    "ensure_smartcity_features",
    "DataCollatorSmartCity",
    "DataCollatorTraffic",
    "FIELD_ORDER",
    "TASK2ROUTER",
    "CHUNK_MARKER",
    "IMAGE_MARKER",
    "SMARTCITY_FEATURES",
    "PROMPT_STYLE_QWEN3",
    "PROMPT_STYLE_PLAIN",
]