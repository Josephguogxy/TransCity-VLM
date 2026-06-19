from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set
from datetime import datetime, timezone
import json
import re
import os


@dataclass
class ExampleMeta:
    task: str = ""
    target: str = ""
    answer: str = ""

    # 新增：reason / no-think 与 evidence-grounded reward 所需字段。
    # 允许为空；为空时 reward 自动退回旧 Rouge/Cosine 路径。
    reasoning_mode: str = "auto"       # auto | think | no_think
    gold_decision: str = ""
    evidence_catalog: str = ""         # JSON string: [{id,type,time,location,handle}, ...]
    key_evidence_ids: List[str] = field(default_factory=list)
    key_source_types: List[str] = field(default_factory=list)
    as_of_time: str = ""
    perturbation_type: str = ""        # full | no_evidence | missing_key | wrong | contradictory | shuffled | time_shifted
    expected_behavior: str = ""
    reward_meta: str = ""              # optional JSON string

    @classmethod
    def from_any(cls, x: Any) -> "ExampleMeta":
        if isinstance(x, ExampleMeta):
            return x
        if not isinstance(x, dict):
            return cls()

        rm = _parse_json_like(x.get("reward_meta", ""))
        if not isinstance(rm, dict):
            rm = {}

        def get(name: str, default: Any = "") -> Any:
            v = x.get(name, None)
            if v is None or v == "":
                v = rm.get(name, default)
            return default if v is None else v

        return cls(
            task=str(get("task", "") or ""),
            target=str(get("target", "") or ""),
            answer=str(get("answer", "") or ""),
            reasoning_mode=str(get("reasoning_mode", "auto") or "auto"),
            gold_decision=str(get("gold_decision", "") or ""),
            evidence_catalog=_json_string(get("evidence_catalog", "") or ""),
            key_evidence_ids=_as_str_list(get("key_evidence_ids", [])),
            key_source_types=_as_str_list(get("key_source_types", [])),
            as_of_time=str(get("as_of_time", "") or ""),
            perturbation_type=str(get("perturbation_type", "") or ""),
            expected_behavior=str(get("expected_behavior", "") or ""),
            reward_meta=_json_string(get("reward_meta", "") or ""),
        )

    def has_structured_reward_meta(self) -> bool:
        return bool(
            self.gold_decision.strip()
            or self.evidence_catalog.strip()
            or self.key_evidence_ids
            or self.key_source_types
            or self.perturbation_type.strip()
            or self.expected_behavior.strip()
        )


_THINK_RE_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_RE_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_END_TOKENS = ("<|im_end|>", "<|endoftext|>")
_TOKEN_PROXY_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def _strip_end_tokens(text: str) -> str:
    t = (text or "").strip()
    for end_tok in _END_TOKENS:
        if end_tok in t:
            t = t.split(end_tok, 1)[0].strip()
    return t


def _proxy_len_units(text: str) -> int:
    t = _strip_end_tokens(text)
    return len(_TOKEN_PROXY_RE.findall(t)) if t else 0


def _as_str_list(x: Any) -> List[str]:
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


def _json_string(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def extract_answer(text: str) -> str:
    """
    Qwen3-aligned:
    - 如果有 </think>，取最后一个 </think> 之后的公开答案。
    - 如果没有 </think>，直接用整个 completion；这支持 no-think QA。
    - 去掉尾部控制 token。
    """
    t = _strip_end_tokens(text)
    low = t.lower()
    key = "</think>"
    if key in low:
        j = low.rfind(key)
        t = t[j + len(key):].strip()
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE).strip()
    return t


def soft_format_reward(text: str, meta: Optional[ExampleMeta] = None) -> float:
    """
    Qwen3-style format reward.

    In Qwen3 thinking mode, the chat template may provide the opening thinking
    context and the generated completion often visibly starts with </think>.
    Therefore, for think-mode StageD we reward the closing tag by default.
    """
    t = text or ""
    mode = ((meta.reasoning_mode if meta else "auto") or "auto").strip().lower().replace("-", "_")
    has_open = bool(_THINK_RE_OPEN.search(t))
    has_close = bool(_THINK_RE_CLOSE.search(t))

    if mode in ("no_think", "nothink", "no", "none", "instant"):
        return 1.0 if (not has_open and not has_close) else 0.2

    if mode in ("think", "thinking", "cot"):
        close_only = os.environ.get("SMARTCITY_QWEN3_THINK_CLOSE_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
        if close_only:
            return 1.0 if has_close else 0.0
        return 0.5 * float(has_open) + 0.5 * float(has_close)

    return 1.0 if has_close else 0.6

_ROUGE_SCORER = None
_ST_MODEL = None
_ST_CFG = None


def _tokenize_for_metrics(text: str, mode: str = "space") -> str:
    s = (text or "").strip()
    if mode == "char":
        s = re.sub(r"\s+", "", s)
        return " ".join(list(s))
    if mode == "space":
        return " ".join(s.split())
    return s


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return _ROUGE_SCORER


def _get_st_model(model_name: str, device: str):
    global _ST_MODEL, _ST_CFG
    cfg = (model_name, device)
    if _ST_MODEL is None or _ST_CFG != cfg:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(model_name, device=device)
        _ST_CFG = cfg
    return _ST_MODEL


def _compute_rougeL_f(preds_tok: List[str], refs_tok: List[str]) -> List[float]:
    scorer = _get_rouge_scorer()
    out = []
    for p, r in zip(preds_tok, refs_tok):
        sc = scorer.score(r, p)
        out.append(float(sc["rougeL"].fmeasure))
    return out


def _compute_cos01(
    preds: List[str],
    refs: List[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> List[float]:
    st = _get_st_model(model_name, device=device)
    emb_p = st.encode(
        preds,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_r = st.encode(
        refs,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cos = (emb_p * emb_r).sum(dim=1).clamp(min=-1.0, max=1.0)
    cos01 = (cos + 1.0) * 0.5
    return [float(x) for x in cos01.detach().cpu().tolist()]


def _extract_first_json_object(text: str) -> str:
    s = (text or "").strip()
    start = s.find("{")
    if start < 0:
        return ""
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
                return s[start:i + 1]
    return ""


def parse_structured_answer(answer_text: str) -> Optional[Dict[str, Any]]:
    ans = (answer_text or "").strip()
    if not ans:
        return None
    obj = _parse_json_like(ans)
    if isinstance(obj, dict):
        return obj
    js = _extract_first_json_object(ans)
    if js:
        obj = _parse_json_like(js)
        if isinstance(obj, dict):
            return obj
    return None


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def _norm_id(x: Any) -> str:
    return str(x or "").strip()


def _answer_text_for_metric(ans: str, obj: Optional[Dict[str, Any]]) -> str:
    if not isinstance(obj, dict):
        return ans
    parts: List[str] = []
    for k in ("answer", "final_answer", "final_recommendation", "recommendation", "decision"):
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if not parts:
        return ans
    return "\n".join(parts)


def _catalog_maps(meta: ExampleMeta) -> Tuple[Set[str], Dict[str, str], Dict[str, str], Dict[str, str], Set[str]]:
    ids: Set[str] = set()
    type_by_id: Dict[str, str] = {}
    link_by_id: Dict[str, str] = {}
    time_by_id: Dict[str, str] = {}
    types: Set[str] = set()

    cat = _parse_json_like(meta.evidence_catalog)
    if isinstance(cat, dict):
        cat = [
            {"id": k, **(v if isinstance(v, dict) else {"handle": str(v)})}
            for k, v in cat.items()
        ]
    if isinstance(cat, list):
        for item in cat:
            if isinstance(item, dict):
                eid = _norm_id(item.get("id", item.get("eid", item.get("evidence_id", ""))))
                typ = str(item.get("type", item.get("source_type", item.get("modality", ""))) or "").strip()
                link = str(item.get("link", item.get("url", "")) or "").strip()
                ev_time = str(item.get("time", item.get("timestamp", "")) or "").strip()
            else:
                eid = _norm_id(item)
                typ = ""
                link = ""
                ev_time = ""
            if eid:
                ids.add(eid)
                if typ:
                    type_by_id[eid] = typ
                    types.add(typ.lower())
                if link:
                    link_by_id[eid] = link
                if ev_time:
                    time_by_id[eid] = ev_time

    for eid in meta.key_evidence_ids:
        if eid:
            ids.add(eid)
    for typ in meta.key_source_types:
        if typ:
            types.add(str(typ).strip().lower())
    return ids, type_by_id, link_by_id, time_by_id, types


def _catalog_info(meta: ExampleMeta) -> Tuple[Set[str], Dict[str, str], Set[str]]:
    ids, type_by_id, _, _, types = _catalog_maps(meta)
    return ids, type_by_id, types


def _parse_time(s: Any) -> Optional[datetime]:
    text = str(s or "").strip()
    if not text:
        return None
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text):
        text += "T00:00:00"
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _collect_evidence_links(obj: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not isinstance(obj, dict):
        return []
    links = obj.get("evidence_links", [])
    if not isinstance(links, list):
        return []
    out: List[Dict[str, str]] = []
    for item in links:
        if not isinstance(item, dict):
            continue
        eid = _norm_id(item.get("id", item.get("eid", item.get("evidence_id", ""))))
        link = str(item.get("link", item.get("url", "")) or "").strip()
        source_type = str(item.get("source_type", item.get("type", "")) or "").strip()
        if eid or link:
            out.append({"id": eid, "link": link, "source_type": source_type})
    return out


def _collect_cited_ids(obj: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(obj, dict):
        return []
    out: List[str] = []

    def add_many(v: Any):
        if v is None:
            return
        if isinstance(v, str):
            for z in re.split(r"[,;\s]+", v):
                z = z.strip()
                if z:
                    out.append(z)
        elif isinstance(v, dict):
            for kk in ("id", "eid", "evidence_id"):
                if v.get(kk):
                    out.append(_norm_id(v.get(kk)))
        elif isinstance(v, (list, tuple, set)):
            for z in v:
                add_many(z)

    for k in ("evidence_used", "used_evidence", "citations", "evidence_ids", "used_evidence_ids"):
        add_many(obj.get(k, None))

    add_many(obj.get("evidence_links", None))

    sf = obj.get("supporting_facts", obj.get("evidence_trace", obj.get("trace", [])))
    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, dict):
                for k in ("evidence_ids", "evidence_used", "citations", "source_ids"):
                    add_many(item.get(k, None))

    seen = set()
    uniq = []
    for eid in out:
        eid = _norm_id(eid)
        if eid and eid not in seen:
            seen.add(eid)
            uniq.append(eid)
    return uniq


def _schema_reward(obj: Optional[Dict[str, Any]], meta: ExampleMeta) -> float:
    if not isinstance(obj, dict):
        return 0.0

    score = 0.0
    score += 0.20  # valid JSON object

    if isinstance(obj.get("answer", None), str) and obj.get("answer", "").strip():
        score += 0.20

    if obj.get("decision", None) not in (None, ""):
        score += 0.15

    if isinstance(obj.get("evidence_links", None), list) and len(obj.get("evidence_links", [])) > 0:
        score += 0.15
    elif _collect_cited_ids(obj):
        score += 0.10

    trace = obj.get("supporting_facts", obj.get("evidence_trace", None))
    if isinstance(trace, list) and len(trace) > 0:
        score += 0.15

    if isinstance(obj.get("final_summary", None), str) and obj.get("final_summary", "").strip():
        score += 0.10

    if isinstance(obj.get("limitations", None), list):
        score += 0.05

    return float(min(1.0, score))


def _decision_reward(
    obj: Optional[Dict[str, Any]],
    answer_metric_text: str,
    meta: ExampleMeta,
) -> Optional[float]:
    gold = _norm_text(meta.gold_decision)
    if not gold:
        return None
    pred = ""
    if isinstance(obj, dict):
        pred = _norm_text(obj.get("decision", ""))
    if not pred:
        pred = _norm_text(answer_metric_text)

    if not pred:
        return 0.0
    if pred == gold:
        return 1.0
    if gold in pred:
        return 0.85

    gold_tokens = set(re.split(r"[^a-z0-9_]+", gold)) - {"", "the", "a", "an"}
    pred_tokens = set(re.split(r"[^a-z0-9_]+", pred)) - {"", "the", "a", "an"}
    if gold_tokens:
        return float(len(gold_tokens & pred_tokens) / max(1, len(gold_tokens)))
    return 0.0


def _limitations_text(obj: Optional[Dict[str, Any]]) -> str:
    if not isinstance(obj, dict):
        return ""
    lim = obj.get("limitations", "")
    if isinstance(lim, list):
        return " ".join(str(x) for x in lim)
    return str(lim or "")


def _catalog_missing_info(meta: ExampleMeta) -> Tuple[Set[str], Set[str]]:
    """
    Return:
      missing_ids: evidence ids marked as missing / present=false
      missing_sources: source_type names marked as missing
    """
    missing_ids: Set[str] = set()
    missing_sources: Set[str] = set()

    cat = _parse_json_like(meta.evidence_catalog)
    if isinstance(cat, dict):
        cat = [
            {"id": k, **(v if isinstance(v, dict) else {"handle": str(v)})}
            for k, v in cat.items()
        ]

    if isinstance(cat, list):
        for item in cat:
            if not isinstance(item, dict):
                continue
            eid = _norm_id(item.get("id", item.get("eid", item.get("evidence_id", ""))))
            src = str(item.get("source_type", item.get("type", item.get("modality", ""))) or "").strip()
            status = str(item.get("status", "") or "").strip().lower()
            present = item.get("present", None)

            is_missing = False
            if present is False:
                is_missing = True
            if status in {"missing", "unavailable", "not_available", "absent"}:
                is_missing = True

            if is_missing:
                if eid:
                    missing_ids.add(eid)
                if src:
                    missing_sources.add(src.lower())

    return missing_ids, missing_sources


def _missing_evidence_handling_reward(
    obj: Optional[Dict[str, Any]],
    meta: ExampleMeta,
) -> Tuple[float, Dict[str, float]]:
    """
    Reward intended to replace confidence_behavior.

    Desired behavior:
    - Do not cite missing evidence in evidence_links.
    - Do not use missing evidence in supporting_facts.
    - If evidence is missing, mention it in limitations when possible.
    """
    if not isinstance(obj, dict):
        return 0.0, {
            "missing_evidence_used": 1.0,
            "missing_limitation_coverage": 0.0,
        }

    missing_ids, missing_sources = _catalog_missing_info(meta)
    cited = set(_collect_cited_ids(obj))

    if not missing_ids and not missing_sources:
        return 1.0, {
            "missing_evidence_used": 0.0,
            "missing_limitation_coverage": 1.0,
        }

    used_missing = len(cited & missing_ids) > 0
    no_missing_use_score = 0.0 if used_missing else 1.0

    lim_text = _norm_text(_limitations_text(obj))
    if missing_sources:
        hits = 0
        for src in missing_sources:
            src_norm = src.replace("_", " ")
            if src in lim_text or src_norm in lim_text:
                hits += 1
        limitation_cov = hits / max(1, len(missing_sources))
    else:
        limitation_cov = 1.0

    score = 0.75 * no_missing_use_score + 0.25 * limitation_cov
    return float(max(0.0, min(1.0, score))), {
        "missing_evidence_used": 1.0 if used_missing else 0.0,
        "missing_limitation_coverage": float(limitation_cov),
    }


def _citation_reward(obj: Optional[Dict[str, Any]], meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    valid_ids, type_by_id, link_by_id, time_by_id, source_types = _catalog_maps(meta)
    key_ids = set(_norm_id(x) for x in meta.key_evidence_ids if _norm_id(x))
    cited = _collect_cited_ids(obj)
    links = _collect_evidence_links(obj)

    if not valid_ids and not key_ids and not source_types:
        return 0.5, {
            "citation_valid_rate": 0.0,
            "evidence_link_valid_rate": 0.0,
            "key_evidence_recall": 0.0,
            "citation_count": float(len(cited)),
            "invalid_link_penalty": 0.0,
            "hallucinated_evidence_penalty": 0.0,
            "future_evidence_penalty": 0.0,
            "overcite_penalty": 0.0,
        }

    if not cited:
        return 0.0, {
            "citation_valid_rate": 0.0,
            "evidence_link_valid_rate": 0.0,
            "key_evidence_recall": 0.0,
            "citation_count": 0.0,
            "invalid_link_penalty": 0.0,
            "hallucinated_evidence_penalty": 0.0,
            "future_evidence_penalty": 0.0,
            "overcite_penalty": 0.0,
        }

    if valid_ids:
        valid_cited = [e for e in cited if e in valid_ids]
        precision = len(valid_cited) / max(1, len(cited))
    else:
        precision = 0.7

    if links:
        ok_links = 0
        invalid_links = 0
        hallucinated = 0
        for item in links:
            eid = item.get("id", "")
            link = item.get("link", "")
            if eid not in valid_ids:
                hallucinated += 1
                continue
            expected = link_by_id.get(eid, "")
            if expected and link != expected:
                invalid_links += 1
                continue
            ok_links += 1
        link_valid = ok_links / max(1, len(links))
    else:
        invalid_links = 0
        hallucinated = 0
        link_valid = precision

    if key_ids:
        recall = len(set(cited) & key_ids) / max(1, len(key_ids))
    else:
        recall = 0.7

    if source_types and type_by_id:
        cited_types = {
            type_by_id[e].lower()
            for e in cited
            if e in type_by_id and type_by_id.get(e)
        }
        type_cov = len(cited_types & source_types) / max(1, len(source_types))
    else:
        type_cov = 0.7

    overcite_pen = 0.0
    if key_ids and len(cited) > max(3, 2 * len(key_ids)):
        overcite_pen = min(0.3, 0.05 * (len(cited) - 2 * len(key_ids)))
    elif valid_ids and len(cited) > max(5, int(0.8 * len(valid_ids))):
        overcite_pen = 0.15

    future_pen = 0.0
    as_of = _parse_time(meta.as_of_time)
    if as_of is not None:
        future_hits = 0
        for eid in set(cited):
            ev_time = _parse_time(time_by_id.get(eid, ""))
            if ev_time is not None and ev_time > as_of:
                future_hits += 1
        future_pen = min(0.3, 0.1 * future_hits)

    invalid_link_pen = min(0.3, 0.1 * invalid_links)
    hallucinated_pen = min(0.4, 0.1 * hallucinated)
    score = 0.35 * precision + 0.35 * link_valid + 0.20 * recall + 0.10 * type_cov
    return float(max(0.0, min(1.0, score))), {
        "citation_valid_rate": float(precision),
        "evidence_link_valid_rate": float(link_valid),
        "key_evidence_recall": float(recall),
        "source_type_coverage": float(type_cov),
        "citation_count": float(len(cited)),
        "invalid_link_penalty": float(invalid_link_pen),
        "hallucinated_evidence_penalty": float(hallucinated_pen),
        "future_evidence_penalty": float(future_pen),
        "overcite_penalty": float(overcite_pen),
    }


def _supporting_fact_reward(obj: Optional[Dict[str, Any]], meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    if not isinstance(obj, dict):
        return 0.0, {"supporting_fact_valid_rate": 0.0, "supporting_fact_count": 0.0}
    facts = obj.get("supporting_facts", obj.get("evidence_trace", []))
    if not isinstance(facts, list) or not facts:
        return 0.0, {"supporting_fact_valid_rate": 0.0, "supporting_fact_count": 0.0}
    valid_ids, _, _, _, _ = _catalog_maps(meta)
    total_ids = 0
    valid_ref = 0
    claim_ok = 0
    for item in facts:
        if not isinstance(item, dict):
            continue
        if str(item.get("claim", item.get("fact", "")) or "").strip():
            claim_ok += 1
        ids = _as_str_list(item.get("evidence_ids", item.get("evidence_used", item.get("citations", []))))
        for eid in ids:
            total_ids += 1
            if (not valid_ids) or eid in valid_ids:
                valid_ref += 1
    id_rate = valid_ref / max(1, total_ids)
    claim_rate = claim_ok / max(1, len(facts))
    score = 0.65 * id_rate + 0.35 * claim_rate
    return float(max(0.0, min(1.0, score))), {
        "supporting_fact_valid_rate": float(id_rate),
        "supporting_fact_count": float(len(facts)),
    }


def _contains_uncertainty(text: str) -> bool:
    s = _norm_text(text)
    keys = [
        "insufficient",
        "not enough",
        "uncertain",
        "unknown",
        "low confidence",
        "missing",
        "cannot determine",
        "需要更多",
        "证据不足",
        "无法确定",
        "不确定",
        "低置信",
    ]
    return any(k in s for k in keys)


def _contains_conflict(text: str) -> bool:
    s = _norm_text(text)
    keys = ["conflict", "contradict", "inconsistent", "disagree", "矛盾", "冲突", "不一致"]
    return any(k in s for k in keys)


def _confidence_is_low(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    c = obj.get("confidence", "")
    if isinstance(c, (int, float)):
        return float(c) <= 0.45
    if isinstance(c, dict):
        level = _norm_text(c.get("level", ""))
        try:
            score = float(c.get("score"))
        except Exception:
            score = None
        return level in {"low", "very low", "低", "低置信", "低置信度"} or (score is not None and score <= 0.45)
    s = _norm_text(c)
    return s in {"low", "very low", "低", "低置信", "低置信度"} or "low" in s


def _confidence_is_high(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    c = obj.get("confidence", "")
    if isinstance(c, (int, float)):
        return float(c) >= 0.75
    if isinstance(c, dict):
        level = _norm_text(c.get("level", ""))
        try:
            score = float(c.get("score"))
        except Exception:
            score = None
        return level == "high" or (score is not None and score >= 0.75)
    return "high" in _norm_text(c)


def _uncertainty_reward(obj: Optional[Dict[str, Any]], answer_text: str, meta: ExampleMeta) -> float:
    pt = _norm_text(meta.perturbation_type)
    eb = _norm_text(meta.expected_behavior)
    full_text = answer_text
    if isinstance(obj, dict):
        full_text += "\n" + json.dumps(obj, ensure_ascii=False)

    wants_insufficient = pt in {"no_evidence", "missing_key", "wrong", "shuffled"} or "insufficient" in eb
    wants_conflict = pt in {"contradictory", "conflicting"} or "conflict" in eb
    wants_no_future = pt in {"time_shifted", "future"} or "future" in eb

    if wants_conflict:
        return 1.0 if (_contains_conflict(full_text) or _confidence_is_low(obj)) else 0.0
    if wants_insufficient:
        return 1.0 if (_contains_uncertainty(full_text) or _confidence_is_low(obj)) else 0.0
    if wants_no_future:
        return 1.0 if (_contains_uncertainty(full_text) or _confidence_is_low(obj)) else 0.0

    if pt in {"full", ""}:
        return 0.4 if _contains_uncertainty(full_text) else 1.0
    return 0.5


def _confidence_behavior_reward(
    obj: Optional[Dict[str, Any]],
    answer_text: str,
    meta: ExampleMeta,
    answer_score: float,
) -> float:
    pt = _norm_text(meta.perturbation_type)
    eb = _norm_text(meta.expected_behavior)
    full_text = answer_text
    if isinstance(obj, dict):
        full_text += "\n" + json.dumps(obj, ensure_ascii=False)

    wants_insufficient = pt in {"no_evidence", "missing_key", "wrong", "shuffled"} or "insufficient" in eb
    wants_conflict = pt in {"contradictory", "conflicting"} or "conflict" in eb
    wants_no_future = pt in {"time_shifted", "future"} or "future" in eb

    if wants_conflict:
        if _confidence_is_high(obj):
            return 0.0
        return 1.0 if (_contains_conflict(full_text) or _confidence_is_low(obj)) else 0.4
    if wants_insufficient or wants_no_future:
        return 1.0 if (_contains_uncertainty(full_text) or _confidence_is_low(obj)) else 0.0
    if pt in {"full", ""}:
        if answer_score >= 0.8:
            return 1.0 if not _confidence_is_low(obj) else 0.4
        return 0.8 if not _confidence_is_high(obj) else 0.5
    return 0.5



# ===== SmartCity StageD freeform evidence reward v1 =====
_EVIDENCE_ID_RE = re.compile(r"\bE_[A-Za-z0-9_]+\b")
_FREEFORM_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "as", "that", "this", "it", "its", "there", "based",
    "about", "during", "after", "before", "than", "into", "across", "given", "using", "use",
}


def _gold_structured_obj(meta: ExampleMeta) -> Optional[Dict[str, Any]]:
    for raw in (meta.answer, meta.target):
        obj = parse_structured_answer(str(raw or ""))
        if isinstance(obj, dict):
            return obj
    return None


def _gold_decision(meta: ExampleMeta) -> str:
    gd = _norm_text(meta.gold_decision)
    if gd:
        return gd
    obj = _gold_structured_obj(meta)
    if isinstance(obj, dict):
        return _norm_text(obj.get("decision", ""))
    return ""


def _gold_public_answer(meta: ExampleMeta) -> str:
    obj = _gold_structured_obj(meta)
    if isinstance(obj, dict):
        parts = []
        for k in ("answer", "final_summary", "decision"):
            v = obj.get(k, None)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            return "\n".join(parts)
    return extract_answer(meta.answer or meta.target or "")


def _text_keywords(text: str) -> Set[str]:
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+|\d+(?:\.\d+)?", text or "")]
    return {t for t in toks if len(t) > 2 and t not in _FREEFORM_STOPWORDS}


def _simple_text_f1(pred: str, ref: str) -> float:
    p = _text_keywords(pred)
    r = _text_keywords(ref)
    if not p or not r:
        return 0.0
    hit = len(p & r)
    prec = hit / max(1, len(p))
    rec = hit / max(1, len(r))
    if prec + rec <= 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _extract_freeform_evidence_ids(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in _EVIDENCE_ID_RE.finditer(text or ""):
        eid = m.group(0).strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _freeform_decision_reward(answer_text: str, meta: ExampleMeta) -> float:
    gold = _gold_decision(meta)
    pred = _norm_text(answer_text)
    if not gold or not pred:
        return 0.0

    if pred == gold:
        return 1.0
    if gold in pred:
        return 0.9

    no_sig = bool(re.search(r"\b(no|not|without|lack of|no evidence of)\s+(significant|substantial|major|abnormal)", pred))
    low_risk_cue = any(k in pred for k in [
        "low risk", "negative or low risk", "no concern", "not concerned", "no significant concern",
        "minimal influence", "minimal impact", "not indicate stress", "not indicate pressure",
        "aligns with", "consistent with", "historical average", "typical", "expected trend",
        "normal pattern", "no anomaly", "not abnormal",
    ])

    sig_change = bool(re.search(r"\b(significant|substantial|major|abnormal)\s+(change|changes|anomaly|risk|pressure|concern|deviation)", pred)) and not no_sig
    high_risk_cue = any(k in pred for k in [
        "high risk", "positive or high risk", "operator should be concerned", "reserve margin pressure",
        "material concern", "event driven", "event-driven", "directly caused", "likely caused",
    ])

    if gold in {"negative_or_low_risk", "negative", "low_risk", "low risk", "negative or low risk"}:
        if no_sig or low_risk_cue:
            return 1.0
        if sig_change or high_risk_cue:
            return 0.0
        return 0.25 * _simple_text_f1(answer_text, _gold_public_answer(meta))

    if gold in {"positive_or_high_risk", "positive", "high_risk", "high risk", "positive or high risk"}:
        if sig_change or high_risk_cue:
            return 1.0
        if no_sig or low_risk_cue:
            return 0.0
        return 0.25 * _simple_text_f1(answer_text, _gold_public_answer(meta))

    gold_tokens = set(re.split(r"[^a-z0-9_]+", gold)) - {"", "the", "a", "an"}
    pred_tokens = set(re.split(r"[^a-z0-9_]+", pred)) - {"", "the", "a", "an"}
    if gold_tokens:
        return float(len(gold_tokens & pred_tokens) / max(1, len(gold_tokens)))
    return 0.0


def _freeform_citation_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    valid_ids, type_by_id, _, time_by_id, source_types = _catalog_maps(meta)
    missing_ids, _ = _catalog_missing_info(meta)
    present_ids = set(valid_ids) - set(missing_ids)
    key_ids_all = {_norm_id(x) for x in meta.key_evidence_ids if _norm_id(x)}
    key_ids = key_ids_all & present_ids if present_ids else key_ids_all
    cited = _extract_freeform_evidence_ids(answer_text)

    if not valid_ids and not key_ids and not source_types:
        return 0.5, {
            "citation_valid_rate": 0.0,
            "evidence_link_valid_rate": 0.0,
            "key_evidence_recall": 0.0,
            "source_type_coverage": 0.0,
            "citation_count": float(len(cited)),
            "invalid_link_penalty": 0.0,
            "hallucinated_evidence_penalty": 0.0,
            "future_evidence_penalty": 0.0,
            "overcite_penalty": 0.0,
        }

    if not cited:
        return 0.0, {
            "citation_valid_rate": 0.0,
            "evidence_link_valid_rate": 0.0,
            "key_evidence_recall": 0.0,
            "source_type_coverage": 0.0,
            "citation_count": 0.0,
            "invalid_link_penalty": 0.0,
            "hallucinated_evidence_penalty": 0.0,
            "future_evidence_penalty": 0.0,
            "overcite_penalty": 0.0,
        }

    present_or_valid = present_ids if present_ids else valid_ids
    valid_cited = [e for e in cited if e in present_or_valid]
    hallucinated = len([e for e in cited if e not in valid_ids])
    missing_used = len([e for e in cited if e in missing_ids])
    precision = len(valid_cited) / max(1, len(cited))

    # Natural-language answers should not need to cite every key id.
    # Reward saturates after 3 useful key evidence ids.
    if key_ids:
        key_hits = len(set(valid_cited) & key_ids)
        recall_full = key_hits / max(1, len(key_ids))
        recall_sat = min(1.0, key_hits / max(1, min(3, len(key_ids))))
    else:
        recall_full = 0.7
        recall_sat = 0.7

    if source_types and type_by_id:
        cited_types = {type_by_id[e].lower() for e in valid_cited if e in type_by_id and type_by_id.get(e)}
        cov_full = len(cited_types & source_types) / max(1, len(source_types))
        cov_sat = min(1.0, len(cited_types & source_types) / max(1, min(3, len(source_types))))
    else:
        cov_full = 0.7
        cov_sat = 0.7

    future_pen = 0.0
    as_of = _parse_time(meta.as_of_time)
    if as_of is not None:
        future_hits = 0
        for eid in set(valid_cited):
            ev_time = _parse_time(time_by_id.get(eid, ""))
            if ev_time is not None and ev_time > as_of:
                future_hits += 1
        future_pen = min(0.3, 0.1 * future_hits)

    hallucinated_pen = min(0.4, 0.1 * hallucinated)
    missing_pen = min(0.4, 0.15 * missing_used)
    overcite_pen = min(0.25, 0.03 * max(0, len(cited) - 8))

    citation_count_ok = min(1.0, len(valid_cited) / 3.0)
    score = 0.45 * precision + 0.25 * recall_sat + 0.15 * cov_sat + 0.15 * citation_count_ok
    score = max(0.0, min(1.0, score - hallucinated_pen - missing_pen - future_pen - overcite_pen))

    return float(score), {
        "citation_valid_rate": float(precision),
        "evidence_link_valid_rate": float(precision),
        "key_evidence_recall": float(recall_full),
        "source_type_coverage": float(cov_full),
        "citation_count": float(len(cited)),
        "invalid_link_penalty": 0.0,
        "hallucinated_evidence_penalty": float(hallucinated_pen),
        "future_evidence_penalty": float(future_pen),
        "overcite_penalty": float(overcite_pen),
    }


def _freeform_supporting_fact_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    valid_ids, _, _, _, _ = _catalog_maps(meta)
    missing_ids, _ = _catalog_missing_info(meta)
    present_ids = set(valid_ids) - set(missing_ids)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", answer_text or "") if s.strip()]
    cited_sentences = []
    valid_ref = 0
    total_ref = 0

    for sent in sentences:
        ids = _extract_freeform_evidence_ids(sent)
        if not ids:
            continue
        cited_sentences.append(sent)
        for eid in ids:
            total_ref += 1
            if (not present_ids and eid in valid_ids) or eid in present_ids:
                valid_ref += 1

    if not cited_sentences:
        return 0.0, {"supporting_fact_valid_rate": 0.0, "supporting_fact_count": 0.0}

    id_rate = valid_ref / max(1, total_ref)
    claim_rate = sum(1 for s in cited_sentences if len(_text_keywords(s)) >= 4) / max(1, len(cited_sentences))
    count_score = min(1.0, len(cited_sentences) / 3.0)
    score = 0.45 * id_rate + 0.35 * claim_rate + 0.20 * count_score

    return float(max(0.0, min(1.0, score))), {
        "supporting_fact_valid_rate": float(id_rate),
        "supporting_fact_count": float(len(cited_sentences)),
    }


def _freeform_missing_evidence_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    missing_ids, missing_sources = _catalog_missing_info(meta)
    cited = set(_extract_freeform_evidence_ids(answer_text))
    used_missing = len(cited & missing_ids) > 0

    if not missing_ids and not missing_sources:
        return 1.0, {"missing_evidence_used": 0.0, "missing_limitation_coverage": 1.0}

    text = _norm_text(answer_text)
    hits = 0

    for src in missing_sources:
        src0 = src.lower().replace("_", " ")
        patterns = [
            f"no {src0} evidence",
            f"{src0} evidence was not provided",
            f"{src0} evidence is not provided",
            f"{src0} evidence unavailable",
            f"{src0} evidence is unavailable",
            f"missing {src0}",
            f"no available {src0}",
            f"no {src0} data",
        ]
        if any(p in text for p in patterns):
            hits += 1

    coverage = hits / max(1, len(missing_sources)) if missing_sources else 1.0
    no_missing_use_score = 0.0 if used_missing else 1.0

    # Do not over-reward merely avoiding missing IDs; mentioning limitations matters.
    score = 0.40 * no_missing_use_score + 0.60 * coverage
    return float(max(0.0, min(1.0, score))), {
        "missing_evidence_used": 1.0 if used_missing else 0.0,
        "missing_limitation_coverage": float(coverage),
    }


def _freeform_evidence_reward(answer_text: str, metric_text: str, meta: ExampleMeta, sem: float, fmt: float) -> Tuple[float, Dict[str, Any]]:
    decision = _freeform_decision_reward(metric_text or answer_text, meta)
    citation, cinfo = _freeform_citation_reward(answer_text, meta)
    supporting, sinfo = _freeform_supporting_fact_reward(answer_text, meta)
    missing, minfo = _freeform_missing_evidence_reward(answer_text, meta)
    text_sim = _simple_text_f1(metric_text or answer_text, _gold_public_answer(meta))

    natural_schema = 1.0 if str(answer_text or "").strip() else 0.0

    score = (
        0.45 * decision
        + 0.30 * citation
        + 0.10 * supporting
        + 0.05 * missing
        + 0.05 * fmt
        + 0.05 * max(float(sem), float(text_sim))
    )

    return float(max(0.0, min(1.0, score))), {
        "answer_score": float(decision),
        "schema": float(natural_schema),
        "citation": float(citation),
        "supporting": float(supporting),
        "missing": float(missing),
        "semantic": float(max(float(sem), float(text_sim))),
        "cinfo": cinfo,
        "sinfo": sinfo,
        "minfo": minfo,
    }
# ===== End SmartCity StageD freeform evidence reward v1 =====


# ===== SmartCity StageD freeform Qwen evidence reward v1 =====
_EVIDENCE_ID_RE = re.compile(r"\bE_[A-Za-z0-9_]+\b")
_FREEFORM_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "as", "that", "this", "it", "its", "there", "based",
    "about", "during", "after", "before", "than", "into", "across", "given", "using", "use",
    "flow", "traffic", "load", "sensor", "time", "window",
}


def _gold_structured_obj(meta: ExampleMeta) -> Optional[Dict[str, Any]]:
    for raw in (meta.answer, meta.target):
        obj = parse_structured_answer(str(raw or ""))
        if isinstance(obj, dict):
            return obj
    return None


def _gold_decision(meta: ExampleMeta) -> str:
    gd = _norm_text(meta.gold_decision)
    if gd:
        return gd
    obj = _gold_structured_obj(meta)
    if isinstance(obj, dict):
        return _norm_text(obj.get("decision", ""))
    return ""


def _gold_public_answer(meta: ExampleMeta) -> str:
    obj = _gold_structured_obj(meta)
    if isinstance(obj, dict):
        parts = []
        for k in ("answer", "final_summary", "decision"):
            v = obj.get(k, None)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            return "\n".join(parts)
    return extract_answer(meta.answer or meta.target or "")


def _text_keywords(text: str) -> Set[str]:
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+|\d+(?:\.\d+)?", text or "")]
    return {t for t in toks if len(t) > 2 and t not in _FREEFORM_STOPWORDS}


def _simple_text_f1(pred: str, ref: str) -> float:
    p = _text_keywords(pred)
    r = _text_keywords(ref)
    if not p or not r:
        return 0.0
    hit = len(p & r)
    prec = hit / max(1, len(p))
    rec = hit / max(1, len(r))
    if prec + rec <= 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _extract_freeform_evidence_ids(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in _EVIDENCE_ID_RE.finditer(text or ""):
        eid = m.group(0).strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _freeform_format_score(answer_text: str) -> float:
    ans = (answer_text or "").strip()
    if not ans:
        return 0.0

    score = 0.45

    # We want natural-language final answers, not JSON objects.
    if not ans.startswith("{"):
        score += 0.25

    if "confidence" not in ans.lower():
        score += 0.15

    if _extract_freeform_evidence_ids(ans):
        score += 0.15

    return float(min(1.0, score))


def _has_negated_significant_change(text: str) -> bool:
    return bool(re.search(
        r"\b(no|not|without|lack of|little evidence of|no evidence of)\s+"
        r"(significant|substantial|major|abnormal|material)\s+"
        r"(change|changes|anomaly|risk|pressure|concern|deviation|impact)",
        text,
    ))


def _has_strong_positive_change(text: str) -> bool:
    if _has_negated_significant_change(text):
        return False

    patterns = [
        r"\bthere\s+(were|are|was|is)\s+(significant|substantial|major|abnormal|material)\s+changes?\b",
        r"\b(significant|substantial|major|abnormal|material)\s+(traffic\s+)?(change|changes|anomaly|deviation|risk|pressure|concern)\b",
        r"\b(high risk|operator should be concerned|reserve margin pressure|material concern)\b",
        r"\b(event[- ]driven|directly caused|likely caused)\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _has_low_risk_cue(text: str) -> bool:
    patterns = [
        r"\b(no|not)\s+significant\b",
        r"\blow risk\b",
        r"\bnegative or low risk\b",
        r"\bno concern\b",
        r"\bnot concerned\b",
        r"\bminimal (influence|impact)\b",
        r"\baligns? with\b",
        r"\bconsistent with\b",
        r"\bhistorical average\b",
        r"\bexpected trend\b",
        r"\bnormal pattern\b",
        r"\bno anomaly\b",
        r"\bnot abnormal\b",
        r"\btypical\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _freeform_decision_reward(answer_text: str, meta: ExampleMeta) -> float:
    gold = _gold_decision(meta)
    pred = _norm_text(answer_text)

    if not gold or not pred:
        return 0.0

    if pred == gold:
        return 1.0

    if gold in pred:
        return 0.9

    strong_pos = _has_strong_positive_change(pred)
    low_cue = _has_low_risk_cue(pred)
    neg_sig = _has_negated_significant_change(pred)

    if gold in {"negative_or_low_risk", "negative", "low_risk", "low risk", "negative or low risk"}:
        if strong_pos and not (neg_sig or low_cue):
            return 0.0
        if neg_sig or low_cue:
            return 1.0
        return 0.25 * _simple_text_f1(answer_text, _gold_public_answer(meta))

    if gold in {"positive_or_high_risk", "positive", "high_risk", "high risk", "positive or high risk"}:
        if strong_pos:
            return 1.0
        if neg_sig or low_cue:
            return 0.0
        return 0.25 * _simple_text_f1(answer_text, _gold_public_answer(meta))

    gold_tokens = set(re.split(r"[^a-z0-9_]+", gold)) - {"", "the", "a", "an"}
    pred_tokens = set(re.split(r"[^a-z0-9_]+", pred)) - {"", "the", "a", "an"}
    if gold_tokens:
        return float(len(gold_tokens & pred_tokens) / max(1, len(gold_tokens)))
    return 0.0


def _freeform_citation_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    valid_ids, type_by_id, _, time_by_id, source_types = _catalog_maps(meta)
    missing_ids, _ = _catalog_missing_info(meta)

    present_ids = set(valid_ids) - set(missing_ids)
    key_ids_all = {_norm_id(x) for x in meta.key_evidence_ids if _norm_id(x)}
    key_ids = key_ids_all & present_ids if present_ids else key_ids_all
    cited = _extract_freeform_evidence_ids(answer_text)

    empty_info = {
        "citation_valid_rate": 0.0,
        "evidence_link_valid_rate": 0.0,
        "key_evidence_recall": 0.0,
        "source_type_coverage": 0.0,
        "citation_count": float(len(cited)),
        "invalid_link_penalty": 0.0,
        "hallucinated_evidence_penalty": 0.0,
        "future_evidence_penalty": 0.0,
        "overcite_penalty": 0.0,
    }

    if not valid_ids and not key_ids and not source_types:
        return 0.5, empty_info

    if not cited:
        return 0.0, empty_info

    support_ids = present_ids if present_ids else valid_ids
    valid_cited = [e for e in cited if e in support_ids]
    hallucinated = len([e for e in cited if e not in valid_ids])
    missing_used = len([e for e in cited if e in missing_ids])

    precision = len(valid_cited) / max(1, len(cited))

    if key_ids:
        key_hits = len(set(valid_cited) & key_ids)
        recall_full = key_hits / max(1, len(key_ids))
        recall_sat = min(1.0, key_hits / max(1, min(3, len(key_ids))))
    else:
        recall_full = 0.7
        recall_sat = 0.7

    if source_types and type_by_id:
        cited_types = {
            type_by_id[e].lower()
            for e in valid_cited
            if e in type_by_id and type_by_id.get(e)
        }
        cov_full = len(cited_types & source_types) / max(1, len(source_types))
        cov_sat = min(1.0, len(cited_types & source_types) / max(1, min(3, len(source_types))))
    else:
        cov_full = 0.7
        cov_sat = 0.7

    future_pen = 0.0
    as_of = _parse_time(meta.as_of_time)
    if as_of is not None:
        future_hits = 0
        for eid in set(valid_cited):
            ev_time = _parse_time(time_by_id.get(eid, ""))
            if ev_time is not None and ev_time > as_of:
                future_hits += 1
        future_pen = min(0.3, 0.1 * future_hits)

    hallucinated_pen = min(0.4, 0.1 * hallucinated)
    missing_pen = min(0.4, 0.15 * missing_used)
    overcite_pen = min(0.25, 0.03 * max(0, len(cited) - 8))

    count_score = min(1.0, len(valid_cited) / 3.0)
    score = 0.45 * precision + 0.25 * recall_sat + 0.15 * cov_sat + 0.15 * count_score
    score = max(0.0, min(1.0, score - hallucinated_pen - missing_pen - future_pen - overcite_pen))

    return float(score), {
        "citation_valid_rate": float(precision),
        "evidence_link_valid_rate": float(precision),
        "key_evidence_recall": float(recall_full),
        "source_type_coverage": float(cov_full),
        "citation_count": float(len(cited)),
        "invalid_link_penalty": 0.0,
        "hallucinated_evidence_penalty": float(hallucinated_pen),
        "future_evidence_penalty": float(future_pen),
        "overcite_penalty": float(overcite_pen),
    }


def _freeform_supporting_fact_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    valid_ids, _, _, _, _ = _catalog_maps(meta)
    missing_ids, _ = _catalog_missing_info(meta)
    present_ids = set(valid_ids) - set(missing_ids)

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？])\s+|\n+", answer_text or "")
        if s.strip()
    ]

    cited_sentences = []
    valid_ref = 0
    total_ref = 0

    for sent in sentences:
        ids = _extract_freeform_evidence_ids(sent)
        if not ids:
            continue
        cited_sentences.append(sent)
        for eid in ids:
            total_ref += 1
            if (not present_ids and eid in valid_ids) or eid in present_ids:
                valid_ref += 1

    if not cited_sentences:
        return 0.0, {
            "supporting_fact_valid_rate": 0.0,
            "supporting_fact_count": 0.0,
        }

    id_rate = valid_ref / max(1, total_ref)
    claim_rate = sum(1 for sent in cited_sentences if len(_text_keywords(sent)) >= 4) / max(1, len(cited_sentences))
    count_score = min(1.0, len(cited_sentences) / 3.0)

    score = 0.45 * id_rate + 0.35 * claim_rate + 0.20 * count_score

    return float(max(0.0, min(1.0, score))), {
        "supporting_fact_valid_rate": float(id_rate),
        "supporting_fact_count": float(len(cited_sentences)),
    }


def _freeform_missing_evidence_reward(answer_text: str, meta: ExampleMeta) -> Tuple[float, Dict[str, float]]:
    missing_ids, missing_sources = _catalog_missing_info(meta)
    cited = set(_extract_freeform_evidence_ids(answer_text))
    used_missing = len(cited & missing_ids) > 0

    if not missing_ids and not missing_sources:
        return 1.0, {
            "missing_evidence_used": 0.0,
            "missing_limitation_coverage": 1.0,
        }

    text = _norm_text(answer_text)
    hits = 0

    for src in missing_sources:
        src0 = src.lower().replace("_", " ")
        patterns = [
            f"no {src0} evidence",
            f"{src0} evidence was not provided",
            f"{src0} evidence is not provided",
            f"{src0} evidence unavailable",
            f"{src0} evidence is unavailable",
            f"missing {src0}",
            f"no available {src0}",
            f"no {src0} data",
            f"{src0} is missing",
        ]
        if any(p in text for p in patterns):
            hits += 1

    coverage = hits / max(1, len(missing_sources)) if missing_sources else 1.0
    no_missing_use_score = 0.0 if used_missing else 1.0

    score = 0.40 * no_missing_use_score + 0.60 * coverage

    return float(max(0.0, min(1.0, score))), {
        "missing_evidence_used": 1.0 if used_missing else 0.0,
        "missing_limitation_coverage": float(coverage),
    }


def _freeform_evidence_reward(
    answer_text: str,
    metric_text: str,
    meta: ExampleMeta,
    sem: float,
    fmt: float,
) -> Tuple[float, Dict[str, Any]]:
    decision = _freeform_decision_reward(metric_text or answer_text, meta)
    citation, cinfo = _freeform_citation_reward(answer_text, meta)
    supporting, sinfo = _freeform_supporting_fact_reward(answer_text, meta)
    missing, minfo = _freeform_missing_evidence_reward(answer_text, meta)
    text_sim = _simple_text_f1(metric_text or answer_text, _gold_public_answer(meta))
    natural_schema = _freeform_format_score(answer_text)

    # Answer-focused freeform reward.
    # Default weights are tuned for direct no-think citefit GRPO:
    # pull answer text toward gold while preserving decision/citation safety.
    W_DEC = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_DECISION", "0.30") or "0.30")
    W_CITE = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_CITATION", "0.20") or "0.20")
    W_SUP = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_SUPPORTING", "0.08") or "0.08")
    W_MISS = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_MISSING", "0.04") or "0.04")
    W_FMT = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_FORMAT", "0.03") or "0.03")
    W_TEXT = float(os.environ.get("SMARTCITY_REWARD_FREEFORM_W_TEXT", "0.35") or "0.35")
    wsum = max(1e-6, W_DEC + W_CITE + W_SUP + W_MISS + W_FMT + W_TEXT)

    score = (
        W_DEC * decision
        + W_CITE * citation
        + W_SUP * supporting
        + W_MISS * missing
        + W_FMT * natural_schema
        + W_TEXT * max(float(sem), float(text_sim))
    ) / wsum

    return float(max(0.0, min(1.0, score))), {
        "answer_score": float(decision),
        "schema": float(natural_schema),
        "citation": float(citation),
        "supporting": float(supporting),
        "missing": float(missing),
        "semantic": float(max(float(sem), float(text_sim))),
        "cinfo": cinfo,
        "sinfo": sinfo,
        "minfo": minfo,
        "freeform_format": float(natural_schema),
    }
# ===== End SmartCity StageD freeform Qwen evidence reward v1 =====

def compute_rewards(
    completions: List[str],
    metas_rep: List[ExampleMeta],
    *,
    return_info: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    StageD reward.

    Default legacy mode keeps structured JSON reward.

    Freeform Qwen mode is enabled with:
        SMARTCITY_REWARD_FREEFORM_EVIDENCE=1

    In freeform mode, the policy is rewarded for Qwen3-style output:
        </think>
        natural-language answer with [E_*] evidence citations

    The dataset's JSON answer is still used as hidden gold metadata for decision,
    evidence, supporting facts, and limitation scoring. The model itself is not
    required to output JSON.
    """
    use_semantic_metrics = os.environ.get("SMARTCITY_REWARD_USE_SEMANTIC", "0").strip().lower() in {"1", "true", "yes", "on"}
    freeform_evidence_mode = os.environ.get("SMARTCITY_REWARD_FREEFORM_EVIDENCE", "0").strip().lower() in {"1", "true", "yes", "on"}

    metric_tokenize = "space"
    metric_batch_size = 16
    cos_model = "sentence-transformers/all-MiniLM-L6-v2"
    metric_device = "cpu"

    FREE_LEN = int(os.environ.get("SMARTCITY_REWARD_FREE_LEN", "650") or "650")
    W_LEN = float(os.environ.get("SMARTCITY_REWARD_LEN_W", "0.08") or "0.08")
    W_MISS_THINK_CLOSE = float(os.environ.get("SMARTCITY_REWARD_MISS_THINK_CLOSE_W", "0.03") or "0.03")
    W_NO_THINK_TAG_PEN = float(os.environ.get("SMARTCITY_REWARD_NO_THINK_TAG_W", "0.02") or "0.02")

    n = len(completions)
    assert len(metas_rep) == n
    metas = [ExampleMeta.from_any(m) for m in metas_rep]

    answers: List[str] = []
    objs: List[Optional[Dict[str, Any]]] = []
    metric_preds: List[str] = []
    refs: List[str] = []
    comp_lens: List[int] = []
    pred_lens: List[int] = []
    fmt: List[float] = []
    has_think_close: List[bool] = []

    for text, meta in zip(completions, metas):
        text0 = text or ""
        ans = extract_answer(text0)
        obj = parse_structured_answer(ans)
        metric_text = _answer_text_for_metric(ans, obj)

        answers.append(ans)
        objs.append(obj)
        metric_preds.append(metric_text)
        refs.append((meta.answer or meta.target or "").strip())
        comp_lens.append(_proxy_len_units(text0))
        pred_lens.append(len(ans.strip()))
        fmt.append(soft_format_reward(text0, meta))
        has_think_close.append(bool(_THINK_RE_CLOSE.search(text0)))

    idx = [i for i, r in enumerate(refs) if r.strip() != ""]
    r_rL = [0.0] * n
    r_cos = [0.0] * n

    if idx and use_semantic_metrics:
        sub_preds = [metric_preds[i] for i in idx]
        sub_refs = [refs[i] for i in idx]
        sub_preds_tok = [_tokenize_for_metrics(x, metric_tokenize) for x in sub_preds]
        sub_refs_tok = [_tokenize_for_metrics(x, metric_tokenize) for x in sub_refs]
        rL_sub = _compute_rougeL_f(sub_preds_tok, sub_refs_tok)
        cos_sub = _compute_cos01(
            sub_preds,
            sub_refs,
            model_name=cos_model,
            device=metric_device,
            batch_size=metric_batch_size,
        )
        for j, i in enumerate(idx):
            r_rL[i] = float(rL_sub[j])
            r_cos[i] = float(cos_sub[j])

    len_pen: List[float] = []
    for L in comp_lens:
        over = max(0.0, (float(L) - float(FREE_LEN)) / max(1.0, float(FREE_LEN)))
        len_pen.append(min(1.0, over))

    rewards: List[float] = []
    schema_vals: List[float] = []
    citation_vals: List[float] = []
    supporting_vals: List[float] = []
    missing_evidence_vals: List[float] = []
    missing_evidence_used_rates: List[float] = []
    missing_limitation_coverages: List[float] = []
    answer_vals: List[float] = []
    structured_flags: List[bool] = []
    freeform_used_flags: List[bool] = []
    freeform_format_vals: List[float] = []
    citation_valid_rates: List[float] = []
    evidence_link_valid_rates: List[float] = []
    supporting_fact_valid_rates: List[float] = []
    key_recalls: List[float] = []
    source_covs: List[float] = []
    invalid_link_pens: List[float] = []
    hallucinated_pens: List[float] = []
    future_pens: List[float] = []
    overcite_pens: List[float] = []
    citation_counts: List[float] = []

    for i in range(n):
        meta = metas[i]
        obj = objs[i]
        sem = 0.45 * r_rL[i] + 0.55 * r_cos[i]

        structured = bool(meta.has_structured_reward_meta() or isinstance(obj, dict))
        structured_flags.append(structured)

        freeform_used = bool(freeform_evidence_mode and structured)

        if freeform_used:
            base, finfo = _freeform_evidence_reward(answers[i], metric_preds[i], meta, sem, fmt[i])
            answer_score = float(finfo.get("answer_score", 0.0))
            schema = float(finfo.get("schema", 0.0))
            citation = float(finfo.get("citation", 0.0))
            supporting = float(finfo.get("supporting", 0.0))
            missing_evidence = float(finfo.get("missing", 0.0))
            cinfo = dict(finfo.get("cinfo", {}))
            sinfo = dict(finfo.get("sinfo", {}))
            minfo = dict(finfo.get("minfo", {}))
            freeform_format = float(finfo.get("freeform_format", schema))

        else:
            schema = _schema_reward(obj, meta)
            citation, cinfo = _citation_reward(obj, meta)
            supporting, sinfo = _supporting_fact_reward(obj, meta)
            missing_evidence, minfo = _missing_evidence_handling_reward(obj, meta)
            dec = _decision_reward(obj, metric_preds[i], meta)
            answer_score = float(dec) if dec is not None else float(sem)
            freeform_format = 0.0

            if structured:
                base = (
                    0.25 * answer_score
                    + 0.15 * sem
                    + 0.20 * citation
                    + 0.15 * supporting
                    + 0.10 * schema
                    + 0.10 * missing_evidence
                    + 0.05 * fmt[i]
                )
            else:
                base = 0.05 * fmt[i] + 0.45 * r_rL[i] + 0.50 * r_cos[i]

        mode = (meta.reasoning_mode or "auto").strip().lower().replace("-", "_")

        miss_close_pen = 0.0
        if mode in ("think", "thinking", "cot"):
            miss_close_pen = W_MISS_THINK_CLOSE * (0.0 if has_think_close[i] else 1.0)
        elif mode in ("auto", ""):
            miss_close_pen = W_MISS_THINK_CLOSE * (0.0 if has_think_close[i] else 1.0)
        elif mode in ("no_think", "nothink", "no", "none", "instant"):
            miss_close_pen = W_NO_THINK_TAG_PEN * (
                1.0 if (_THINK_RE_OPEN.search(completions[i] or "") or has_think_close[i]) else 0.0
            )

        invalid_link_pen = float(cinfo.get("invalid_link_penalty", 0.0))
        hallucinated_pen = float(cinfo.get("hallucinated_evidence_penalty", 0.0))
        future_pen = float(cinfo.get("future_evidence_penalty", 0.0))
        overcite_pen = float(cinfo.get("overcite_penalty", 0.0))

        penalty = (
            W_LEN * len_pen[i]
            + miss_close_pen
            + invalid_link_pen
            + hallucinated_pen
            + future_pen
            + overcite_pen
        )

        rewards.append(float(base - penalty))

        schema_vals.append(schema)
        citation_vals.append(citation)
        supporting_vals.append(supporting)
        missing_evidence_vals.append(missing_evidence)
        missing_evidence_used_rates.append(float(minfo.get("missing_evidence_used", 0.0)))
        missing_limitation_coverages.append(float(minfo.get("missing_limitation_coverage", 0.0)))
        answer_vals.append(answer_score)
        freeform_used_flags.append(freeform_used)
        freeform_format_vals.append(freeform_format)
        citation_valid_rates.append(float(cinfo.get("citation_valid_rate", 0.0)))
        evidence_link_valid_rates.append(float(cinfo.get("evidence_link_valid_rate", 0.0)))
        supporting_fact_valid_rates.append(float(sinfo.get("supporting_fact_valid_rate", 0.0)))
        key_recalls.append(float(cinfo.get("key_evidence_recall", 0.0)))
        source_covs.append(float(cinfo.get("source_type_coverage", 0.0)))
        invalid_link_pens.append(invalid_link_pen)
        hallucinated_pens.append(hallucinated_pen)
        future_pens.append(future_pen)
        overcite_pens.append(overcite_pen)
        citation_counts.append(float(cinfo.get("citation_count", 0.0)))

    info: Dict[str, float] = {}
    if return_info:
        info = {
            "format_mean": float(sum(fmt) / max(1, n)),
            "rougeL_mean": float(sum(r_rL[i] for i in idx) / max(1, len(idx))) if idx else 0.0,
            "cosine_mean": float(sum(r_cos[i] for i in idx) / max(1, len(idx))) if idx else 0.0,
            "semantic_mean": float(sum(0.45 * r_rL[i] + 0.55 * r_cos[i] for i in range(n)) / max(1, n)),
            "schema_mean": float(sum(schema_vals) / max(1, n)),
            "citation_mean": float(sum(citation_vals) / max(1, n)),
            "supporting_fact_mean": float(sum(supporting_vals) / max(1, n)),
            "missing_evidence_handling_mean": float(sum(missing_evidence_vals) / max(1, n)),
            "missing_evidence_used_rate": float(sum(missing_evidence_used_rates) / max(1, n)),
            "missing_limitation_coverage": float(sum(missing_limitation_coverages) / max(1, n)),
            "answer_score_mean": float(sum(answer_vals) / max(1, n)),
            "structured_rate": float(sum(1.0 for x in structured_flags if x) / max(1, n)),
            "freeform_used_rate": float(sum(1.0 for x in freeform_used_flags if x) / max(1, n)),
            "freeform_format_mean": float(sum(freeform_format_vals) / max(1, n)),
            "citation_valid_rate": float(sum(citation_valid_rates) / max(1, n)),
            "evidence_link_valid_rate": float(sum(evidence_link_valid_rates) / max(1, n)),
            "supporting_fact_valid_rate": float(sum(supporting_fact_valid_rates) / max(1, n)),
            "key_evidence_recall": float(sum(key_recalls) / max(1, n)),
            "source_type_coverage": float(sum(source_covs) / max(1, n)),
            "citation_count_mean": float(sum(citation_counts) / max(1, n)),
            "invalid_link_penalty_mean": float(sum(invalid_link_pens) / max(1, n)),
            "hallucinated_evidence_penalty_mean": float(sum(hallucinated_pens) / max(1, n)),
            "future_evidence_penalty_mean": float(sum(future_pens) / max(1, n)),
            "overcite_penalty_mean": float(sum(overcite_pens) / max(1, n)),
            "pred_len_mean": float(sum(pred_lens) / max(1, n)),
            "think_close_rate": float(sum(has_think_close) / max(1, n)),
            "comp_len_mean": float(sum(comp_lens) / max(1, n)),
            "len_pen_mean": float(sum(len_pen) / max(1, n)),
            "miss_think_close_rate": float(sum(1.0 for x in has_think_close if not x) / max(1, n)),
        }

    return rewards, info


# ===== SmartCity StageD strict always-think reward wrapper =====
# This wrapper is intentionally append-only. It preserves the original evidence
# reward, then caps rewards when the raw model completion does not satisfy the
# target think-tag format.
#
# IMPORTANT:
# - This wrapper assumes completions are RAW model-generated text.
# - If <think> is prefilled outside the model, do NOT count that as generated.
# - For self-generated think training, do not prefill <think>; put the format
#   requirement in the prompt and let the model generate the tags.
_smartcity_stageD_compute_rewards_orig = compute_rewards


def _smartcity_env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on", "y"}


def _smartcity_reason_mode(meta: ExampleMeta) -> str:
    return ((getattr(meta, "reasoning_mode", "") or "auto").strip().lower().replace("-", "_"))


def _smartcity_extract_think_block(text: str):
    t = _strip_end_tokens(text or "")
    m = re.search(r"(?is)<think>(.*?)</think>", t)
    if not m:
        return "", "", False
    body = (m.group(1) or "").strip()
    after = t[m.end():].strip()
    return body, after, True


def _smartcity_starts_with_think(text: str) -> bool:
    return bool(re.match(r"(?is)^\s*<think>", _strip_end_tokens(text or "")))


def _smartcity_starts_with_empty_think(text: str) -> bool:
    return bool(re.match(r"(?is)^\s*<think>\s*</think>\s*\S", _strip_end_tokens(text or "")))


def _smartcity_reason_format_info(text: str, meta: ExampleMeta) -> Dict[str, Any]:
    t = _strip_end_tokens(text or "")
    mode = _smartcity_reason_mode(meta)

    open_count = len(re.findall(r"(?is)<think>", t))
    close_count = len(re.findall(r"(?is)</think>", t))
    has_open = open_count > 0
    has_close = close_count > 0
    starts_open = _smartcity_starts_with_think(t)

    body, after, paired = _smartcity_extract_think_block(t)
    body_units = _proxy_len_units(body)
    after_units = _proxy_len_units(after)

    # For think samples, the model must generate the whole pair itself:
    #   <think> non-empty planning </think> final answer
    if mode in {"think", "thinking", "cot"}:
        ok = (
            starts_open
            and paired
            and body_units >= int(os.environ.get("SMARTCITY_REWARD_MIN_THINK_UNITS", "8"))
            and after_units >= int(os.environ.get("SMARTCITY_REWARD_MIN_PUBLIC_UNITS", "8"))
        )
        return {
            "mode": mode,
            "ok": bool(ok),
            "has_open": has_open,
            "has_close": has_close,
            "starts_open": starts_open,
            "open_count": open_count,
            "close_count": close_count,
            "think_body_units": body_units,
            "public_units": after_units,
            "empty_think": False,
        }

    # For no_think samples, the model must generate an empty block itself:
    #   <think></think> final answer
    if mode in {"no_think", "nothink", "no", "none", "instant"}:
        starts_empty = _smartcity_starts_with_empty_think(t)
        extra_open_after_first = False
        extra_close_after_first = False
        if starts_empty:
            t_after = re.sub(r"(?is)^\s*<think>\s*</think>\s*", "", t, count=1)
            extra_open_after_first = bool(re.search(r"(?is)<think>", t_after))
            extra_close_after_first = bool(re.search(r"(?is)</think>", t_after))

        ok = (
            starts_empty
            and not extra_open_after_first
            and not extra_close_after_first
            and _proxy_len_units(extract_answer(t)) >= int(os.environ.get("SMARTCITY_REWARD_MIN_PUBLIC_UNITS", "8"))
        )
        return {
            "mode": mode,
            "ok": bool(ok),
            "has_open": has_open,
            "has_close": has_close,
            "starts_open": starts_open,
            "open_count": open_count,
            "close_count": close_count,
            "think_body_units": body_units,
            "public_units": after_units,
            "empty_think": bool(starts_empty),
            "extra_open_after_empty": bool(extra_open_after_first),
            "extra_close_after_empty": bool(extra_close_after_first),
        }

    # Auto reason samples: require at least a valid pair.
    ok = starts_open and paired and after_units >= int(os.environ.get("SMARTCITY_REWARD_MIN_PUBLIC_UNITS", "8"))
    return {
        "mode": mode,
        "ok": bool(ok),
        "has_open": has_open,
        "has_close": has_close,
        "starts_open": starts_open,
        "open_count": open_count,
        "close_count": close_count,
        "think_body_units": body_units,
        "public_units": after_units,
        "empty_think": False,
    }


def compute_rewards(
    completions: List[str],
    metas_rep: List[ExampleMeta],
    *,
    return_info: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    rewards, info = _smartcity_stageD_compute_rewards_orig(
        completions,
        metas_rep,
        return_info=True,
    )

    if not _smartcity_env_true("SMARTCITY_REWARD_STRICT_ALWAYS_THINK_TAGS", "0"):
        return (rewards, info) if return_info else (rewards, {})

    metas = [ExampleMeta.from_any(m) for m in metas_rep]

    bad_think_cap = float(os.environ.get("SMARTCITY_REWARD_BAD_THINK_CAP", "0.03"))
    bad_nothink_cap = float(os.environ.get("SMARTCITY_REWARD_BAD_NOTHINK_CAP", "0.03"))
    no_citation_cap = float(os.environ.get("SMARTCITY_REWARD_NO_CITATION_CAP", "0.10"))
    long_cap = float(os.environ.get("SMARTCITY_REWARD_LONG_CAP", "0.20"))
    answer_zero_cap = float(os.environ.get("SMARTCITY_REWARD_ANSWER_ZERO_CAP", "0.35"))
    long_units = int(os.environ.get("SMARTCITY_REWARD_STRICT_LONG_UNITS", "900"))

    fmt_infos = [_smartcity_reason_format_info(t, m) for t, m in zip(completions, metas)]

    capped = []
    caps = []
    fmt_ok = []
    think_open = []
    think_close = []
    starts_open = []
    empty_ok = []
    no_cite_flags = []
    long_flags = []

    # info from the original reward is aggregate-only. For answer-zero cap, use
    # the original aggregate as a weak global guard; the strict format cap is the
    # primary mechanism.
    orig_answer_score_mean = float(info.get("answer_score_mean", 0.0) or 0.0)

    for r, text, meta, fi in zip(rewards, completions, metas, fmt_infos):
        mode = _smartcity_reason_mode(meta)
        ans = extract_answer(text or "")
        cited = _extract_freeform_evidence_ids(ans)
        L = _proxy_len_units(text or "")

        cap = 1.0

        if mode in {"think", "thinking", "cot"}:
            if not fi["ok"]:
                cap = min(cap, bad_think_cap)

        elif mode in {"no_think", "nothink", "no", "none", "instant"}:
            if not fi["ok"]:
                cap = min(cap, bad_nothink_cap)

        else:
            if not fi["ok"]:
                cap = min(cap, bad_think_cap)

        if not cited:
            cap = min(cap, no_citation_cap)

        if L > long_units:
            cap = min(cap, long_cap)

        if orig_answer_score_mean <= 0.001:
            # Do not let citation alone keep scores around 0.5 when decision
            # correctness is zero. Format caps above are stricter when triggered.
            cap = min(cap, answer_zero_cap)

        rr = float(r)
        capped.append(float(max(-1.0, min(rr, cap))))
        caps.append(float(cap))
        fmt_ok.append(bool(fi["ok"]))
        think_open.append(bool(fi["has_open"]))
        think_close.append(bool(fi["has_close"]))
        starts_open.append(bool(fi["starts_open"]))
        empty_ok.append(bool(fi.get("empty_think", False)))
        no_cite_flags.append(len(cited) == 0)
        long_flags.append(L > long_units)

    n = max(1, len(capped))
    info = dict(info or {})
    info.update({
        "strict_always_think_enabled": 1.0,
        "strict_format_ok_rate": float(sum(fmt_ok) / n),
        "strict_think_open_rate": float(sum(think_open) / n),
        "strict_think_close_rate": float(sum(think_close) / n),
        "strict_starts_with_think_rate": float(sum(starts_open) / n),
        "strict_empty_think_rate": float(sum(empty_ok) / n),
        "strict_no_citation_rate": float(sum(no_cite_flags) / n),
        "strict_long_rate": float(sum(long_flags) / n),
        "strict_reward_cap_mean": float(sum(caps) / n),
        "strict_reward_mean_after_cap": float(sum(capped) / n),
    })

    return (capped, info) if return_info else (capped, {})
# ===== End SmartCity StageD strict always-think reward wrapper =====
