from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import torch

@dataclass
class ExampleMeta:
    task: str
    target: str
    answer: str

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
    if not t:
        return 0
    return len(_TOKEN_PROXY_RE.findall(t))


def extract_answer(text: str) -> str:
    """
    Qwen3-aligned:
    - Take content AFTER the last </think> as answer.
    - If no </think>, use whole text.
    - Strip trailing control tokens like <|im_end|>, <|endoftext|>.
    - Do NOT require <answer> tags.
    """
    t = _strip_end_tokens(text)

    low = t.lower()
    key = "</think>"
    if key in low:
        j = low.rfind(key)
        t = t[j + len(key):].strip()

    t = t.replace("<think>", "").replace("</think>", "").strip()
    return t


def soft_format_reward(text: str) -> float:
    t = text or ""
    tags = ["<think>", "</think>"]
    score = 0.0
    for tg in tags:
        if tg in t:
            score += 1.0 / len(tags)
    return float(score)


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

def _compute_cos01(preds: List[str], refs: List[str],
                   *, model_name: str, device: str, batch_size: int) -> List[float]:
    st = _get_st_model(model_name, device=device)
    emb_p = st.encode(preds, batch_size=batch_size, convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=False)
    emb_r = st.encode(refs,  batch_size=batch_size, convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=False)
    cos = (emb_p * emb_r).sum(dim=1).clamp(min=-1.0, max=1.0)
    cos01 = (cos + 1.0) * 0.5
    return [float(x) for x in cos01.detach().cpu().tolist()]


def compute_rewards(
    completions: List[str],
    metas_rep: List[ExampleMeta],
    *,
    return_info: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Base reward:
        reward = 0.05 * format + 0.45 * Rouge-L + 0.50 * cosine
    Score Rouge/Cos only on extracted answer (after LAST </think>).

    NEW (to suppress len_stop_mean inflation):
    - small length penalty on the WHOLE completion length (proxy units)
    - small penalty if </think> is missing (often correlates with runaway generations)
    """
    # ---- base weights ----
    w_format = 0.05
    w_rL = 0.45
    w_cos = 0.50

    FREE_LEN = 650
    W_LEN = 0.08
    W_MISS_THINK_CLOSE = 0.03

    metric_tokenize = "space"
    metric_batch_size = 16
    cos_model = "sentence-transformers/all-MiniLM-L6-v2"
    metric_device = "cpu" 

    n = len(completions)
    assert len(metas_rep) == n

    fmt = [soft_format_reward(t) for t in completions]

    preds, refs = [], []
    pred_lens = []
    comp_lens = []
    has_think_close = []

    for text, meta in zip(completions, metas_rep):
        text0 = text or ""
        has_close = bool(_THINK_RE_CLOSE.search(text0))
        has_think_close.append(has_close)

        ans = extract_answer(text0)
        preds.append(ans)
        pred_lens.append(len(ans.strip()))

        comp_lens.append(_proxy_len_units(text0))

        gt = (meta.answer or meta.target or "").strip()
        refs.append(gt)

    idx = [i for i, r in enumerate(refs) if r.strip() != ""]
    r_rL  = [0.0] * n
    r_cos = [0.0] * n

    if idx:
        sub_preds = [preds[i] for i in idx]
        sub_refs  = [refs[i]  for i in idx]

        sub_preds_tok = [_tokenize_for_metrics(x, metric_tokenize) for x in sub_preds]
        sub_refs_tok  = [_tokenize_for_metrics(x, metric_tokenize) for x in sub_refs]
        rL_sub = _compute_rougeL_f(sub_preds_tok, sub_refs_tok)

        cos_sub = _compute_cos01(sub_preds, sub_refs, model_name=cos_model,
                                 device=metric_device, batch_size=metric_batch_size)

        for j, i in enumerate(idx):
            r_rL[i]  = float(rL_sub[j])
            r_cos[i] = float(cos_sub[j])

    len_pen = []
    for L in comp_lens:
        over = max(0.0, (float(L) - float(FREE_LEN)) / max(1.0, float(FREE_LEN)))
        over = min(1.0, over)  # clamp [0,1]
        len_pen.append(over)

    rewards: List[float] = []
    for i in range(n):
        base = w_format * fmt[i] + w_rL * r_rL[i] + w_cos * r_cos[i]
        penalty = W_LEN * len_pen[i] + (W_MISS_THINK_CLOSE * (0.0 if has_think_close[i] else 1.0))
        r = base - penalty
        rewards.append(float(r))

    info: Dict[str, float] = {}
    if return_info:
        info = {
            "format_mean": float(sum(fmt) / max(1, n)),
            "rougeL_mean": float(sum(r_rL[i] for i in idx) / max(1, len(idx))) if idx else 0.0,
            "cosine_mean": float(sum(r_cos[i] for i in idx) / max(1, len(idx))) if idx else 0.0,
            "pred_len_mean": float(sum(pred_lens) / max(1, n)),
            "think_close_rate": float(sum(has_think_close) / max(1, n)),

            "comp_len_mean": float(sum(comp_lens) / max(1, n)),
            "len_pen_mean": float(sum(len_pen) / max(1, n)),
            "miss_think_close_rate": float(sum(1.0 for x in has_think_close if not x) / max(1, n)),
        }

    return rewards, info
