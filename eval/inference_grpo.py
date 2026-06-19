#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SMARTCITY_REWARD_FREEFORM_EVIDENCE", "1")
os.environ.setdefault("SMARTCITY_REWARD_USE_SEMANTIC", "0")
os.environ.setdefault("SMARTCITY_REASON_KEEP_FREEFORM_ANSWER", "1")
os.environ.setdefault("SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

from model.builder import build_smartcity_llm
from dataprocessing import (
    load_jsonl_dataset,
    ensure_smartcity_features,
    DataCollatorSmartCity,
)
from train_grpo import (
    load_tokenizer_no_modify,
    build_stop_token_ids,
    inject_soft_think_in_user,
    inject_soft_no_think_in_user,
    infer_reasoning_mode_from_example,
)
from model.reinforcement_learning.rewards import (
    ExampleMeta,
    compute_rewards,
    extract_answer,
)

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

EVID_RE = re.compile(r"\bE_[A-Za-z0-9_]+\b")


def qwen_reason_template_prefix_enabled() -> bool:
    return os.environ.get("SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX", "0").strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_json_maybe(x: Any):
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


def compact(x: Any, n: int = 500) -> str:
    x = str(x or "")
    x = re.sub(r"\s+", " ", x).strip()
    return x if len(x) <= n else x[:n] + " ...[truncated]"


def unique_eids(text: str) -> List[str]:
    out, seen = [], set()
    for m in EVID_RE.finditer(text or ""):
        eid = m.group(0)
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def read_raw_reason_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSON at {p}:{lineno}: {e}") from e
            if str(row.get("task", "") or "").strip().lower() == "reason":
                rows.append(row)
    if not rows:
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def merge_raw_gold_for_reward(norm_ex: Dict[str, Any], raw_ex: Dict[str, Any]) -> Dict[str, Any]:
    y = dict(norm_ex)
    preserve_keys = [
        "gold_answer_json",
        "answer",
        "target",
        "reward_meta",
        "evidence_catalog",
        "key_evidence_ids",
        "key_source_types",
        "gold_decision",
        "reasoning_mode",
        "task",
        "sample_id",
    ]
    for k in preserve_keys:
        if k in raw_ex:
            y[k] = raw_ex[k]
    return y


def make_reward_meta_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    y = dict(ex)

    gold = parse_json_maybe(y.get("gold_answer_json", ""))

    if not isinstance(gold, dict):
        rm = parse_json_maybe(y.get("reward_meta", ""))
        if isinstance(rm, dict):
            for kk in ("gold_answer_json", "answer_json", "gold_json"):
                cand = parse_json_maybe(rm.get(kk, ""))
                if isinstance(cand, dict):
                    gold = cand
                    break

    if isinstance(gold, dict):
        y["answer"] = json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
        y["target"] = ""
        return y

    ans = parse_json_maybe(y.get("answer", ""))
    if isinstance(ans, dict):
        y["answer"] = json.dumps(ans, ensure_ascii=False, separators=(",", ":"))
        y["target"] = ""
        return y

    print("[debug] keys:", sorted(y.keys()))
    print("[debug] answer:", str(y.get("answer", ""))[:500])
    print("[debug] gold_answer_json:", str(y.get("gold_answer_json", ""))[:500])
    print("[debug] reward_meta:", str(y.get("reward_meta", ""))[:500])
    raise RuntimeError("No JSON gold found for reward.")


def evidence_catalog_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    out = []
    for line in str(prompt or "").splitlines():
        line = line.strip()
        if not line.startswith("- E_"):
            continue
        raw = line[2:].strip()
        parts = [p.strip() for p in raw.split("|")]
        if not parts:
            continue
        eid = parts[0]
        if not eid.startswith("E_"):
            continue
        source_type = parts[1] if len(parts) > 1 else ""
        status = parts[2] if len(parts) > 2 else ""
        role = parts[3] if len(parts) > 3 else ""
        out.append({
            "id": eid,
            "source_type": source_type,
            "status": status,
            "role": role,
            "present": status.lower() == "present",
            "should_use_as_support": role.lower() == "support",
        })
    return out


def evidence_maps(ex: Dict[str, Any]):
    raw = ex.get("evidence_catalog", "")
    cat = parse_json_maybe(raw)
    if not isinstance(cat, list) or not cat:
        cat = evidence_catalog_from_prompt(ex.get("prompt", ""))

    valid_ids, support_ids, missing_ids = set(), set(), set()
    type_by_id = {}

    for item in cat:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id", "") or "").strip()
        if not eid:
            continue

        status = str(item.get("status", "") or "").lower()
        role = str(item.get("role", "") or "").lower()
        source_type = str(item.get("source_type", "") or item.get("type", "") or "").strip()

        present = bool(item.get("present", False)) or status == "present"
        should_support = bool(item.get("should_use_as_support", False)) or role == "support"

        valid_ids.add(eid)
        type_by_id[eid] = source_type

        if present and should_support:
            support_ids.add(eid)
        if (not present) or status == "missing" or "do_not_use" in role:
            missing_ids.add(eid)

    return valid_ids, support_ids, missing_ids, type_by_id


def resolve_ckpt_dir(path_like: str) -> Path:
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(p)

    names = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.bin",
    ]
    if p.is_dir() and any((p / n).is_file() for n in names):
        return p

    children = []
    if p.is_dir():
        for d in p.glob("checkpoint-*"):
            if d.is_dir() and any((d / n).is_file() for n in names):
                m = re.search(r"checkpoint-(\d+)$", d.name)
                step = int(m.group(1)) if m else -1
                children.append((step, d))
    if children:
        return sorted(children, reverse=True)[0][1]

    raise FileNotFoundError(f"No loadable checkpoint found under {p}")


def load_state_dict_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.name.endswith(".safetensors"):
        if safe_load_file is None:
            raise RuntimeError("safetensors is required but not importable")
        return safe_load_file(str(path), device="cpu")

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a state dict: {type(obj)}")
    return obj


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(str(k).startswith("module.") for k in sd.keys()):
        return {str(k)[len("module."):]: v for k, v in sd.items()}
    return sd


def copy_state_dict_flexible(model, sd: Dict[str, torch.Tensor], label: str, verbose: bool = True):
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(f"{label}: empty or invalid state_dict")

    sd = strip_module_prefix(sd)

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if verbose:
            print(f"[load:{label}] normal load_state_dict", flush=True)
            print(f"[load:{label}] tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing[:10]:
                print(f"[load:{label}][missing:first10]", list(missing[:10]), flush=True)
            if unexpected[:10]:
                print(f"[load:{label}][unexpected:first10]", list(unexpected[:10]), flush=True)

        if len(unexpected) < len(sd):
            return
    except Exception as e:
        print(f"[load:{label}][warn] normal load_state_dict failed: {repr(e)}", flush=True)

    named_tensors = {}
    named_tensors.update(dict(model.named_parameters()))
    named_tensors.update(dict(model.named_buffers()))

    def candidates(k: str):
        outs = [k]
        for pref in ("module.", "_forward_module.", "model."):
            if k.startswith(pref):
                outs.append(k[len(pref):])
        outs.append("module." + k)
        return list(dict.fromkeys(outs))

    loaded, skipped, mismatch = 0, [], []
    with torch.no_grad():
        for k, v in sd.items():
            if not torch.is_tensor(v):
                skipped.append(k)
                continue
            hit = None
            for kk in candidates(str(k)):
                if kk in named_tensors:
                    hit = kk
                    break
            if hit is None:
                skipped.append(k)
                continue
            tgt = named_tensors[hit]
            if tuple(tgt.shape) != tuple(v.shape):
                mismatch.append((k, tuple(v.shape), tuple(tgt.shape)))
                continue
            tgt.copy_(v.to(device=tgt.device, dtype=tgt.dtype))
            loaded += 1

    if verbose:
        print(f"[load:{label}] flexible copy loaded={loaded} skipped={len(skipped)} mismatch={len(mismatch)}", flush=True)
        if skipped[:10]:
            print(f"[load:{label}][skipped:first10]", skipped[:10], flush=True)
        if mismatch[:10]:
            print(f"[load:{label}][mismatch:first10]", mismatch[:10], flush=True)

    if loaded == 0:
        raise RuntimeError(f"{label}: no tensors loaded")


def load_base_checkpoint_flexible(model, base_dir: Path):
    print(f"[load-base] {base_dir}", flush=True)

    sf = base_dir / "model.safetensors"
    if sf.is_file():
        sd = load_state_dict_file(sf)
        copy_state_dict_flexible(model, sd, "base_model_safetensors")
        return

    pt = base_dir / "pytorch_model.bin"
    if pt.is_file():
        sd = load_state_dict_file(pt)
        copy_state_dict_flexible(model, sd, "base_pytorch_bin")
        return

    if hasattr(model, "load_full_state_dict"):
        model.load_full_state_dict(str(base_dir), cast_to_target_dtype=True, verbose=True)
        print(f"[load-base] loaded by model.load_full_state_dict: {base_dir}", flush=True)
        return

    raise FileNotFoundError(f"Cannot load base checkpoint from {base_dir}")


def load_base_then_adapter(model, base_dir: str, ckpt_dir: str) -> Path:
    base = resolve_ckpt_dir(base_dir)
    ckpt = resolve_ckpt_dir(ckpt_dir)

    print(f"[load] base_dir={base}", flush=True)
    print(f"[load] ckpt_dir={ckpt}", flush=True)

    load_base_checkpoint_flexible(model, base)

    if (ckpt / "adapter_model.bin").is_file():
        print("[load] adapter format: adapter_model.bin", flush=True)
        sd = load_state_dict_file(ckpt / "adapter_model.bin")
        copy_state_dict_flexible(model, sd, "adapter_model_bin")
        return ckpt

    if (ckpt / "model.safetensors").is_file():
        print("[load] checkpoint format: model.safetensors", flush=True)
        sd = load_state_dict_file(ckpt / "model.safetensors")
        copy_state_dict_flexible(model, sd, "ckpt_model_safetensors")
        return ckpt

    if (ckpt / "pytorch_model.bin").is_file():
        print("[load] checkpoint format: pytorch_model.bin", flush=True)
        sd = load_state_dict_file(ckpt / "pytorch_model.bin")
        copy_state_dict_flexible(model, sd, "ckpt_pytorch_bin")
        return ckpt

    if (ckpt / "pytorch_model.bin.index.json").is_file() or (ckpt / "model.safetensors.index.json").is_file():
        print("[load] checkpoint format: sharded full checkpoint", flush=True)
        model.load_full_state_dict(str(ckpt), cast_to_target_dtype=True, verbose=True)
        return ckpt

    raise FileNotFoundError(f"Unsupported checkpoint format: {ckpt}")


def build_model(decoder: str, encoder: str):
    mix_cfg_overrides = dict(
        num_experts=20,
        lora_r=8,
        lora_alpha=16.0,
        enable_attention=False,
        enable_ffn=True,
        share_router_for_wi=True,
        num_tasks=3,
        ensure_nonzero_gating=False,
        num_universal=4,
        num_pred=8,
        num_reason=8,
        rank_universal_mul=2.0,
        target_modules_override={
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [],
            "k": [],
            "v": [],
            "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        },
    )

    return build_smartcity_llm(
        decoder_name=decoder,
        encoder_name=encoder,
        alpha_understand=0.2,
        beta_prediction=1.0,
        beta_reason=1.0,
        lambda_rat=0.2,
        num_chunk_tokens=4,
        adapter_heads=4,
        encoder_max_length=512,
        imagebind_variant="google/siglip-so400m-patch14-384",
        num_image_tokens=32,
        group_layers=2,
        group_heads=8,
        group_tau=1.0,
        open_roberta=False,
        open_vpma=False,
        open_imagebind=False,
        open_image_projector=False,
        freeze_decoder=True,
        use_mixlora=True,
        mixlora_cfg_overrides=mix_cfg_overrides,
    )


def parse_indices(s: str) -> List[int]:
    out = []
    for x in re.split(r"[,\s]+", str(s or "").strip()):
        if not x:
            continue
        out.append(int(x))
    return out


def select_indices(ds, n_per_mode: int) -> List[int]:
    hits = {"think": [], "no_think": []}
    for idx in range(len(ds)):
        row = dict(ds[idx])
        mode = str(row.get("reasoning_mode", "") or "").lower().replace("-", "_")
        if mode in hits and len(hits[mode]) < n_per_mode:
            hits[mode].append(idx)
        if all(len(v) >= n_per_mode for v in hits.values()):
            break

    print("[selected]", hits, flush=True)
    return hits["think"] + hits["no_think"]


def normalize_mode(ex: Dict[str, Any]) -> str:
    mode = infer_reasoning_mode_from_example(ex, default_mode="auto")
    mode = str(mode or "").lower().replace("-", "_")
    if mode not in {"think", "no_think"}:
        mode = str(ex.get("reasoning_mode", "") or "").lower().replace("-", "_")
    if mode not in {"think", "no_think"}:
        mode = "think"
    return mode


def clean_decoded(raw: str) -> str:
    for end_tok in ("<|im_end|>", "<|endoftext|>"):
        if end_tok in raw:
            raw = raw.split(end_tok, 1)[0].strip()
    return raw.strip()


def run_one_sample(
    *,
    model,
    tok,
    enc_tok,
    collator,
    ds,
    raw_rows: List[Dict[str, Any]],
    idx: int,
    device: torch.device,
    max_new_tokens: int,
    temp: float,
    top_p: float,
    top_k: int,
    stop_ids,
    strict_prompt_check: bool,
    force_think_prefix: bool,
    out_dir: Path,
    print_full: bool,
    force_nothink_empty_prefix: bool = False,
) -> Dict[str, Any]:
    idx = idx % len(ds)
    orig = dict(ds[idx])
    raw_orig = dict(raw_rows[idx % len(raw_rows)])
    reward_source = merge_raw_gold_for_reward(orig, raw_orig)

    mode = normalize_mode(orig)
    reward_ex = make_reward_meta_row(reward_source)

    ex = dict(orig)
    ex["answer"] = ""
    ex["target"] = ""
    ex["thinking"] = []
    ex["reasoning_mode"] = mode

    prompt = ex.get("prompt", "") or ""
    prompt_has_freeform = "qwen3_freeform_evidence_v1" in prompt
    prompt_has_old_json = (
        "single-line JSON object" in prompt
        or "The JSON must contain" in prompt
        or "public JSON" in prompt
    )

    if strict_prompt_check and prompt_has_old_json:
        raise RuntimeError(f"idx={idx}: prompt still contains old JSON instruction")

    if not qwen_reason_template_prefix_enabled():
        if mode == "think":
            ex["prompt"] = inject_soft_think_in_user(prompt)
        elif mode == "no_think":
            ex["prompt"] = inject_soft_no_think_in_user(prompt)
    else:
        # Qwen-prefix route: DataCollatorSmartCity puts <think> or empty think block
        # into suffix_ids / assistant context. Do not add legacy /think tokens.
        ex["prompt"] = prompt

    batch = collator([ex])
    batch = to_device(batch, device)

    valid_ids, support_ids, missing_ids, _ = evidence_maps(reward_ex)
    gold_obj = parse_json_maybe(reward_ex.get("answer", ""))
    gold_decision = reward_ex.get("gold_decision", "")
    if isinstance(gold_obj, dict) and not gold_decision:
        gold_decision = gold_obj.get("decision", "")

    print("\n" + "=" * 100, flush=True)
    print(f"[sample] idx={idx} sample_id={orig.get('sample_id','')} mode={mode} force_think={force_think_prefix} force_nothink_empty={force_nothink_empty_prefix}", flush=True)
    print(f"[gold_decision] {gold_decision}", flush=True)
    print(f"[prompt] freeform={prompt_has_freeform} old_json={prompt_has_old_json}", flush=True)
    print(f"[evidence] valid={len(valid_ids)} support={len(support_ids)} missing={len(missing_ids)}", flush=True)
    print(f"[shape] prefix={tuple(batch['prefix_ids'].shape)} suffix={tuple(batch['suffix_ids'].shape)}", flush=True)

    context_prefix_text = ""
    with torch.inference_mode():
        emb = model.decoder.get_input_embeddings()
        dev = emb.weight.device
        dtype = emb.weight.dtype

        chunk_embeds, chunk_mask, image_embeds, image_mask, _, _ = model.encode_modalities(
            chunk_txts_all=batch.get("chunk_txts", None),
            pixel_values=batch.get("pixel_values", None),
            n_images=batch.get("n_images", None),
            batch_size=1,
            device=dev,
            dtype=dtype,
        )

        print(f"[modalities] chunk={tuple(chunk_embeds.shape)} image={tuple(image_embeds.shape)}", flush=True)

        if force_think_prefix and mode == "think":
            context_prefix_text = "<think>\n"
        elif force_nothink_empty_prefix and mode == "no_think":
            context_prefix_text = "<think></think>\n\n"

        if context_prefix_text:
            prefix_ids = tok(context_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            batch["suffix_ids"] = torch.cat([prefix_ids, batch["suffix_ids"]], dim=1)
            batch["suffix_mask"] = torch.cat(
                [
                    torch.ones_like(prefix_ids, device=batch["suffix_mask"].device),
                    batch["suffix_mask"],
                ],
                dim=1,
            )

        completion_ids = model.generate_with_modalities_cached(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
            suffix_ids=batch["suffix_ids"],
            suffix_mask=batch["suffix_mask"],
            chunk_embeds=chunk_embeds,
            chunk_mask=chunk_mask,
            image_embeds=image_embeds,
            image_mask=image_mask,
            task_ids=batch.get("task_ids", None),
            max_new_tokens=max_new_tokens,
            temperature=temp,
            do_sample=(temp > 0.0),
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stop_token_ids=stop_ids,
        )

    raw_generated = tok.decode(completion_ids[0].detach().cpu().tolist(), skip_special_tokens=False)
    raw_generated = clean_decoded(raw_generated)

    # For force-think diagnostics, the model's effective output includes the prefixed <think>.
    raw_for_reward = context_prefix_text + raw_generated
    public_answer = extract_answer(raw_for_reward)

    cited_ids = unique_eids(public_answer)
    invalid_ids = [x for x in cited_ids if x not in valid_ids]
    missing_cited = [x for x in cited_ids if x in missing_ids]
    support_cited = [x for x in cited_ids if x in support_ids]
    key_ids = set(reward_ex.get("key_evidence_ids", []) or [])
    key_hits = [x for x in cited_ids if x in key_ids]

    rewards, info = compute_rewards([raw_for_reward], [ExampleMeta.from_any(reward_ex)], return_info=True)

    has_think_open = "<think>" in raw_for_reward
    has_think_close = "</think>" in raw_for_reward
    answer_nonempty = bool(public_answer.strip())
    json_like = public_answer.strip().startswith("{") or public_answer.strip().startswith("[")

    summary = {
        "ckpt_sample_index": idx,
        "sample_id": orig.get("sample_id", ""),
        "reasoning_mode": mode,
        "force_think_prefix": force_think_prefix,
        "force_nothink_empty_prefix": force_nothink_empty_prefix,
        "generated_extra_think_open": "<think>" in raw_generated,
        "generated_extra_think_close": "</think>" in raw_generated,
        "gold_decision": gold_decision,
        "has_think_open": has_think_open,
        "has_think_close": has_think_close,
        "answer_nonempty": answer_nonempty,
        "json_like_answer": json_like,
        "cited_count": len(cited_ids),
        "invalid_cited_count": len(invalid_ids),
        "missing_cited_count": len(missing_cited),
        "support_cited_count": len(support_cited),
        "key_hit_count": len(key_hits),
        "reward": float(rewards[0]),
        "freeform_used_rate": info.get("freeform_used_rate"),
        "freeform_format_mean": info.get("freeform_format_mean"),
        "answer_score_mean": info.get("answer_score_mean"),
        "citation_mean": info.get("citation_mean"),
        "citation_valid_rate": info.get("citation_valid_rate"),
        "key_evidence_recall": info.get("key_evidence_recall"),
        "source_type_coverage": info.get("source_type_coverage"),
        "citation_count_mean": info.get("citation_count_mean"),
        "pred_len_mean": info.get("pred_len_mean"),
    }

    print("[generated preview]", compact(raw_for_reward, 700), flush=True)
    print("[public preview]", compact(public_answer, 700), flush=True)
    print("[cited_ids]", cited_ids, flush=True)
    print("[invalid_ids]", invalid_ids, flush=True)
    print("[missing_cited]", missing_cited, flush=True)
    print("[reward]", json.dumps(summary, ensure_ascii=False), flush=True)

    if print_full:
        print("\n[generated raw]\n" + raw_for_reward, flush=True)
        print("\n[public answer]\n" + public_answer, flush=True)

    out = {
        "summary": summary,
        "raw": raw_for_reward,
        "raw_generated_only": raw_generated,
        "public_answer": public_answer,
        "cited_ids": cited_ids,
        "support_cited": support_cited,
        "key_hits": key_hits,
        "invalid_ids": invalid_ids,
        "missing_cited": missing_cited,
        "reward_info": info,
        "gold_answer": reward_ex.get("answer", ""),
        "prompt_tail": (ex.get("prompt", "") or "")[-2200:],
    }

    if force_think_prefix:
        suffix = "_force"
    elif force_nothink_empty_prefix:
        suffix = "_nothink_empty"
    else:
        suffix = ""
    out_file = out_dir / f"sample_{idx}{suffix}.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", out_file, flush=True)

    del batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def mean_or_none(vals: List[Any]):
    nums = [float(x) for x in vals if isinstance(x, (int, float))]
    return sum(nums) / len(nums) if nums else None


def aggregate_and_save(results: List[Dict[str, Any]], out_dir: Path, name: str):
    rows = [r["summary"] for r in results]
    agg = {
        "n": len(rows),
        "reward": mean_or_none([r.get("reward") for r in rows]),
        "answer_score_mean": mean_or_none([r.get("answer_score_mean") for r in rows]),
        "freeform_used_rate": mean_or_none([r.get("freeform_used_rate") for r in rows]),
        "freeform_format_mean": mean_or_none([r.get("freeform_format_mean") for r in rows]),
        "citation_mean": mean_or_none([r.get("citation_mean") for r in rows]),
        "citation_valid_rate": mean_or_none([r.get("citation_valid_rate") for r in rows]),
        "key_evidence_recall": mean_or_none([r.get("key_evidence_recall") for r in rows]),
        "source_type_coverage": mean_or_none([r.get("source_type_coverage") for r in rows]),
        "citation_count_mean": mean_or_none([r.get("citation_count_mean") for r in rows]),
        "pred_len_mean": mean_or_none([r.get("pred_len_mean") for r in rows]),
        "think_close_count": sum(1 for r in rows if r.get("has_think_close")),
        "think_open_count": sum(1 for r in rows if r.get("has_think_open")),
        "json_like_count": sum(1 for r in rows if r.get("json_like_answer")),
        "invalid_cited_total": sum(int(r.get("invalid_cited_count") or 0) for r in rows),
        "missing_cited_total": sum(int(r.get("missing_cited_count") or 0) for r in rows),
        "generated_extra_think_open_count": sum(1 for r in rows if r.get("generated_extra_think_open")),
        "generated_extra_think_close_count": sum(1 for r in rows if r.get("generated_extra_think_close")),
    }

    out = {
        "aggregate": agg,
        "rows": rows,
    }
    path = out_dir / f"{name}_aggregate.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100, flush=True)
    print(f"[aggregate:{name}]", json.dumps(agg, ensure_ascii=False, indent=2), flush=True)
    print("[aggregate saved]", path, flush=True)

    tsv = out_dir / f"{name}_summary.tsv"
    cols = [
        "ckpt_sample_index",
        "sample_id",
        "reasoning_mode",
        "force_think_prefix",
        "force_nothink_empty_prefix",
        "generated_extra_think_open",
        "generated_extra_think_close",
        "gold_decision",
        "reward",
        "answer_score_mean",
        "has_think_open",
        "has_think_close",
        "cited_count",
        "invalid_cited_count",
        "missing_cited_count",
        "key_evidence_recall",
        "source_type_coverage",
        "citation_count_mean",
        "pred_len_mean",
    ]
    with tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print("[summary tsv saved]", tsv, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.environ.get("RUN_DIR", "outputs/stageD_grpo_halfepoch_zero2_G4_new256"))
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", ""))
    ap.add_argument("--base_dir", default=os.environ.get("BASE_DIR", "output_stageC_warmup/full_2epoch"))
    ap.add_argument("--eval_file", default=os.environ.get("EVAL_FILE", "dataset_stageD/train_dataset_reas.jsonl"))
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", ""))
    ap.add_argument("--indices", default=os.environ.get("INDICES", ""))
    ap.add_argument("--n_per_mode", type=int, default=int(os.environ.get("N_PER_MODE", "4")))
    ap.add_argument("--maxlen", type=int, default=int(os.environ.get("MAXLEN", "2048")))
    ap.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    ap.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.0")))
    ap.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "20")))
    ap.add_argument("--force_think_one", action="store_true", default=os.environ.get("FORCE_THINK_ONE", "1").lower() in {"1", "true", "yes"})
    ap.add_argument("--force_think_all", action="store_true", default=os.environ.get("FORCE_THINK_ALL", "0").lower() in {"1", "true", "yes"})
    ap.add_argument("--force_think_only", action="store_true", default=os.environ.get("FORCE_THINK_ONLY", "0").lower() in {"1", "true", "yes"})
    ap.add_argument("--force_think_n", type=int, default=int(os.environ.get("FORCE_THINK_N", "5")))
    ap.add_argument("--force_think_indices", default=os.environ.get("FORCE_THINK_INDICES", ""))
    ap.add_argument("--force_nothink_empty_all", action="store_true", default=os.environ.get("FORCE_NOTHINK_EMPTY_ALL", "0").lower() in {"1", "true", "yes"})
    ap.add_argument("--force_nothink_empty_only", action="store_true", default=os.environ.get("FORCE_NOTHINK_EMPTY_ONLY", "0").lower() in {"1", "true", "yes"})
    ap.add_argument("--force_nothink_empty_n", type=int, default=int(os.environ.get("FORCE_NOTHINK_EMPTY_N", "5")))
    ap.add_argument("--force_nothink_empty_indices", default=os.environ.get("FORCE_NOTHINK_EMPTY_INDICES", ""))
    ap.add_argument("--strict_prompt_check", action="store_true", default=os.environ.get("STRICT_PROMPT_CHECK", "1").lower() in {"1", "true", "yes"})
    ap.add_argument("--print_full", action="store_true", default=os.environ.get("PRINT_FULL", "0").lower() in {"1", "true", "yes"})
    ap.add_argument("--decoder", default=os.environ.get("DECODER", "/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"))
    ap.add_argument("--encoder", default=os.environ.get("ENCODER", "/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoint-1600"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"diag_{ckpt.name}_multi_once"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, flush=True)
    print("[torch]", torch.__version__, "cuda_build=", torch.version.cuda, flush=True)
    print("[run_dir]", run_dir, flush=True)
    print("[ckpt]", ckpt, flush=True)
    print("[base_dir]", args.base_dir, flush=True)
    print("[eval_file]", args.eval_file, flush=True)
    print("[out_dir]", out_dir, flush=True)

    if device.type != "cuda":
        raise RuntimeError("CUDA is not available.")

    torch.set_grad_enabled(False)

    ckpt_resolved = resolve_ckpt_dir(str(ckpt))

    print("\n[1/5] load tokenizer", flush=True)
    tok = load_tokenizer_no_modify(
        ckpt_dir=str(ckpt_resolved),
        fallback_name=args.decoder,
        padding_token="<|pad|>",
        local_files_only=True,
    )
    enc_tok = AutoTokenizer.from_pretrained(
        args.encoder,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    stop_ids = build_stop_token_ids(tok)
    print(f"[tok] pad={tok.pad_token}({tok.pad_token_id}) eos={tok.eos_token}({tok.eos_token_id}) len={len(tok)}", flush=True)
    print("[stop_ids]", stop_ids, flush=True)

    print("\n[2/5] build model", flush=True)
    model = build_model(args.decoder, args.encoder)

    print("\n[3/5] load base once, then adapter once", flush=True)
    ckpt_resolved = load_base_then_adapter(model, args.base_dir, str(ckpt_resolved))

    model.to(device)
    model.eval()
    try:
        if hasattr(model.decoder, "gradient_checkpointing_disable"):
            model.decoder.gradient_checkpointing_disable()
    except Exception:
        pass

    print("\n[4/5] load dataset once", flush=True)
    raw_rows = read_raw_reason_rows(args.eval_file)
    ds = load_jsonl_dataset(args.eval_file, split_name="validation", ensure_markers=True)
    ds = ensure_smartcity_features(ds, num_proc=1, ensure_markers=False)
    ds = ds.filter(lambda ex: str(ex.get("task", "") or "").lower() == "reason")
    if len(ds) == 0:
        raise RuntimeError(f"No task=reason samples in {args.eval_file}")
    print(f"[dataset] normalized_reason_rows={len(ds)} raw_reason_rows={len(raw_rows)}", flush=True)

    collator = DataCollatorSmartCity(
        tok,
        chunk_tokenizer=enc_tok,
        chunk_size_tokens=510,
        chunk_overlap=0,
        join_fields_before_chunk=True,
        add_field_header=True,
        max_length=args.maxlen,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=4,
        image_token_factor=32,
        max_images_per_sample=1,
        image_size=384,
        add_eos_to_labels=True,
        wrap_reason_with_tags=True,
    )

    if args.indices.strip():
        indices = parse_indices(args.indices)
        print("[indices from env/arg]", indices, flush=True)
    else:
        indices = select_indices(ds, args.n_per_mode)
        print("[indices selected]", indices, flush=True)

    print("\n[5/5] infer samples with already-loaded model", flush=True)
    started = time.time()
    normal_results = []
    if args.force_think_only or args.force_nothink_empty_only:
        print("[normal] skipped because FORCE_THINK_ONLY=1 or FORCE_NOTHINK_EMPTY_ONLY=1", flush=True)
    else:
        for idx in indices:
            normal_results.append(
                run_one_sample(
                    model=model,
                    tok=tok,
                    enc_tok=enc_tok,
                    collator=collator,
                    ds=ds,
                    raw_rows=raw_rows,
                    idx=idx,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temp=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stop_ids=stop_ids,
                    strict_prompt_check=args.strict_prompt_check,
                    force_think_prefix=False,
                    out_dir=out_dir,
                    print_full=args.print_full,
                )
            )

        aggregate_and_save(normal_results, out_dir, "normal")

    # Multi-sample force-think diagnostic.
    # Modes:
    #   FORCE_THINK_INDICES="0,1,4,5,11"  -> force these explicit think indices
    #   FORCE_THINK_ALL=1 FORCE_THINK_N=5 -> force first N think indices from selected indices
    #   FORCE_THINK_ONLY=1                -> skip normal generation, useful for quick validation
    force_indices = []

    if args.force_think_indices.strip():
        force_indices = parse_indices(args.force_think_indices)

    elif args.force_think_all or args.force_think_only:
        for idx in indices:
            row = dict(ds[idx % len(ds)])
            if normalize_mode(row) == "think":
                force_indices.append(idx)
                if args.force_think_n > 0 and len(force_indices) >= args.force_think_n:
                    break

        # If the selected indices do not contain enough think samples, scan the dataset.
        if args.force_think_n <= 0 or len(force_indices) < args.force_think_n:
            seen = set(force_indices)
            for i in range(len(ds)):
                if i in seen:
                    continue
                row = dict(ds[i])
                if normalize_mode(row) == "think":
                    force_indices.append(i)
                    seen.add(i)
                    if args.force_think_n > 0 and len(force_indices) >= args.force_think_n:
                        break

    elif args.force_think_one:
        think_idx = None
        for idx in indices:
            row = dict(ds[idx % len(ds)])
            if normalize_mode(row) == "think":
                think_idx = idx
                break
        if think_idx is None:
            for i in range(len(ds)):
                row = dict(ds[i])
                if normalize_mode(row) == "think":
                    think_idx = i
                    break
        if think_idx is not None:
            force_indices = [think_idx]

    if force_indices:
        force_results = []
        print(f"[force-think diagnostic] force_indices={force_indices}", flush=True)

        for think_idx in force_indices:
            row = dict(ds[think_idx % len(ds)])
            mode_i = normalize_mode(row)
            if mode_i != "think":
                print(f"[force-think diagnostic][skip] idx={think_idx} mode={mode_i}; expected think", flush=True)
                continue

            print("\n" + "=" * 100, flush=True)
            print(f"[force-think diagnostic] idx={think_idx}", flush=True)

            force_results.append(
                run_one_sample(
                    model=model,
                    tok=tok,
                    enc_tok=enc_tok,
                    collator=collator,
                    ds=ds,
                    raw_rows=raw_rows,
                    idx=think_idx,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temp=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stop_ids=stop_ids,
                    strict_prompt_check=args.strict_prompt_check,
                    force_think_prefix=True,
                    out_dir=out_dir,
                    print_full=args.print_full,
                )
            )

        if force_results:
            aggregate_and_save(force_results, out_dir, "force_think")
        else:
            print("[force-think diagnostic] no force-think results produced", flush=True)
    elif args.force_think_one or args.force_think_all or args.force_think_only:
        print("[force-think diagnostic] no think sample found", flush=True)


    # Multi-sample force no-think empty-block diagnostic.
    # For no_think samples, prefix the complete empty block:
    #   <think></think>\n\n
    # Then the model should generate only the public final answer.
    force_nothink_indices = []

    if args.force_nothink_empty_indices.strip():
        force_nothink_indices = parse_indices(args.force_nothink_empty_indices)

    elif args.force_nothink_empty_all or args.force_nothink_empty_only:
        for idx in indices:
            row = dict(ds[idx % len(ds)])
            if normalize_mode(row) == "no_think":
                force_nothink_indices.append(idx)
                if args.force_nothink_empty_n > 0 and len(force_nothink_indices) >= args.force_nothink_empty_n:
                    break

        if args.force_nothink_empty_n <= 0 or len(force_nothink_indices) < args.force_nothink_empty_n:
            seen = set(force_nothink_indices)
            for i in range(len(ds)):
                if i in seen:
                    continue
                row = dict(ds[i])
                if normalize_mode(row) == "no_think":
                    force_nothink_indices.append(i)
                    seen.add(i)
                    if args.force_nothink_empty_n > 0 and len(force_nothink_indices) >= args.force_nothink_empty_n:
                        break

    if force_nothink_indices:
        force_nothink_results = []
        print(f"[force-nothink-empty diagnostic] force_indices={force_nothink_indices}", flush=True)

        for no_idx in force_nothink_indices:
            row = dict(ds[no_idx % len(ds)])
            mode_i = normalize_mode(row)
            if mode_i != "no_think":
                print(f"[force-nothink-empty diagnostic][skip] idx={no_idx} mode={mode_i}; expected no_think", flush=True)
                continue

            print("\n" + "=" * 100, flush=True)
            print(f"[force-nothink-empty diagnostic] idx={no_idx}", flush=True)

            force_nothink_results.append(
                run_one_sample(
                    model=model,
                    tok=tok,
                    enc_tok=enc_tok,
                    collator=collator,
                    ds=ds,
                    raw_rows=raw_rows,
                    idx=no_idx,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temp=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stop_ids=stop_ids,
                    strict_prompt_check=args.strict_prompt_check,
                    force_think_prefix=False,
                    force_nothink_empty_prefix=True,
                    out_dir=out_dir,
                    print_full=args.print_full,
                )
            )

        if force_nothink_results:
            aggregate_and_save(force_nothink_results, out_dir, "force_nothink_empty")
        else:
            print("[force-nothink-empty diagnostic] no force no_think results produced", flush=True)
    elif args.force_nothink_empty_all or args.force_nothink_empty_only:
        print("[force-nothink-empty diagnostic] no no_think sample found", flush=True)


    elapsed = time.time() - started
    print("\n[done]", flush=True)
    print(f"[elapsed_sec] {elapsed:.1f}", flush=True)
    print(f"[outputs] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
