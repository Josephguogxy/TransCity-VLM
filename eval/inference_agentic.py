#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DECODER = "/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Qwen3-8B"
DEFAULT_ENCODER = "/apdcephfs_qy4/share_302593112/haorutan/smartCity/trafficPrediction/hf_models/Roberta-Large"


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json", exclude_none=True)
    if hasattr(x, "dict"):
        return x.dict(exclude_none=True)
    return str(x)


def compact_text(x: Any, max_chars: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(x or "")).strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 14)].rstrip() + " ...[truncated]"


def safe_label(x: Any, default: str = "sample") -> str:
    s = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(x or "").strip())
    return (s.strip("_") or default)[:160]


def parse_float_list(raw: str) -> Optional[List[float]]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(v) for v in obj]
    except Exception:
        pass
    vals = [x for x in re.split(r"[,;\s]+", s) if x]
    return [float(v) for v in vals] if vals else None


def read_questions(args: argparse.Namespace) -> List[str]:
    questions: List[str] = []
    for q in args.question or []:
        q = str(q or "").strip()
        if q:
            questions.append(q)

    if args.questions_file:
        p = Path(args.questions_file)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    questions.append(s)

    return questions


def read_agent_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"{p}:{lineno} is not a JSON object")
                rows.append(obj)
        return rows

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        return list(obj)
    raise TypeError(f"{p} must contain one JSON object, a list of objects, or JSONL rows")


def timestamp_out_dir(base: str) -> Path:
    if base:
        return Path(base)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "agentic_inference" / ts


def configure_prompt_env(stage: str, prompt_mode: str, qwen_prefix: str) -> str:
    mode = str(prompt_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "freeform" if stage == "grpo" else "json"
    if mode not in {"json", "freeform"}:
        raise ValueError("--prompt_mode must be one of: auto, json, freeform")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["SMARTCITY_REASON_ALWAYS_MARKERS"] = "1"

    qwen = str(qwen_prefix or "auto").strip().lower()
    if qwen == "auto":
        qwen_enabled = mode == "freeform"
    elif qwen in {"1", "true", "yes", "on", "y"}:
        qwen_enabled = True
    elif qwen in {"0", "false", "no", "off", "n"}:
        qwen_enabled = False
    else:
        raise ValueError("--qwen_prefix must be auto, true, or false")

    if mode == "freeform":
        os.environ["SMARTCITY_REASON_KEEP_FREEFORM_ANSWER"] = "1"
        os.environ["SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE"] = "1"
        os.environ["SMARTCITY_REASON_ALWAYS_THINK_TAGS"] = "1"
        os.environ["SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX"] = "1" if qwen_enabled else "0"
        os.environ["SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE"] = os.environ.get(
            "SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE",
            "qwen",
        )
        os.environ["SMARTCITY_GRPO_DISABLE_LEGACY_SOFT_THINK"] = "1" if qwen_enabled else "0"
        os.environ["SMARTCITY_REWARD_FREEFORM_EVIDENCE"] = "1"
        os.environ["SMARTCITY_REWARD_USE_SEMANTIC"] = "0"
    else:
        os.environ["SMARTCITY_REASON_KEEP_FREEFORM_ANSWER"] = "0"
        os.environ["SMARTCITY_REASON_PROMPT_FREEFORM_OVERRIDE"] = "0"
        os.environ["SMARTCITY_REASON_ALWAYS_THINK_TAGS"] = "0"
        os.environ["SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX"] = "1" if qwen_enabled else "0"
        os.environ["SMARTCITY_GRPO_DISABLE_LEGACY_SOFT_THINK"] = "1" if qwen_enabled else "0"
        os.environ.setdefault("SMARTCITY_REWARD_USE_SEMANTIC", "0")

    return mode


async def collect_agent_record(question: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from agentic_system.core.context import ExecutionContext
    from agentic_system.core.executor import Executor
    from agentic_system.core.logging import setup_logging
    from agentic_system.core.plan_builder import FeatureFlags, PlanBuilder
    from agentic_system.core.runner import QuestionRunner, utc_now_iso
    from agentic_system.schemas import UserRequest

    setup_logging(verbose=bool(args.verbose_agent))

    traffic_flow = parse_float_list(args.traffic_flow)
    history_average_flow = parse_float_list(args.history_average_flow)

    req = UserRequest(
        question=question,
        request_time_utc=args.request_time_utc or utc_now_iso(),
        lat=args.lat,
        lon=args.lon,
        local_start=args.local_start,
        local_end=args.local_end,
        sensor_id=args.sensor_id,
        lanes=args.lanes,
        direction=args.direction,
        freeway=args.freeway,
        traffic_flow=traffic_flow,
        history_average_flow=history_average_flow,
    )

    web_events = bool(args.with_web_events)
    if not args.with_context and web_events:
        web_events = False

    runner = QuestionRunner()
    agents = runner.build_agents(image_store=args.image_store)
    executor = Executor(agents=agents)

    try:
        flags = FeatureFlags(
            flow=bool(args.with_flow),
            satellite=bool(args.with_satellite),
            context=bool(args.with_context),
            web_events=web_events,
        )
        plan = PlanBuilder().build(req, flags=flags)
        ctx = ExecutionContext(request=req)
        ctx = await executor.execute(plan, ctx)

        record = ctx.outputs.get("dataset_record")
        if record is None:
            raise RuntimeError(f"agent did not produce dataset_record; outputs={list(ctx.outputs.keys())}")

        payload = jsonable(record)
        debug = {
            "request": jsonable(req),
            "plan": jsonable(plan),
            "outputs_keys": list(ctx.outputs.keys()),
        }
        if args.save_agent_outputs:
            debug["outputs"] = jsonable(ctx.outputs)
        return payload, debug
    finally:
        runner.close_agents(agents)


def enrich_agent_record(
    record: Dict[str, Any],
    *,
    question: str,
    index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = dict(record)
    row["task"] = "reason"

    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        sample_id = f"agentic_{int(time.time())}_{index:04d}"
    row["sample_id"] = sample_id

    row.setdefault("dataset_source", "agentic_runtime")
    row.setdefault("question_type", "agentic_reason")
    row.setdefault("question", question)
    row["reasoning_mode"] = args.reasoning_mode
    row["answer"] = ""
    row["target"] = ""

    reward_meta = row.get("reward_meta")
    if not reward_meta:
        row["reward_meta"] = {
            "source": "agentic_runtime",
            "question": question,
            "as_of_time": args.request_time_utc or "",
            "model_should_output_json": getattr(args, "resolved_prompt_mode", args.prompt_mode) == "json",
        }

    return row


def latest_checkpoint(root: str) -> Optional[Path]:
    p = Path(root)
    if not p.exists():
        return None
    hits: List[Tuple[int, Path]] = []
    for d in p.glob("checkpoint-*"):
        if not d.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)$", d.name)
        hits.append((int(m.group(1)) if m else -1, d))
    if not hits:
        return None
    return sorted(hits, key=lambda x: x[0])[-1][1]


def resolve_loadable_or_latest(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    direct_names = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    if p.is_dir() and any((p / name).is_file() for name in direct_names):
        return p

    latest = latest_checkpoint(str(p))
    if latest is not None:
        return latest

    return p


def resolve_grpo_checkpoint(args: argparse.Namespace) -> Path:
    candidates: List[Path] = []
    if args.grpo_ckpt:
        candidates.append(Path(args.grpo_ckpt))
    candidates.extend([
        Path("outputs/stageD_grpo_qwenprefix_best"),
        Path("output_stageC_warmup/stageD_grpo_qwenprefix_best"),
    ])
    latest = latest_checkpoint(args.grpo_run_dir)
    if latest is not None:
        candidates.append(latest)

    for p in candidates:
        if p.exists():
            return p

    msg = ["Could not find a GRPO checkpoint. Tried:"]
    msg.extend(f"  - {p}" for p in candidates)
    raise FileNotFoundError("\n".join(msg))


def build_mixlora_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_experts": int(args.mixlora_num_universal) + int(args.mixlora_num_pred) + int(args.mixlora_num_reason),
        "lora_r": int(args.mixlora_r),
        "lora_alpha": float(args.mixlora_alpha),
        "enable_attention": False,
        "enable_ffn": True,
        "share_router_for_wi": True,
        "num_tasks": 3,
        "ensure_nonzero_gating": False,
        "num_universal": int(args.mixlora_num_universal),
        "num_pred": int(args.mixlora_num_pred),
        "num_reason": int(args.mixlora_num_reason),
        "rank_universal_mul": float(args.mixlora_rank_universal_mul),
        "target_modules_override": {
            "atte": "self_attn",
            "ffn": "mlp",
            "q": [],
            "k": [],
            "v": [],
            "o": [],
            "wi": ["mlp.gate_proj", "mlp.up_proj"],
            "wo": ["mlp.down_proj"],
        },
    }


class AgenticInferenceEngine:
    def __init__(self, args: argparse.Namespace, *, prompt_mode: str) -> None:
        self.args = args
        self.prompt_mode = prompt_mode

        import torch
        from transformers import AutoTokenizer
        from dataprocessing import DataCollatorSmartCity, _normalize_row_core

        self.torch = torch
        self.normalize_row_core = _normalize_row_core

        if args.stage == "stagec":
            self.tok, self.enc_tok, self.model, self.stop_ids = self._load_stagec_model(AutoTokenizer)
        elif args.stage == "grpo":
            self.tok, self.enc_tok, self.model, self.stop_ids = self._load_grpo_model(AutoTokenizer)
        else:
            raise ValueError(f"unknown stage: {args.stage}")

        if torch.cuda.is_available():
            default_device = "cuda:0"
        else:
            default_device = "cpu"

        device_name = default_device if args.device == "auto" else args.device
        self.device = torch.device(device_name)
        if self.device.type != "cuda" and not args.allow_cpu:
            raise RuntimeError("CUDA is not available. Pass --allow_cpu only for tiny debug runs.")

        self.model.to(self.device)
        self.model.eval()

        try:
            if hasattr(self.model.decoder, "gradient_checkpointing_disable"):
                self.model.decoder.gradient_checkpointing_disable()
        except Exception:
            pass

        self.collator = DataCollatorSmartCity(
            self.tok,
            chunk_tokenizer=self.enc_tok,
            chunk_size_tokens=int(args.chunk_size_tokens),
            chunk_overlap=int(args.chunk_overlap),
            join_fields_before_chunk=True,
            add_field_header=True,
            max_length=int(args.max_length),
            pad_token_id=self.tok.pad_token_id,
            label_pad_token_id=-100,
            chunk_token_factor=int(args.num_chunk_tokens),
            image_token_factor=int(args.num_image_tokens),
            max_images_per_sample=int(args.max_images_per_sample),
            image_size=int(args.image_size),
            max_chunks_per_sample=args.max_chunks_per_sample,
            add_eos_to_labels=True,
            wrap_reason_with_tags=True,
        )

    def _load_stagec_model(self, AutoTokenizer):
        from types import SimpleNamespace
        from inference_stageBC_compare import assert_vocab_safe_no_resize, setup_tokenizer_no_modify
        from model.builder import build_smartcity_llm

        stagec_dir = resolve_loadable_or_latest(self.args.stagec_dir)

        model_args = SimpleNamespace(
            full_model_load_dir=str(stagec_dir),
            decoder_name_or_path=self.args.decoder,
            padding_token=self.args.padding_token,
        )
        tok = setup_tokenizer_no_modify(model_args)
        enc_tok = AutoTokenizer.from_pretrained(
            self.args.encoder,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=bool(self.args.local_files_only),
        )

        model = build_smartcity_llm(
            decoder_name=self.args.decoder,
            encoder_name=self.args.encoder,
            alpha_understand=0.2,
            beta_prediction=1.0,
            beta_reason=1.0,
            lambda_rat=0.3,
            num_chunk_tokens=int(self.args.num_chunk_tokens),
            adapter_heads=4,
            encoder_max_length=512,
            imagebind_variant=self.args.imagebind_variant,
            num_image_tokens=int(self.args.num_image_tokens),
            group_layers=2,
            group_heads=8,
            group_tau=1.0,
            open_roberta=False,
            open_vpma=False,
            open_imagebind=False,
            open_image_projector=False,
            freeze_decoder=True,
            use_mixlora=True,
            mixlora_cfg_overrides=build_mixlora_overrides(self.args),
        )
        print(f"[load-stagec] full_model_load_dir={stagec_dir}", flush=True)
        model.load_full_state_dict(str(stagec_dir), cast_to_target_dtype=True, verbose=True)
        self._sync_token_ids(model, tok)
        assert_vocab_safe_no_resize(model, tok)
        return tok, enc_tok, model, build_stop_token_ids_local(tok)

    def _load_grpo_model(self, AutoTokenizer):
        from scripts.diag_stageD_ckpt_multi_once import (
            build_model,
            load_base_then_adapter,
            resolve_ckpt_dir,
        )
        from train_grpo import build_stop_token_ids, load_tokenizer_no_modify

        ckpt = resolve_grpo_checkpoint(self.args)
        ckpt_resolved = resolve_ckpt_dir(str(ckpt))
        print(f"[load-grpo] base_dir={self.args.base_dir}", flush=True)
        print(f"[load-grpo] ckpt={ckpt_resolved}", flush=True)

        tok = load_tokenizer_no_modify(
            ckpt_dir=str(ckpt_resolved),
            fallback_name=self.args.decoder,
            padding_token=self.args.padding_token,
            local_files_only=bool(self.args.local_files_only),
        )
        enc_tok = AutoTokenizer.from_pretrained(
            self.args.encoder,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=bool(self.args.local_files_only),
        )

        model = build_model(self.args.decoder, self.args.encoder)
        load_base_then_adapter(model, self.args.base_dir, str(ckpt_resolved))
        self._sync_token_ids(model, tok)
        return tok, enc_tok, model, build_stop_token_ids(tok)

    @staticmethod
    def _sync_token_ids(model: Any, tok: Any) -> None:
        try:
            if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
                model.decoder.config.pad_token_id = tok.pad_token_id
                model.decoder.config.eos_token_id = tok.eos_token_id
                model.decoder.config.use_cache = True
            if hasattr(model, "config"):
                model.config.pad_token_id = tok.pad_token_id
                model.config.eos_token_id = tok.eos_token_id
                model.config.use_cache = True
        except Exception:
            pass

    def normalize_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self.normalize_row_core(row, prompt_style="qwen3", ensure_markers=True)

    def assistant_prefill(self, row: Dict[str, Any]) -> str:
        enabled = os.environ.get("SMARTCITY_QWEN_REASON_TEMPLATE_PREFIX", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "y",
        }
        if not enabled or str(row.get("task", "")).lower() != "reason":
            return ""

        mode = str(row.get("reasoning_mode", "") or "").strip().lower().replace("-", "_")
        if mode in {"no_think", "nothink", "no", "none", "instant"}:
            if os.environ.get("SMARTCITY_QWEN_NOTHINK_EMPTY_TEMPLATE", "qwen").strip().lower() == "compact":
                return "<think></think>\n\n"
            return "<think>\n\n</think>\n\n"
        return "<think>\n"

    def infer_one(self, row: Dict[str, Any]) -> Dict[str, Any]:
        torch = self.torch
        norm = self.normalize_record(row)

        ex = dict(norm)
        ex["answer"] = ""
        ex["target"] = ""
        if str(ex.get("reasoning_mode", "") or "").lower().replace("-", "_") == "no_think":
            ex["thinking"] = []

        batch = self.collator([ex])
        meta = batch.get("meta", None)

        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)

        with torch.inference_mode():
            dec_emb = self.model.decoder.get_input_embeddings()
            dev = dec_emb.weight.device
            dtype = dec_emb.weight.dtype
            batch_size = int(batch["prefix_ids"].size(0))

            enc_out = self.model.encode_modalities(
                chunk_txts_all=batch.get("chunk_txts", None),
                pixel_values=batch.get("pixel_values", None),
                n_images=batch.get("n_images", None),
                batch_size=batch_size,
                device=dev,
                dtype=dtype,
            )
            chunk_embeds, chunk_mask, image_embeds, image_mask = enc_out[:4]

            gen_ids = self.model.generate_with_modalities_cached(
                prefix_ids=batch["prefix_ids"],
                prefix_mask=batch["prefix_mask"],
                suffix_ids=batch["suffix_ids"],
                suffix_mask=batch["suffix_mask"],
                chunk_embeds=chunk_embeds,
                chunk_mask=chunk_mask,
                image_embeds=image_embeds,
                image_mask=image_mask,
                task_ids=batch.get("task_ids", None),
                max_new_tokens=int(self.args.max_new_tokens),
                temperature=float(self.args.temperature),
                top_k=int(self.args.top_k),
                top_p=float(self.args.top_p),
                do_sample=bool(self.args.do_sample or float(self.args.temperature) > 0.0),
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                stop_token_ids=self.stop_ids,
            )

        completion = self.tok.decode(
            gen_ids[0].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        completion = clean_generation_text(completion)
        full_text = self.assistant_prefill(ex) + completion
        public_answer = extract_public_answer(full_text)

        parsed_json = parse_first_json_object(public_answer)
        cited_ids = unique_evidence_ids(public_answer)

        return {
            "sample_id": ex.get("sample_id", ""),
            "stage": self.args.stage,
            "prompt_mode": self.prompt_mode,
            "reasoning_mode": ex.get("reasoning_mode", ""),
            "prediction": full_text,
            "completion_only": completion,
            "public_answer": public_answer,
            "public_json_parse_ok": parsed_json is not None,
            "parsed_public_json": parsed_json,
            "cited_ids": cited_ids,
            "prompt_preview": compact_text(ex.get("prompt", ""), 1800),
            "evidence_catalog": ex.get("evidence_catalog", ""),
            "key_evidence_ids": ex.get("key_evidence_ids", []),
            "key_source_types": ex.get("key_source_types", []),
            "meta": jsonable(meta),
        }


def build_stop_token_ids_local(tok: Any) -> Optional[List[int]]:
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    unk = getattr(tok, "unk_token_id", None)
    out: List[int] = []
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(s)
        if not isinstance(tid, int) or tid < 0:
            continue
        if unk is not None and tid == unk:
            continue
        if tid == eos:
            continue
        if pad is not None and tid == pad:
            continue
        out.append(int(tid))
    out = list(dict.fromkeys(out))
    return out or None


def clean_generation_text(text: str) -> str:
    s = str(text or "")
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if tok in s:
            s = s.split(tok, 1)[0]
    return s.strip()


def extract_public_answer(text: str) -> str:
    s = str(text or "").strip()
    low = s.lower()
    if "</think>" in low:
        idx = low.rfind("</think>")
        return s[idx + len("</think>") :].strip()
    return re.sub(r"</?think>", "", s, flags=re.I).strip()


def parse_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

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
                try:
                    obj = json.loads(s[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


EVID_RE = re.compile(r"\bE_[A-Za-z0-9_]+\b")


def unique_evidence_ids(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for m in EVID_RE.finditer(text or ""):
        eid = m.group(0)
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run agentic data collection, then run TransCity StageC or GRPO inference on the collected record.",
    )

    p.add_argument("--question", action="append", default=[], help="Question to ask. Can be used multiple times.")
    p.add_argument("--questions_file", default="", help="UTF-8 text file with one question per line.")
    p.add_argument("--agent_record_json", default="", help="Use an existing agent record JSON/JSONL instead of running agents.")

    p.add_argument("--stage", choices=["stagec", "grpo"], default="grpo")
    p.add_argument("--prompt_mode", choices=["auto", "json", "freeform"], default="auto")
    p.add_argument("--qwen_prefix", default="auto", help="auto/true/false. Default: true for freeform, false for json.")
    p.add_argument("--reasoning_mode", choices=["think", "no_think"], default="think")

    p.add_argument("--decoder", default=os.environ.get("DECODER", DEFAULT_DECODER))
    p.add_argument("--encoder", default=os.environ.get("ENCODER", DEFAULT_ENCODER))
    p.add_argument("--stagec_dir", default=os.environ.get("STAGEC_DIR", "checkpoint/stageC"))
    p.add_argument("--base_dir", default=os.environ.get("BASE_DIR", "output_stageC_warmup/stageC_final"))
    p.add_argument("--grpo_ckpt", default=os.environ.get("CKPT", ""))
    p.add_argument("--grpo_run_dir", default=os.environ.get("RUN_DIR", "outputs_grpo"))
    p.add_argument("--padding_token", default="<|pad|>")
    p.add_argument("--local_files_only", type=str2bool, default=True)

    p.add_argument("--device", default="auto")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--imagebind_variant", default=os.environ.get("IMAGEBIND_VARIANT", "google/siglip-so400m-patch14-384"))
    p.add_argument("--num_chunk_tokens", type=int, default=4)
    p.add_argument("--num_image_tokens", type=int, default=32)
    p.add_argument("--mixlora_num_universal", type=int, default=4)
    p.add_argument("--mixlora_num_pred", type=int, default=8)
    p.add_argument("--mixlora_num_reason", type=int, default=8)
    p.add_argument("--mixlora_rank_universal_mul", type=float, default=2.0)
    p.add_argument("--mixlora_r", type=int, default=8)
    p.add_argument("--mixlora_alpha", type=float, default=16.0)

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--max_images_per_sample", type=int, default=1)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--max_chunks_per_sample", type=int, default=None)
    p.add_argument("--chunk_size_tokens", type=int, default=510)
    p.add_argument("--chunk_overlap", type=int, default=0)

    p.add_argument("--image_store", choices=["base64", "file", "none"], default="file")
    p.add_argument("--with_flow", type=str2bool, default=True)
    p.add_argument("--with_satellite", type=str2bool, default=True)
    p.add_argument("--with_context", type=str2bool, default=True)
    p.add_argument("--with_web_events", type=str2bool, default=True)
    p.add_argument("--no-flow", dest="with_flow", action="store_false")
    p.add_argument("--no-satellite", dest="with_satellite", action="store_false")
    p.add_argument("--no-context", dest="with_context", action="store_false")
    p.add_argument("--no-web-events", dest="with_web_events", action="store_false")
    p.add_argument("--verbose_agent", action="store_true")
    p.add_argument("--save_agent_outputs", action="store_true")

    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--local_start", default=None)
    p.add_argument("--local_end", default=None)
    p.add_argument("--request_time_utc", default=None)
    p.add_argument("--sensor_id", default=None)
    p.add_argument("--lanes", type=int, default=None)
    p.add_argument("--direction", default=None)
    p.add_argument("--freeway", default=None)
    p.add_argument("--traffic_flow", default="")
    p.add_argument("--history_average_flow", default="")

    p.add_argument("--skip_model", action="store_true", help="Only run/normalize the agent collection path.")
    p.add_argument("--out_dir", default=os.environ.get("OUT_DIR", ""))
    p.add_argument("--print_full", action="store_true")
    return p


async def main_async() -> None:
    args = build_arg_parser().parse_args()
    out_dir = timestamp_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_mode = configure_prompt_env(args.stage, args.prompt_mode, args.qwen_prefix)
    setattr(args, "resolved_prompt_mode", prompt_mode)

    raw_records: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    if args.agent_record_json:
        loaded = read_agent_records(args.agent_record_json)
        for i, rec in enumerate(loaded):
            q = str(rec.get("question", "") or rec.get("sample_id", "") or f"record_{i}")
            raw_records.append(enrich_agent_record(rec, question=q, index=i, args=args))
            debug_rows.append({"source": args.agent_record_json, "index": i})
    else:
        questions = read_questions(args)
        if not questions:
            raise ValueError("Provide --question, --questions_file, or --agent_record_json.")

        for i, question in enumerate(questions):
            print(f"[agent] collecting {i + 1}/{len(questions)}: {compact_text(question, 160)}", flush=True)
            rec, dbg = await collect_agent_record(question, args)
            raw_records.append(enrich_agent_record(rec, question=question, index=i, args=args))
            debug_rows.append(dbg)

    write_jsonl(out_dir / "agent_records.jsonl", raw_records)
    write_jsonl(out_dir / "agent_debug.jsonl", debug_rows)
    print(f"[saved] {out_dir / 'agent_records.jsonl'}", flush=True)

    if args.skip_model:
        print(f"[done] agent records only: {out_dir}", flush=True)
        return

    if args.with_satellite and args.image_store == "base64":
        raise ValueError("Model inference needs file-backed images. Use --image_store file, --image_store none, or --no-satellite.")

    print(f"[model] stage={args.stage} prompt_mode={prompt_mode}", flush=True)
    engine = AgenticInferenceEngine(args, prompt_mode=prompt_mode)

    predictions: List[Dict[str, Any]] = []
    normalized_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(raw_records):
        print(f"[infer] {i + 1}/{len(raw_records)} sample_id={row.get('sample_id', '')}", flush=True)
        pred = engine.infer_one(row)
        norm = engine.normalize_record(row)
        predictions.append(pred)
        normalized_rows.append(norm)

        per_sample = out_dir / f"prediction_{safe_label(pred.get('sample_id'), str(i))}.json"
        per_sample.write_text(json.dumps(jsonable(pred), ensure_ascii=False, indent=2), encoding="utf-8")

        print("[answer]", compact_text(pred.get("public_answer", ""), 1000), flush=True)
        if args.print_full:
            print("[full prediction]\n" + str(pred.get("prediction", "")), flush=True)

    write_jsonl(out_dir / "normalized_model_inputs.jsonl", normalized_rows)
    write_jsonl(out_dir / "predictions.jsonl", predictions)

    latest_txt = out_dir / "latest_answer.txt"
    latest_txt.write_text(
        "\n\n".join(str(p.get("public_answer", "")) for p in predictions),
        encoding="utf-8",
    )

    print(f"[saved] {out_dir / 'normalized_model_inputs.jsonl'}", flush=True)
    print(f"[saved] {out_dir / 'predictions.jsonl'}", flush=True)
    print(f"[saved] {latest_txt}", flush=True)
    print(f"[done] {out_dir}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
