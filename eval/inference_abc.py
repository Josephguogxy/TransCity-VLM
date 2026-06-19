from __future__ import annotations

import inspect
import json
import math
import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import datasets as hf_datasets
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataprocessing import (
    DataCollatorSmartCity,
    ensure_smartcity_features,
    load_jsonl_dataset,
)
from model.builder import build_smartcity_llm


# =========================
# Args
# =========================
@dataclass
class ModelArguments:
    decoder_name_or_path: str = field(metadata={"help": "LLM decoder name/path"})
    encoder_name_or_path: str = field(metadata={"help": "Chunk text encoder name/path"})

    num_chunk_tokens: int = 4
    adapter_heads: int = 4
    encoder_max_length: int = 512

    num_image_tokens: int = 32
    imagebind_variant: str = "google/siglip-so400m-patch14-384"
    group_layers: int = 2
    group_heads: int = 8
    group_tau: float = 1.0

    open_roberta: bool = True
    open_vpma: bool = True
    open_imagebind: bool = False
    open_image_projector: bool = True
    freeze_decoder: bool = False

    alpha_understand: float = 0.3
    beta_prediction: float = 0.7
    beta_reason: float = 0.0
    lambda_rat: float = 0.0

    padding_token: Optional[str] = field(
        default="<|pad|>",
        metadata={"help": "If tokenizer has no pad_token, try to use this EXISTING token (NO add)."},
    )

    full_model_load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Stage-B/Stage-C checkpoint dir (for model.load_full_state_dict)."},
    )

    # Stage-C uses MixLoRA. Default False keeps Stage-B behavior unchanged.
    use_mixlora: bool = False
    mixlora_num_experts: int = 20
    mixlora_r: int = 8
    mixlora_alpha: float = 16.0
    mixlora_share_router_for_wi: bool = True
    mixlora_enable_attention: bool = False
    mixlora_ensure_nonzero_gating: bool = False
    mixlora_num_universal: Optional[int] = 4
    mixlora_num_pred: Optional[int] = 8
    mixlora_num_reason: Optional[int] = 8
    mixlora_rank_universal_mul: float = 2.0


@dataclass
class DataInferenceArguments:
    test_files: List[str] = field(default_factory=list, metadata={"help": "List of test jsonl paths."})

    max_infer_samples: Optional[int] = None
    do_shuffle: bool = False

    max_length: int = 4096
    max_images_per_sample: int = 1
    image_size: int = 384
    max_chunks_per_sample: Optional[int] = None

    gen_max_new_tokens: int = 256
    gen_num_beams: int = 1

    # Sampling generation controls.
    # temperature only matters when gen_do_sample=True.
    gen_temperature: float = 0.0
    gen_top_p: float = 1.0
    gen_top_k: int = 0
    gen_do_sample: bool = False

    mape_eps: float = 1e-3


# =========================
# Dist helpers
# =========================
def get_dist_info() -> Tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def init_dist_if_needed(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def ddp_sync_weights_from_rank0(model: torch.nn.Module, *, enabled: bool = True) -> None:
    """
    In torchrun DDP inference (no ZeRO), if your custom loader only loads rank0,
    broadcasting parameters from rank0 makes all ranks consistent.
    """
    if not enabled:
        return
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return

    src = 0
    for _, p in model.named_parameters():
        if p is not None and torch.is_tensor(p.data):
            dist.broadcast(p.data, src=src)

    for _, b in model.named_buffers():
        if b is not None and torch.is_tensor(b.data):
            dist.broadcast(b.data, src=src)


# =========================
# Tokenizer (NO add / NO resize)
# =========================
def setup_tokenizer_no_modify(model_args: ModelArguments) -> AutoTokenizer:
    candidates = []
    if model_args.full_model_load_dir:
        candidates.append(model_args.full_model_load_dir)
    candidates.append(model_args.decoder_name_or_path)

    tok = None
    tok_src = None
    last_err = None
    for path in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(
                path,
                use_fast=True,
                padding_side="left",
                trust_remote_code=True,
                local_files_only=True,
            )
            tok_src = path
            break
        except Exception as e:
            last_err = e

    if tok is None:
        raise last_err

    tok.padding_side = "left"
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot run safely.")

    if tok.pad_token_id is None:
        candidates2 = []
        if model_args.padding_token:
            candidates2.append(model_args.padding_token)
        candidates2 += ["<|pad|>", "<|endoftext|>", "<|im_end|>"]

        chosen = None
        for cand in candidates2:
            if not cand:
                continue
            if cand not in tok.get_vocab():
                continue
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tok.eos_token_id:
                chosen = cand
                break

        tok.pad_token = chosen if chosen is not None else tok.eos_token

    print(f"[tok] src={tok_src} | class={tok.__class__.__name__} fast={getattr(tok,'is_fast',None)} "
          f"| len={len(tok)} vocab_size={getattr(tok,'vocab_size',None)} "
          f"| pad={tok.pad_token}({tok.pad_token_id}) eos={tok.eos_token}({tok.eos_token_id})")
    return tok


def tokenizer_effective_vocab_size(tok: AutoTokenizer) -> int:
    vocab = tok.get_vocab()
    max_id = max(vocab.values()) if vocab else -1
    for t in tok.all_special_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            max_id = max(max_id, tid)
    return int(max_id + 1)


def assert_vocab_safe_no_resize(model, tok: AutoTokenizer) -> None:
    dec = getattr(model, "decoder", model)
    emb = dec.get_input_embeddings()
    if emb is None or (not hasattr(emb, "weight")):
        return
    cur = int(emb.weight.size(0))
    tgt_eff = int(tokenizer_effective_vocab_size(tok))
    if tgt_eff > cur:
        raise ValueError(f"[FATAL] tokenizer_effective_vocab={tgt_eff} > model_embedding_rows={cur} (OOB risk).")
    if cur != tgt_eff:
        print(f"[OK] padded vocab: model_embedding_rows={cur} vs tokenizer_effective_vocab={tgt_eff} (len(tok)={len(tok)})")


def safe_batch_decode(tok: AutoTokenizer, ids: torch.Tensor, *, name: str = "ids") -> List[str]:
    if not isinstance(ids, torch.Tensor):
        return tok.batch_decode(ids, skip_special_tokens=True)

    vsz = int(len(tok))
    bad = ids >= vsz
    if bad.any():
        n_bad = int(bad.sum().item())
        print(f"[decode] {name}: found {n_bad} ids >= len(tokenizer)={vsz}, replace with eos({int(tok.eos_token_id)})")
        ids = ids.clone()
        ids[bad] = int(tok.eos_token_id)

    return tok.batch_decode(ids, skip_special_tokens=True)


# =========================
# Debug helpers (PRINT)
# =========================
def _md5_tensor_prefix(t: torch.Tensor, k: int = 2048) -> str:
    x = t.detach().float().view(-1)[:k].cpu().numpy().tobytes()
    return hashlib.md5(x).hexdigest()


def debug_check_weight_consistency_print(model, *, strict: bool = True) -> None:
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return

    rank = dist.get_rank()
    world = dist.get_world_size()

    dec = getattr(model, "decoder", model)
    emb = dec.get_input_embeddings().weight
    h = _md5_tensor_prefix(emb, k=2048)
    payload = {"rank": rank, "hash": h, "shape": tuple(emb.shape), "dtype": str(emb.dtype), "device": str(emb.device)}

    obj_list: List[Any] = [None for _ in range(world)]
    dist.all_gather_object(obj_list, payload)

    if rank == 0:
        hashes = [x["hash"] for x in obj_list]
        print("[weights] per-rank emb md5:", hashes)
        if len(set(hashes)) != 1:
            print("[weights][ERROR] NOT CONSISTENT across ranks! Example payloads:", obj_list[:2])
            if strict:
                raise RuntimeError("Model weights differ across ranks. Fix loading or enable SMARTCITY_SYNC_WEIGHTS=1.")


def debug_print_chunk_len(enc_tok: AutoTokenizer, batch: Dict[str, Any], *, limit_chunks: int = 64) -> None:
    chunk_txts = batch.get("chunk_txts", None)
    if not isinstance(chunk_txts, list):
        print("[debug] no chunk_txts in batch")
        return

    flat: List[str] = []
    for lst in chunk_txts:
        if isinstance(lst, list):
            for s in lst:
                if isinstance(s, str) and s:
                    flat.append(s)
    flat = flat[:limit_chunks]
    if not flat:
        print("[debug] chunk_txts is empty")
        return

    old = getattr(enc_tok, "model_max_length", None)
    try:
        enc_tok.model_max_length = int(1e9)
    except Exception:
        pass

    lens = []
    for s in flat:
        ids = enc_tok(s, add_special_tokens=True, truncation=False).input_ids
        lens.append(len(ids))

    if old is not None:
        try:
            enc_tok.model_max_length = old
        except Exception:
            pass

    lens_sorted = sorted(lens)
    mx = lens_sorted[-1]
    p95 = lens_sorted[max(0, int(0.95 * len(lens_sorted)) - 1)]
    n_over = sum(1 for x in lens_sorted if x > 512)
    print(f"[debug] encoder chunk token lens: max={mx} p95={p95} over512={n_over}/{len(lens_sorted)} top5={lens_sorted[-5:]}")


# =========================
# Dataset / Collator
# =========================
def load_test_dataset(path: str) -> HFDataset:
    assert path.endswith(".jsonl"), f"Expect *.jsonl, got {path!r}"
    ds = load_jsonl_dataset(path, split_name="test")

    num_proc = int(os.environ.get("SMARTCITY_MAP_NUM_PROC", "16"))
    try:
        sig = inspect.signature(ensure_smartcity_features)
        kwargs = {}
        if "num_proc" in sig.parameters:
            kwargs["num_proc"] = num_proc
        ds = ensure_smartcity_features(ds, **kwargs)
    except Exception:
        ds = ensure_smartcity_features(ds, num_proc=num_proc)

    return ds


def build_collator(
    tok: AutoTokenizer,
    enc_tok: AutoTokenizer,
    model_args: ModelArguments,
    data_args: DataInferenceArguments,
) -> DataCollatorSmartCity:
    kwargs: Dict[str, Any] = dict(
        max_length=data_args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=model_args.num_chunk_tokens,
        image_token_factor=model_args.num_image_tokens,
        max_images_per_sample=data_args.max_images_per_sample,
        image_size=data_args.image_size,
        max_chunks_per_sample=data_args.max_chunks_per_sample,
    )

    try:
        sig = inspect.signature(DataCollatorSmartCity.__init__)
        if "chunk_tokenizer" in sig.parameters:
            kwargs["chunk_tokenizer"] = enc_tok
        if "chunk_size_tokens" in sig.parameters:
            kwargs["chunk_size_tokens"] = 510
        if "chunk_overlap" in sig.parameters:
            kwargs["chunk_overlap"] = 0
        if "join_fields_before_chunk" in sig.parameters:
            kwargs["join_fields_before_chunk"] = True
        if "add_field_header" in sig.parameters:
            kwargs["add_field_header"] = True
    except Exception:
        pass

    return DataCollatorSmartCity(tok, **kwargs)


# =========================
# Task type + numeric parsing
# =========================
def infer_task_type(test_file: str) -> str:
    name = os.path.basename(test_file).lower()
    if "elec" in name or "electric" in name or "power" in name:
        return "elec"
    if "traffic" in name or "flow" in name:
        return "traffic"
    return "generic"


_NUM_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")

def _extract_segment(text: str) -> str:
    m = re.search(r"\[([^\]]+)\]", text or "", flags=re.DOTALL)
    return m.group(1) if m else (text or "")

def extract_numbers(text: str) -> List[float]:
    seg = _extract_segment(text or "")
    nums = _NUM_PATTERN.findall(seg)
    return [float(x) for x in nums]


def _finite_or_nan(x: float) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _step_dict(values: List[float]) -> Dict[str, float]:
    return {f"step_{i + 1}": _finite_or_nan(v) for i, v in enumerate(values)}


def empty_metrics(mape_eps: float = 1e-3) -> Dict[str, Any]:
    nan = float("nan")
    return {
        "num_valid_samples": 0,
        "num_steps": 0,
        "num_points": 0,
        "MAE": nan,
        "MAPE": nan,
        "RMSE": nan,
        "SMAPE": nan,
        "WMAPE": nan,
        "MSE": nan,
        "MAPE_eps": float(mape_eps),
        "MAPE_num_used_points": 0,
        "MAPE_num_ignored_zero_or_near_zero_points": 0,
        "SMAPE_num_used_points": 0,
        "metric_units": {
            "MAE": "same_as_target",
            "RMSE": "same_as_target",
            "MSE": "squared_target_unit",
            "MAPE": "percent",
            "SMAPE": "percent",
            "WMAPE": "percent",
        },
        "MAE_by_step_1_to_12": {},
        "MAE_cumsum_1_to_12": {},
        "MAE_cummean_1_to_12": {},
        "MAE_12_step_cumulative_final": nan,
        "MAE_12_step_cummean_final": nan,
        "CUMULATIVE_FORECAST_MAE_1_to_12": {},
        "CUMULATIVE_FORECAST_MAE_12_final": nan,
    }


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, mape_eps: float = 1e-3) -> Dict[str, Any]:
    """
    y_true/y_pred shape: [N, T].

    Main metrics:
      MAE, RMSE, SMAPE, WMAPE use all aligned points.
      MAPE ignores target points with abs(target) <= mape_eps, so zero/near-zero labels are not counted.

    12-step MAE cumulative error:
      MAE_by_step_1_to_12: per-horizon MAE.
      MAE_cumsum_1_to_12: cumulative sum of per-horizon MAE over first k steps.
      MAE_cummean_1_to_12: average MAE over first k steps.
      CUMULATIVE_FORECAST_MAE_1_to_12: mean(abs(cumsum(pred - true))) for each horizon k.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0 or yp.size == 0:
        return empty_metrics(mape_eps)
    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)

    n = min(yt.shape[0], yp.shape[0])
    t = min(yt.shape[1], yp.shape[1])
    if n <= 0 or t <= 0:
        return empty_metrics(mape_eps)
    yt = yt[:n, :t]
    yp = yp[:n, :t]

    diff = yp - yt
    abs_err = np.abs(diff)

    mae = float(np.mean(abs_err))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))

    denom = np.abs(yt)
    # Strictly ignore 0 and near-0 target values for MAPE.
    mape_mask = denom > float(mape_eps)
    mape = float(np.mean(abs_err[mape_mask] / denom[mape_mask]) * 100.0) if np.any(mape_mask) else float("nan")

    smape_denom = np.abs(yt) + np.abs(yp)
    smape_mask = smape_denom > float(mape_eps)
    smape = float(np.mean(200.0 * abs_err[smape_mask] / smape_denom[smape_mask])) if np.any(smape_mask) else float("nan")

    wden = float(np.sum(np.abs(yt)))
    wmape = float(np.sum(abs_err) / wden * 100.0) if wden > 0 else float("nan")

    h = min(12, t)
    per_step_mae = np.mean(abs_err[:, :h], axis=0) if h > 0 else np.asarray([], dtype=np.float64)
    mae_cumsum = np.cumsum(per_step_mae) if h > 0 else np.asarray([], dtype=np.float64)
    mae_cummean = mae_cumsum / np.arange(1, h + 1) if h > 0 else np.asarray([], dtype=np.float64)

    # Another common meaning of cumulative forecast error:
    # for each k, first accumulate signed error over steps 1..k, then take MAE over samples.
    cumulative_forecast_mae = (
        np.mean(np.abs(np.cumsum(diff[:, :h], axis=1)), axis=0)
        if h > 0 else np.asarray([], dtype=np.float64)
    )

    return {
        "num_valid_samples": int(n),
        "num_steps": int(t),
        "num_points": int(n * t),
        "MAE": _finite_or_nan(mae),
        "MAPE": _finite_or_nan(mape),
        "RMSE": _finite_or_nan(rmse),
        "SMAPE": _finite_or_nan(smape),
        "WMAPE": _finite_or_nan(wmape),
        "MSE": _finite_or_nan(mse),
        "MAPE_eps": float(mape_eps),
        "MAPE_num_used_points": int(np.sum(mape_mask)),
        "MAPE_num_ignored_zero_or_near_zero_points": int(mape_mask.size - np.sum(mape_mask)),
        "SMAPE_num_used_points": int(np.sum(smape_mask)),
        "metric_units": {
            "MAE": "same_as_target",
            "RMSE": "same_as_target",
            "MSE": "squared_target_unit",
            "MAPE": "percent",
            "SMAPE": "percent",
            "WMAPE": "percent",
        },
        "MAE_by_step_1_to_12": _step_dict([float(x) for x in per_step_mae.tolist()]),
        "MAE_cumsum_1_to_12": _step_dict([float(x) for x in mae_cumsum.tolist()]),
        "MAE_cummean_1_to_12": _step_dict([float(x) for x in mae_cummean.tolist()]),
        "MAE_12_step_cumulative_final": _finite_or_nan(float(mae_cumsum[-1])) if h > 0 else float("nan"),
        "MAE_12_step_cummean_final": _finite_or_nan(float(mae_cummean[-1])) if h > 0 else float("nan"),
        "CUMULATIVE_FORECAST_MAE_1_to_12": _step_dict([float(x) for x in cumulative_forecast_mae.tolist()]),
        "CUMULATIVE_FORECAST_MAE_12_final": _finite_or_nan(float(cumulative_forecast_mae[-1])) if h > 0 else float("nan"),
    }


# =========================
# Main
# =========================
def main():
    try:
        hf_datasets.disable_progress_bar()
    except Exception:
        pass

    parser = HfArgumentParser((ModelArguments, DataInferenceArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank, world_size, local_rank = get_dist_info()
    is_main = (rank == 0)

    set_seed(int(training_args.seed) + int(rank))

    debug = os.environ.get("SMARTCITY_DEBUG", "0") == "1"
    debug_print_samples = int(os.environ.get("SMARTCITY_DEBUG_PRINT_SAMPLES", "3"))
    debug_loss_steps = int(os.environ.get("SMARTCITY_DEBUG_LOSS_STEPS", "1"))
    sync_weights = os.environ.get("SMARTCITY_SYNC_WEIGHTS", "1") == "1"
    strict_weights = os.environ.get("SMARTCITY_STRICT_WEIGHTS", "1") == "1"

    # stagger load
    stagger = int(os.environ.get("SMARTCITY_STAGGER_LOAD_SEC", "0"))
    if stagger > 0:
        time.sleep(stagger * int(local_rank))

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    init_dist_if_needed(world_size)

    if is_main:
        print("==== StageB/StageC single-GPU queued inference start ====")
        print("world_size=", world_size, "| output_root=", training_args.output_dir)
        print("test_files=", data_args.test_files)
        print(f"[gen] temperature={data_args.gen_temperature} top_p={data_args.gen_top_p} "
              f"top_k={data_args.gen_top_k} do_sample={data_args.gen_do_sample}")

    tok = setup_tokenizer_no_modify(model_args)
    enc_tok = AutoTokenizer.from_pretrained(
        model_args.encoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    mix_cfg_overrides = None
    if bool(model_args.use_mixlora):
        # Must match train_mm_stageC.py / run_stageC_ds.sh.
        _u = model_args.mixlora_num_universal
        _p = model_args.mixlora_num_pred
        _r = model_args.mixlora_num_reason
        _sum_counts = ((_u or 0) + (_p or 0) + (_r or 0)) or None
        mix_cfg_overrides = dict(
            num_experts=(_sum_counts or model_args.mixlora_num_experts),
            lora_r=model_args.mixlora_r,
            lora_alpha=model_args.mixlora_alpha,
            enable_attention=bool(model_args.mixlora_enable_attention),
            enable_ffn=True,
            share_router_for_wi=bool(model_args.mixlora_share_router_for_wi),
            num_tasks=3,
            ensure_nonzero_gating=bool(model_args.mixlora_ensure_nonzero_gating),
            num_universal=model_args.mixlora_num_universal,
            num_pred=model_args.mixlora_num_pred,
            num_reason=model_args.mixlora_num_reason,
            rank_universal_mul=model_args.mixlora_rank_universal_mul,
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

    model = build_smartcity_llm(
        decoder_name=model_args.decoder_name_or_path,
        encoder_name=model_args.encoder_name_or_path,
        alpha_understand=model_args.alpha_understand,
        beta_prediction=model_args.beta_prediction,
        beta_reason=model_args.beta_reason,
        lambda_rat=model_args.lambda_rat,
        num_chunk_tokens=model_args.num_chunk_tokens,
        adapter_heads=model_args.adapter_heads,
        encoder_max_length=model_args.encoder_max_length,
        imagebind_variant=model_args.imagebind_variant,
        num_image_tokens=model_args.num_image_tokens,
        group_layers=model_args.group_layers,
        group_heads=model_args.group_heads,
        group_tau=model_args.group_tau,
        open_roberta=model_args.open_roberta,
        open_vpma=model_args.open_vpma,
        open_imagebind=model_args.open_imagebind,
        open_image_projector=model_args.open_image_projector,
        freeze_decoder=model_args.freeze_decoder,
        use_mixlora=bool(model_args.use_mixlora),
        mixlora_cfg_overrides=mix_cfg_overrides,
    )

    if model_args.full_model_load_dir:
        if is_main:
            print("Loading weights via load_full_state_dict():", model_args.full_model_load_dir)
        model.load_full_state_dict(model_args.full_model_load_dir, cast_to_target_dtype=True)

    # sync pad/eos
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
            model.decoder.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
            model.config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    assert_vocab_safe_no_resize(model, tok)

    model.to(device)
    model.eval()

    # multi-rank: broadcast rank0 weights to all ranks
    ddp_sync_weights_from_rank0(model, enabled=(sync_weights and world_size > 1))
    if world_size > 1:
        debug_check_weight_consistency_print(model, strict=strict_weights)

    collator = build_collator(tok, enc_tok, model_args, data_args)

    dec = getattr(model, "decoder", model)
    dec_emb = dec.get_input_embeddings()
    dev_ref, dtype_ref = dec_emb.weight.device, dec_emb.weight.dtype

    os.makedirs(training_args.output_dir, exist_ok=True)

    for test_file in data_args.test_files:
        base = os.path.basename(test_file).replace(".jsonl", "")
        out_dir = os.path.join(training_args.output_dir, base)
        os.makedirs(out_dir, exist_ok=True)

        if is_main:
            print("---- Running file:", test_file, "----")

        full_ds = load_test_dataset(test_file)

        if data_args.max_infer_samples is not None:
            full_ds = full_ds.select(range(min(len(full_ds), int(data_args.max_infer_samples))))

        if data_args.do_shuffle:
            full_ds = full_ds.shuffle(seed=int(training_args.seed))

        total_samples = len(full_ds)
        if total_samples == 0:
            if is_main:
                print("[warn] Empty test set, skip:", test_file)
            continue

        indices = [i for i in range(total_samples) if i % world_size == rank]
        test_ds = full_ds.select(indices)

        print(f"[rank{rank}] {base}: {len(test_ds)} samples (total={total_samples})")

        test_dl = DataLoader(
            test_ds,
            batch_size=max(1, int(training_args.per_device_eval_batch_size)),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(training_args.dataloader_num_workers),
            pin_memory=bool(training_args.dataloader_pin_memory),
        )

        task_type = infer_task_type(test_file)

        raw_records: List[Dict] = []
        sample_true_seqs: List[List[float]] = []
        sample_pred_seqs: List[List[float]] = []

        pbar = tqdm(total=total_samples, desc=base, disable=(not is_main))

        loss_printed = 0
        samples_printed = 0
        debug_first_batch_done = False

        for batch in test_dl:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            B = int(batch["prefix_ids"].size(0))

            # debug: first batch per file
            if debug and is_main and (not debug_first_batch_done):
                debug_print_chunk_len(enc_tok, batch)

            # optional debug loss (teacher forcing)
            if debug and is_main and loss_printed < debug_loss_steps:
                with torch.no_grad():
                    out = model(**batch)
                    if isinstance(out, dict) and "loss" in out:
                        lv = float(out["loss"].detach().item())
                        extras = []
                        for kk in ("loss_understand", "loss_prediction", "loss_reason_label", "loss_reason_think"):
                            if kk in out:
                                extras.append(f"{kk}={float(out[kk].detach().item()):.4f}")
                        print("[debug] TF loss=", lv, (" ".join(extras) if extras else ""))
                    else:
                        try:
                            lv = float(out.loss.detach().item())
                            print("[debug] TF loss=", lv)
                        except Exception:
                            print("[debug] TF loss: unknown output")
                loss_printed += 1

            with torch.no_grad():
                chunk_embeds, chunk_mask, image_embeds, image_mask, *rest = model.encode_modalities(
                    chunk_txts_all=batch.get("chunk_txts", None),
                    pixel_values=batch.get("pixel_values", None),
                    n_images=batch.get("n_images", None),
                    batch_size=B,
                    device=dev_ref,
                    dtype=dtype_ref,
                )

                if debug and is_main and (not debug_first_batch_done):
                    print("[debug] chunk_mask sum=", int(chunk_mask.sum().item()), "shape=", tuple(chunk_mask.shape),
                          "| image_mask sum=", int(image_mask.sum().item()), "shape=", tuple(image_mask.shape))
                    debug_first_batch_done = True

                gen_only_ids = model.generate_with_modalities_cached(
                    prefix_ids=batch["prefix_ids"],
                    prefix_mask=batch["prefix_mask"],
                    suffix_ids=batch["suffix_ids"],
                    suffix_mask=batch["suffix_mask"],
                    chunk_embeds=chunk_embeds,
                    chunk_mask=chunk_mask,
                    image_embeds=image_embeds,
                    image_mask=image_mask,
                    task_ids=batch.get("task_ids", None),
                    max_new_tokens=int(data_args.gen_max_new_tokens),
                    temperature=float(data_args.gen_temperature),
                    top_k=int(data_args.gen_top_k),
                    top_p=float(data_args.gen_top_p),
                    do_sample=bool(data_args.gen_do_sample),
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            pred_texts = safe_batch_decode(tok, gen_only_ids, name="gen_only_ids")

            labels_ids = batch["labels"].detach().clone()
            labels_ids[labels_ids == -100] = tok.pad_token_id
            target_texts = safe_batch_decode(tok, labels_ids, name="labels_ids")

            for true_text, pred_text in zip(target_texts, pred_texts):
                vals_true = extract_numbers(true_text)
                vals_pred = extract_numbers(pred_text)

                raw_records.append(
                    {
                        "target_text": true_text,
                        "pred_text": pred_text,
                        "target_values": vals_true,
                        "pred_values": vals_pred,
                        "task_type": task_type,
                    }
                )

                if debug and is_main and samples_printed < debug_print_samples:
                    print("[sample] T:", (true_text[:200] + ("..." if len(true_text) > 200 else "")))
                    print("[sample] P:", (pred_text[:200] + ("..." if len(pred_text) > 200 else "")))
                    print("[sample] tv:", vals_true[:12], "pv:", vals_pred[:12])
                    print("-" * 60)
                    samples_printed += 1

                if len(vals_true) == 0 or len(vals_pred) == 0:
                    continue
                L = min(len(vals_true), len(vals_pred))
                sample_true_seqs.append(vals_true[:L])
                sample_pred_seqs.append(vals_pred[:L])

            if is_main:
                inc = B * world_size
                remaining = total_samples - pbar.n
                if remaining > 0:
                    pbar.update(min(inc, remaining))

        pbar.close()

        raw_path_rank = os.path.join(out_dir, f"raw_{base}_rank{rank}.jsonl")
        with open(raw_path_rank, "w", encoding="utf-8") as f:
            for rec in raw_records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        print(f"[rank{rank}] saved raw -> {raw_path_rank}")

        merged_true_seqs = sample_true_seqs
        merged_pred_seqs = sample_pred_seqs

        if world_size > 1:
            import torch.distributed as dist
            obj_local = (sample_true_seqs, sample_pred_seqs)
            obj_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj_list, obj_local)

            if rank == 0:
                merged_true_seqs, merged_pred_seqs = [], []
                for t_list, p_list in obj_list:
                    if not t_list or not p_list:
                        continue
                    merged_true_seqs.extend(t_list)
                    merged_pred_seqs.extend(p_list)

            dist.barrier()

        if rank == 0:
            if not merged_true_seqs:
                metrics = empty_metrics(mape_eps=float(data_args.mape_eps))
            else:
                T = min(len(s) for s in merged_true_seqs)
                labels_arr = np.asarray([s[:T] for s in merged_true_seqs], dtype=np.float64)
                preds_arr = np.asarray([s[:T] for s in merged_pred_seqs], dtype=np.float64)

                np.savetxt(os.path.join(out_dir, f"labels_{base}.txt"), labels_arr, fmt="%.8f")
                np.savetxt(os.path.join(out_dir, f"preds_{base}.txt"), preds_arr, fmt="%.8f")

                metrics = compute_metrics_np(labels_arr, preds_arr, mape_eps=float(data_args.mape_eps))

            metrics_path = os.path.join(out_dir, f"metrics_{base}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print("[rank0] metrics saved ->", metrics_path, "|", metrics)

        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()

    if is_main:
        print("==== ALL FILES DONE ====")


if __name__ == "__main__":
    main()
