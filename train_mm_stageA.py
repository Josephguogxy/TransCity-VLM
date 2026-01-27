# train_mm_stageA.py
from __future__ import annotations

import os
import sys
import math
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets as hf_datasets
from datasets import concatenate_datasets, Dataset as HFDataset

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)

from dataprocessing import (
    load_jsonl_dataset,
    ensure_smartcity_features,
    DataCollatorSmartCity,
)
from model.builder import build_smartcity_llm

logger = logging.getLogger(__name__)


# =========================
# helpers (rank / cache)
# =========================
def _rk_env() -> int:
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            pass
    return -1


def _is_rank0_env() -> bool:
    r = _rk_env()
    return r in (-1, 0)


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _wait_for_file(path: str, timeout_sec: int) -> None:
    t0 = time.time()
    while not os.path.exists(path):
        if (time.time() - t0) > timeout_sec:
            raise TimeoutError(f"wait_for_file timeout: {path}")
        time.sleep(1.0)


def _dataset_cache_key(files: List[str], split_name: str, seed: int, extra: dict) -> str:
    obj = {
        "files": [os.path.abspath(x) for x in files],
        "mtimes": [os.path.getmtime(x) if os.path.exists(x) else None for x in files],
        "split": split_name,
        "seed": int(seed),
        "extra": extra,
        "v": 2,
    }
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(s).hexdigest()


# =========================
# args
# =========================
@dataclass
class ModelArguments:
    decoder_name_or_path: str = field(metadata={"help": "LLM 解码器 (local path or HF id)"})
    encoder_name_or_path: str = field(default="FacebookAI/roberta-large", metadata={"help": "Chunk 文本编码器"})

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
    freeze_decoder: bool = True

    alpha_understand: float = 1.0
    beta_prediction: float = 0.0
    beta_reason: float = 0.0
    lambda_rat: float = 0.3

    padding_token: Optional[str] = field(
        default="<|pad|>",
        metadata={"help": "若 tokenizer 无 pad_token，将注册此符号为 pad"},
    )


@dataclass
class DataTrainingArguments:
    train_files: List[str] = field(default_factory=list)
    validation_files: List[str] = field(default_factory=list)

    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_length: int = 2048

    max_images_per_sample: int = 5
    image_size: int = 384
    max_chunks_per_sample: Optional[int] = None

    add_eos_to_labels: bool = True

    def __post_init__(self):
        for flist in (self.train_files, self.validation_files):
            for fp in flist:
                assert fp.endswith(".jsonl"), f"Expect *.jsonl data files, got {fp!r}"


# =========================
# trainer
# =========================
class TrafficTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        bs = self.args.per_device_train_batch_size
        dist_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
            seed=self.args.seed,
            drop_last=False,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=bs,
            sampler=dist_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )


# =========================
# dataset build (silent) + cache
# =========================
def _build_dataset_no_cache(files: List[str], split_name: str, seed: int, num_proc: int) -> HFDataset:
    assert len(files) > 0, f"No {split_name} files provided."
    dss = [load_jsonl_dataset(p, split_name=split_name) for p in files]
    ds = dss[0] if len(dss) == 1 else concatenate_datasets(dss)
    ds = ensure_smartcity_features(ds, num_proc=num_proc)
    ds = ds.shuffle(seed=seed)
    return ds


def _load_and_merge_cached(
    files: List[str],
    split_name: str,
    seed: int,
    cache_root: str,
) -> HFDataset:
    """
    默认：只让 rank0 构建并 save_to_disk，其它 rank 等待并 load_from_disk。
    环境变量：
      SMARTCITY_DATASET_CACHE=0  关闭缓存
      SMARTCITY_MAP_NUM_PROC=16  控制 ensure_smartcity_features 的 num_proc
      SMARTCITY_CACHE_ROOT=...   指定缓存根目录
      SMARTCITY_CACHE_WAIT_SEC=86400 等待超时
    """
    assert len(files) > 0, f"No {split_name} files provided."

    use_cache = os.environ.get("SMARTCITY_DATASET_CACHE", "1") == "1"
    num_proc = int(os.environ.get("SMARTCITY_MAP_NUM_PROC", "16"))
    cache_root = os.environ.get("SMARTCITY_CACHE_ROOT", cache_root)
    wait_sec = int(os.environ.get("SMARTCITY_CACHE_WAIT_SEC", str(24 * 3600)))

    if not use_cache:
        return _build_dataset_no_cache(files, split_name, seed, num_proc=num_proc)

    extra = {"num_proc": num_proc}
    key = _dataset_cache_key(files, split_name, seed, extra=extra)
    ds_dir = os.path.join(cache_root, f"{split_name}_{key}")
    done_flag = os.path.join(ds_dir, "_DONE")

    _safe_mkdir(cache_root)

    if _is_rank0_env():
        if not os.path.exists(done_flag):
            _safe_mkdir(ds_dir)
            ds = _build_dataset_no_cache(files, split_name, seed, num_proc=num_proc)
            ds.save_to_disk(ds_dir)
            tmp_flag = done_flag + ".tmp"
            with open(tmp_flag, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ok": True, "ts": time.time()}, ensure_ascii=False))
            os.replace(tmp_flag, done_flag)
    else:
        _wait_for_file(done_flag, timeout_sec=wait_sec)

    return hf_datasets.load_from_disk(ds_dir)


# =========================
# main
# =========================
def main():
    # datasets 的 save_to_disk / map 进度条太吵，直接关掉
    try:
        hf_datasets.disable_progress_bar()
    except Exception:
        pass

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    set_seed(training_args.seed)

    # -------- tokenizer --------
    tok = AutoTokenizer.from_pretrained(
        model_args.decoder_name_or_path,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    tok.padding_side = "left"

    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        pad_str = (model_args.padding_token or "<|pad|>")
        if pad_str not in tok.get_vocab():
            tok.add_special_tokens({"pad_token": pad_str})
        else:
            tok.pad_token = pad_str

    assert tok.eos_token_id is not None
    assert tok.pad_token_id != tok.eos_token_id

    # -------- model --------
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
    )

    # pad_token 同步
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            model.decoder.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "_cached_pad_id"):
            model._cached_pad_id = int(tok.pad_token_id)
    except Exception:
        pass

    try:
        model.resize_token_embeddings(len(tok))
    except Exception:
        pass

    if training_args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            setattr(model.decoder.config, "use_cache", False)

    # -------- datasets --------
    assert len(data_args.train_files) > 0, "请通过 --train_files 提供至少一个 *.jsonl"

    cache_root = os.path.join(training_args.output_dir or ".", ".dataset_cache")
    train_ds = _load_and_merge_cached(
        data_args.train_files, "train", training_args.seed, cache_root=cache_root
    )

    eval_ds = None
    if len(data_args.validation_files) > 0:
        eval_ds = _load_and_merge_cached(
            data_args.validation_files, "validation", training_args.seed, cache_root=cache_root
        )

    if data_args.max_train_samples:
        train_ds = train_ds.select(range(min(len(train_ds), data_args.max_train_samples)))
    if data_args.max_eval_samples and (eval_ds is not None):
        eval_ds = eval_ds.select(range(min(len(eval_ds), data_args.max_eval_samples)))

    if len(train_ds) == 0:
        raise ValueError("训练集为空，请检查 --train_files")

    if (eval_ds is None) or (len(eval_ds) == 0):
        training_args.evaluation_strategy = "no"
        logger.warning("评估集为空：已自动将 evaluation_strategy='no'。")

    # -------- collator --------
    collator = DataCollatorSmartCity(
        tok,
        max_length=data_args.max_length,
        pad_token_id=tok.pad_token_id,
        label_pad_token_id=-100,
        chunk_token_factor=model_args.num_chunk_tokens,
        image_token_factor=model_args.num_image_tokens,
        max_images_per_sample=data_args.max_images_per_sample,
        image_size=data_args.image_size,
        max_chunks_per_sample=data_args.max_chunks_per_sample,
        add_eos_to_labels=getattr(data_args, "add_eos_to_labels", True),
    )

    # -------- trainer --------
    trainer = TrafficTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if (training_args.do_eval and eval_ds is not None) else None,
        tokenizer=tok,
        data_collator=collator,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if trainer.is_world_process_zero():
            tok.save_pretrained(training_args.output_dir)

    if training_args.do_eval and (eval_ds is not None):
        metrics = trainer.evaluate()
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except (KeyError, OverflowError):
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()