# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import math, random, collections, os
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler


class PRBalancedBatchSampler(BatchSampler):
    """
    Three tasks: U (understand) / P (prediction) / R (reason)

    Design goal (plan B):
    - Each batch contains exactly two tasks: U+P or U+R; never mix P+R in the same batch.
    - Intended to work with MixLoRA task routing:
        * understand: update the Universal expert (U) only
        * prediction: update U ∪ P
        * reason:     update U ∪ R
    - The U-to-X (P/R) ratio is fixed per batch:
        pr_per_batch = (u_cnt, x_cnt), with u_cnt + x_cnt == batch_size
        * If pr_per_batch is None, use defaults:
            bs=2 -> 1U + 1X
            bs=4 -> 3U + 1X
            bs=8 -> 6U + 2X
            otherwise: default to 3/4 U and 1/4 X (floor)

    Other features:
    - DDP-friendly: if a DistributedSampler is provided, sampling happens only within the current rank's shard.
    - epoch_batches: cap the maximum number of batches per epoch.
    - minority_repeat_cap: reserved parameter (currently unused in the 3-task mode).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        pr_per_batch: Optional[Tuple[int, int]] = None,   # In 3-task mode, interpret as (U, X) counts
        sampler: Optional[Sampler[int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        epoch_batches: Optional[int] = None,
        minority_repeat_cap: Optional[int] = None,        # Reserved (currently unused in 3-task mode)
        task_field: str = "task",
        label_pred: str = "prediction",
        label_reason: str = "reason",
        label_understand: str = "understand",
    ):
        assert batch_size > 0, "batch_size must be > 0"
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.sampler = sampler
        self.seed = int(seed)

        # === Key: U/X ratio ===
        if pr_per_batch is None:
            if self.bs == 2:
                u, x = 1, 1
            elif self.bs == 4:
                u, x = 3, 1
            elif self.bs == 8:
                u, x = 6, 2
            else:
                u = max(1, (3 * self.bs) // 4)
                x = self.bs - u
        else:
            u, x = pr_per_batch
        assert u >= 0 and x >= 0 and (u + x) == self.bs, "pr_per_batch invalid (expect u_cnt + x_cnt == batch_size)"
        self.u_per_batch = int(u)
        self.x_per_batch = int(x)

        # Other configs
        self.epoch_batches = None if epoch_batches is None else int(epoch_batches)
        self.minority_repeat_cap = None if minority_repeat_cap is None else int(minority_repeat_cap)
        self.task_field = task_field
        self.label_pred = label_pred
        self.label_reason = label_reason
        self.label_understand = label_understand

        # Read task labels
        self.task_col: List[str] = dataset[self.task_field]
        self.epoch: int = 0
        self._cached_batches: Optional[List[List[int]]] = None

    # ---------- public ----------
    def __iter__(self):
        if self._cached_batches is None:
            self._rebuild_for_epoch(self.epoch)
        for b in self._cached_batches:
            yield b
        self.epoch += 1
        self._cached_batches = None

    def __len__(self) -> int:
        if self._cached_batches is None:
            self._rebuild_for_epoch(self.epoch)
        return len(self._cached_batches)

    # ---------- core ----------
    def _rebuild_for_epoch(self, epoch: int):
        rnd = random.Random(self.seed + epoch)

        # Per-rank indices (sharded by DistributedSampler first)
        if self.sampler is not None:
            if hasattr(self.sampler, "set_epoch"):
                try:
                    self.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
                except Exception:
                    pass
            all_indices = list(self.sampler)
        else:
            all_indices = list(range(len(self.ds)))
            if self.shuffle:
                rnd.shuffle(all_indices)

        # Bucket into U / P / R
        idx_u: List[int] = []
        idx_p: List[int] = []
        idx_r: List[int] = []
        for i in all_indices:
            t = self.task_col[i]
            if t == self.label_understand:
                idx_u.append(i)
            elif t == self.label_pred:
                idx_p.append(i)
            elif t == self.label_reason:
                idx_r.append(i)
            # Ignore other labels

        if self.shuffle:
            rnd.shuffle(idx_u)
            rnd.shuffle(idx_p)
            rnd.shuffle(idx_r)

        batches: List[List[int]] = []

        # Only U available (e.g., Stage-A) -> all-U batches
        if (len(idx_p) + len(idx_r)) == 0:
            batches = self._make_all_u_batches(idx_u, rnd)
            self._sync_and_set(batches)
            return

        # Normal case: build U+P batches and U+R batches, then shuffle together
        batches_up = self._pair_with_u(
            idx_u, idx_p, rnd,
            need_u=self.u_per_batch,
            need_x=self.x_per_batch,
            key_prefix="p"
        )
        batches_ur = self._pair_with_u(
            idx_u, idx_r, rnd,
            need_u=self.u_per_batch,
            need_x=self.x_per_batch,
            key_prefix="r"
        )

        all_batches = batches_up + batches_ur
        if self.shuffle:
            rnd.shuffle(all_batches)

        # Per-epoch batch cap (if set)
        if self.epoch_batches is not None and len(all_batches) > self.epoch_batches:
            all_batches = all_batches[: self.epoch_batches]

        self._sync_and_set(all_batches)

    # ---------- helpers ----------
    def _cycle_take(self, pool: List[int], k: int, state: dict, *, key: str, rnd: random.Random) -> List[int]:
        """
        Cycle through ``pool`` to take ``k`` items (with replacement), continuing from the last position.

        ``state[key]`` stores the current position; if missing, start from a random position to add randomness.
        """
        if not pool or k <= 0:
            return []
        n = len(pool)
        pos = state.get(key)
        if pos is None:
            pos = rnd.randrange(n) if self.shuffle else 0
        out: List[int] = []
        for _ in range(k):
            out.append(pool[pos])
            pos = (pos + 1) % n
        state[key] = pos
        return out

    def _pair_with_u(
        self,
        idx_u: List[int],
        idx_x: List[int],
        rnd: random.Random,
        *,
        need_u: int,
        need_x: int,
        key_prefix: str,
    ) -> List[List[int]]:
        """
        Build batches of (U + X) with size ``need_u + need_x``:

        - X ∈ {P, R}
        - If X is exhausted, oversample cyclically
        - If U is insufficient, oversample cyclically as well
        - If drop_last=True, an incomplete batch is dropped
        """
        batches: List[List[int]] = []
        if need_u + need_x != self.bs:
            raise AssertionError("Internal constraint: need_u + need_x must equal batch_size")

        if not idx_x:
            # No samples for this X class; cannot form U+X batches
            return batches

        state: Dict[str, int] = {}
        # Approximate number of batches: ceil(len(X)/need_x)
        x_batches = (len(idx_x) + need_x - 1) // need_x if need_x > 0 else 0

        for b in range(x_batches):
            # Take X
            x_start = b * need_x
            cur_x = idx_x[x_start: x_start + need_x]
            if len(cur_x) < need_x and len(idx_x) > 0:
                # If not enough, oversample cyclically
                cur_x += self._cycle_take(
                    idx_x,
                    need_x - len(cur_x),
                    state,
                    key=f"{key_prefix}_x",
                    rnd=rnd,
                )

            # Take U (cyclic oversampling allowed)
            cur_u = self._cycle_take(idx_u, need_u, state, key=f"{key_prefix}_u", rnd=rnd)

            batch = cur_u + cur_x
            if self.shuffle:
                rnd.shuffle(batch)

            # If still short and drop_last=False, fill using U or X
            if len(batch) < self.bs and not self.drop_last:
                remain = self.bs - len(batch)
                fill_pool = idx_u or idx_x
                batch += self._cycle_take(fill_pool, remain, state, key=f"{key_prefix}_fill", rnd=rnd)

            if len(batch) == self.bs or (len(batch) > 0 and not self.drop_last):
                batches.append(batch)

        return batches

    def _make_all_u_batches(self, idx_u: List[int], rnd: random.Random) -> List[List[int]]:
        """
        U-only scenario: pack by ``batch_size``. If the last batch is incomplete:
        - drop_last=True: drop it
        - drop_last=False: fill it by cycling through U
        """
        batches: List[List[int]] = []
        if not idx_u:
            return batches

        state: Dict[str, int] = {}
        for s in range(0, len(idx_u), self.bs):
            cur = idx_u[s: s + self.bs]
            if len(cur) < self.bs:
                if self.drop_last:
                    break
                need = self.bs - len(cur)
                cur += self._cycle_take(idx_u, need, state, key="u_only", rnd=rnd)
            if self.shuffle:
                rnd.shuffle(cur)
            batches.append(cur)
        return batches

    def _sync_and_set(self, batches: List[List[int]]):
        # Only do all_reduce when distributed is initialized
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")

            local_n = torch.tensor([len(batches)], device=device, dtype=torch.long)

            rk = dist.get_rank()
            print(f"[rank {rk}] before all_reduce, local_batches={len(batches)}", flush=True)

            # Use the minimum batch count across all ranks
            dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
            global_min = int(local_n.item())

            print(f"[rank {rk}] after all_reduce, global_min={global_min}", flush=True)

            if global_min < len(batches):
                batches = batches[:global_min]
        else:
            # Non-distributed or dist not initialized: use local batches
            print(f"[sampler] dist not initialized, local_batches={len(batches)}", flush=True)

        self._cached_batches = batches


