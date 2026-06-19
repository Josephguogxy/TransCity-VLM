from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import math, random, collections, os
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Sampler


class PRBalancedBatchSampler(BatchSampler):
    """
    三类任务：U(understand) / P(prediction) / R(reason)

    设计目标（路线 B）：
    - 每个 batch 只包含两类：U+P 或 U+R，不出现 P+R 混在同一 batch；
    - 任务路由配合 MixLoRA：
        * understand: 只更新 Universal 专家(U)
        * prediction: 更新 U ∪ P
        * reason:     更新 U ∪ R
    - U 与 X(P/R) 的数量按固定配比：
        pr_per_batch = (u_cnt, x_cnt)，要求 u_cnt + x_cnt == batch_size
        * 若 pr_per_batch=None，则：
            bs=2 -> 1U + 1X
            bs=4 -> 3U + 1X
            bs=8 -> 6U + 2X
            其它 batch_size：默认 U 占 3/4，X 占 1/4（向下取整）

    其它特性：
    - 兼容 DDP：若传入 DistributedSampler，则只在当前 rank 的子集上做采样；
    - epoch_batches: 限制每个 epoch 的最大 batch 数；
    - minority_repeat_cap: 预留参数，目前在 3-task 模式下暂不使用；
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        pr_per_batch: Optional[Tuple[int, int]] = None,   # 在 3-task 模式下解释为 (U, X) 数量
        sampler: Optional[Sampler[int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        epoch_batches: Optional[int] = None,
        minority_repeat_cap: Optional[int] = None,        # 预留：当前 3-task 实现未使用
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

        # === 关键：U/X 配比 ===
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

        # 其它配置
        self.epoch_batches = None if epoch_batches is None else int(epoch_batches)
        self.minority_repeat_cap = None if minority_repeat_cap is None else int(minority_repeat_cap)
        self.task_field = task_field
        self.label_pred = label_pred
        self.label_reason = label_reason
        self.label_understand = label_understand

        # 读出标签列
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

        # rank 内样本索引（先由 DistributedSampler 分片）
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

        # 三类分桶：U / P / R
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
            # 其它标签忽略

        if self.shuffle:
            rnd.shuffle(idx_u)
            rnd.shuffle(idx_p)
            rnd.shuffle(idx_r)

        batches: List[List[int]] = []

        # 只有 U（例如 Stage-A）→ 全 U 批
        if (len(idx_p) + len(idx_r)) == 0:
            batches = self._make_all_u_batches(idx_u, rnd)
            self._sync_and_set(batches)
            return

        # 正常三类场景：生成 U+P 批 和 U+R 批，再打乱混在一起
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

        # 每个 epoch 的 batch 上限（若设置）
        if self.epoch_batches is not None and len(all_batches) > self.epoch_batches:
            all_batches = all_batches[: self.epoch_batches]

        self._sync_and_set(all_batches)

    # ---------- helpers ----------
    def _cycle_take(self, pool: List[int], k: int, state: dict, *, key: str, rnd: random.Random) -> List[int]:
        """
        从 pool 循环取 k 个（带回采样），每次从上次停留处继续。
        state[key] 保存当前位置；若不存在则设成随机起点以增加扰动。
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
        生成若干批（U + X），每批大小 need_u + need_x：
        - X ∈ {P, R}；
        - X 用尽后循环补齐（oversample）；
        - U 不足也会循环补齐；
        - drop_last=True 时不足一个 batch 会被丢弃。
        """
        batches: List[List[int]] = []
        if need_u + need_x != self.bs:
            raise AssertionError("内部约束：need_u + need_x 必须等于 batch_size")

        if not idx_x:
            # 没有这一类 X，则无法生成对应的 U+X 批
            return batches

        state: Dict[str, int] = {}
        # 能产生的批数 ≈ ceil(len(X)/need_x)
        x_batches = (len(idx_x) + need_x - 1) // need_x if need_x > 0 else 0

        for b in range(x_batches):
            # 取 X
            x_start = b * need_x
            cur_x = idx_x[x_start: x_start + need_x]
            if len(cur_x) < need_x and len(idx_x) > 0:
                # 不够就循环补齐
                cur_x += self._cycle_take(
                    idx_x,
                    need_x - len(cur_x),
                    state,
                    key=f"{key_prefix}_x",
                    rnd=rnd,
                )

            # 取 U（可循环）
            cur_u = self._cycle_take(idx_u, need_u, state, key=f"{key_prefix}_u", rnd=rnd)

            batch = cur_u + cur_x
            if self.shuffle:
                rnd.shuffle(batch)

            # 若还不满 batch_size，且 drop_last=False，则用 U 或 X 再补齐
            if len(batch) < self.bs and not self.drop_last:
                remain = self.bs - len(batch)
                fill_pool = idx_u or idx_x
                batch += self._cycle_take(fill_pool, remain, state, key=f"{key_prefix}_fill", rnd=rnd)

            if len(batch) == self.bs or (len(batch) > 0 and not self.drop_last):
                batches.append(batch)

        return batches

    def _make_all_u_batches(self, idx_u: List[int], rnd: random.Random) -> List[List[int]]:
        """
        只有 U 的场景：按 batch_size 直接打包；若 U 不够整除，
        drop_last=True 时丢弃尾批，drop_last=False 时循环补齐。
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
        # 只在分布式已初始化时做 all_reduce
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")

            local_n = torch.tensor([len(batches)], device=device, dtype=torch.long)

            rk = dist.get_rank()
            print(f"[rank {rk}] before all_reduce, local_batches={len(batches)}", flush=True)

            # 所有 rank 上取 batch 数量的最小值
            dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
            global_min = int(local_n.item())

            print(f"[rank {rk}] after all_reduce, global_min={global_min}", flush=True)

            if global_min < len(batches):
                batches = batches[:global_min]
        else:
            # 非分布式 / 分布式尚未初始化的情况，直接用本地 batches
            print(f"[sampler] dist not initialized, local_batches={len(batches)}", flush=True)

        self._cached_batches = batches
