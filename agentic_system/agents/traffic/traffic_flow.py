# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentic_system.agents.base import ActionAgent
from agentic_system.core.context import ExecutionContext
from agentic_system.db.mysql_pool import get_conn
from agentic_system.utils.time import parse_local_datetime


def _fetch_flow_year(sensor_idx: int, year: int) -> Optional[Dict[str, Any]]:
    cnx = get_conn()
    try:
        cur = cnx.cursor(dictionary=True)
        cur.execute(
            """
            SELECT sensor_idx, year, freq_min, start_local, n_steps, values_zlib
            FROM sensor_flow_year
            WHERE sensor_idx=%s AND year=%s
            """,
            (int(sensor_idx), int(year)),
        )
        row = cur.fetchone()
        cur.close()
        return row
    finally:
        cnx.close()


def _to_datetime_naive(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x.replace(tzinfo=None)
    if isinstance(x, str) and x.strip():
        s = x.strip().replace("T", " ")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass
        try:
            return parse_local_datetime(s[:16])
        except Exception:
            return datetime(int(s[0:4]), 1, 1, 0, 0)
    return datetime(1970, 1, 1, 0, 0)


def _as_bytes(b: Any) -> bytes:
    if b is None:
        return b""
    if isinstance(b, bytes):
        return b
    if isinstance(b, bytearray):
        return bytes(b)
    try:
        return bytes(b)
    except Exception:
        return b""


class TrafficFlowAgent(ActionAgent):
    name = "TrafficFlowAgent"

    def __init__(self, adj_path: Optional[str] = None) -> None:
        self.adj_path = self._pick_adj_path(adj_path)
        self.adj = np.load(self.adj_path, mmap_mode="r")
        if self.adj.ndim != 2 or self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError(f"adj must be square, got {self.adj.shape}")

        self._sensor_code_map_loaded = False
        self._sensor_code_by_idx: Dict[int, str] = {}

        self._flow_row_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._avg_cache: Dict[Tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _pick_adj_path(adj_path: Optional[str]) -> str:
        if adj_path:
            p = Path(adj_path)
            if p.exists():
                return str(p)

        try:
            from agentic_system.config import settings
            p2 = Path(getattr(settings, "gla_adj_path", "") or "")
            if p2.exists():
                return str(p2)
        except Exception:
            pass

        pkg_root = Path(__file__).resolve().parents[2]  # .../agentic_system
        cands = [
            pkg_root / "dataset" / "gla_rn_adj.npy",
            pkg_root / "db" / "dataset" / "gla_rn_adj.npy",
        ]
        for p in cands:
            if p.exists():
                return str(p)

        raise FileNotFoundError("Cannot find gla_rn_adj.npy. Provide settings.gla_adj_path or adj_path=...")

    async def _ensure_sensor_code_map(self) -> None:
        if self._sensor_code_map_loaded:
            return
        # Reuse DB: fetch from sensors table.
        cnx = get_conn()
        try:
            cur = cnx.cursor(dictionary=True)
            cur.execute("SELECT sensor_idx, sensor_code FROM sensors")
            rows = cur.fetchall() or []
            cur.close()
        finally:
            cnx.close()

        m: Dict[int, str] = {}
        for r in rows:
            idx = int(r["sensor_idx"])
            code = r.get("sensor_code")
            m[idx] = str(code) if code is not None else str(idx)

        self._sensor_code_by_idx = m
        self._sensor_code_map_loaded = True

    def _neighbors_1hop(self, sensor_idx: int, topk: int = 8, eps: float = 1e-12) -> List[int]:
        row = np.asarray(self.adj[sensor_idx])
        out_idx = np.where(row > eps)[0].astype(int)
        out_idx = out_idx[out_idx != sensor_idx]

        col = np.asarray(self.adj[:, sensor_idx])
        in_idx = np.where(col > eps)[0].astype(int)
        in_idx = in_idx[in_idx != sensor_idx]

        cand = np.union1d(out_idx, in_idx).astype(int)
        if cand.size == 0:
            return []

        w = np.maximum(row[cand], col[cand])
        order = np.argsort(w)[::-1][: int(topk)]
        return cand[order].tolist()

    async def _get_flow_row(self, sensor_idx: int, year: int) -> Optional[Dict[str, Any]]:
        key = (int(sensor_idx), int(year))
        if key in self._flow_row_cache:
            return self._flow_row_cache[key]
        row = await asyncio.to_thread(_fetch_flow_year, int(sensor_idx), int(year))
        if row:
            self._flow_row_cache[key] = row
        return row

    @staticmethod
    def _slice_indices(series_start: datetime, freq_min: int, n_steps: int, start_dt: datetime, end_dt: datetime) -> Tuple[int, int]:
        step_s = int(freq_min) * 60
        i0 = int((start_dt - series_start).total_seconds() // step_s)
        i1 = int((end_dt - series_start).total_seconds() // step_s) + 1
        i0 = max(i0, 0)
        i1 = min(i1, int(n_steps))
        return i0, i1

    @staticmethod
    def _flow_to_list(arr: np.ndarray) -> List[float]:
        out = arr.astype(np.float32).tolist()
        return [float(round(x, 3)) for x in out]

    def _compute_weekday_slot_mean(self, *, arr: np.ndarray, series_start: datetime, freq_min: int) -> np.ndarray:
        steps_per_day = int(1440 // freq_min)
        n_steps = int(arr.shape[0])

        start_offset_steps = int((series_start.hour * 60 + series_start.minute) // freq_min)
        start_weekday = int(series_start.weekday())

        i = np.arange(n_steps, dtype=np.int64)
        total = start_offset_steps + i
        day_index = total // steps_per_day
        slot = total % steps_per_day
        weekday = (start_weekday + day_index) % 7

        sums = np.zeros((7, steps_per_day), dtype=np.float64)
        cnts = np.zeros((7, steps_per_day), dtype=np.int64)

        for w in range(7):
            mask = (weekday == w)
            if not np.any(mask):
                continue
            s = np.bincount(slot[mask], weights=arr[mask].astype(np.float64), minlength=steps_per_day)
            c = np.bincount(slot[mask], minlength=steps_per_day)
            sums[w, :] = s
            cnts[w, :] = c

        mean = np.zeros((7, steps_per_day), dtype=np.float32)
        for w in range(7):
            c = cnts[w]
            m = np.zeros(steps_per_day, dtype=np.float32)
            nz = c > 0
            if np.any(nz):
                m[nz] = (sums[w][nz] / c[nz]).astype(np.float32)
            mean[w] = m
        return mean

    def _history_avg_for_window(
        self,
        *,
        avg_mat: np.ndarray,
        series_start: datetime,
        freq_min: int,
        i0: int,
        i1: int
    ) -> List[float]:
        steps_per_day = int(1440 // freq_min)
        start_offset_steps = int((series_start.hour * 60 + series_start.minute) // freq_min)
        start_weekday = int(series_start.weekday())

        idx = np.arange(i0, i1, dtype=np.int64)
        total = start_offset_steps + idx
        day_index = total // steps_per_day
        slot = total % steps_per_day
        weekday = (start_weekday + day_index) % 7

        vals = avg_mat[weekday, slot]
        return self._flow_to_list(vals)

    async def _get_series_and_history_for_window(self, sensor_idx: int, start_dt: datetime, end_dt: datetime) -> Tuple[List[float], List[float]]:
        years = list(range(int(start_dt.year), int(end_dt.year) + 1))
        flow_out: List[float] = []
        hist_out: List[float] = []

        for y in years:
            row = await self._get_flow_row(sensor_idx, y)
            if not row:
                continue

            freq_min = int(row["freq_min"])
            n_steps = int(row["n_steps"])
            series_start = _to_datetime_naive(row.get("start_local"))

            blob = _as_bytes(row.get("values_zlib"))
            if not blob:
                continue

            dec = zlib.decompress(blob)
            arr = np.frombuffer(dec, dtype=np.float32)
            if arr.shape[0] != n_steps:
                n_steps = int(arr.shape[0])

            seg_start = max(start_dt, series_start)
            seg_end = min(end_dt, series_start + timedelta(minutes=freq_min * (n_steps - 1)))

            i0, i1 = self._slice_indices(series_start, freq_min, n_steps, seg_start, seg_end)
            if i1 <= i0:
                continue

            flow_out.extend(self._flow_to_list(arr[i0:i1]))

            key = (int(sensor_idx), int(y))
            if key not in self._avg_cache:
                self._avg_cache[key] = self._compute_weekday_slot_mean(arr=arr, series_start=series_start, freq_min=freq_min)
            avg_mat = self._avg_cache[key]
            hist_out.extend(self._history_avg_for_window(avg_mat=avg_mat, series_start=series_start, freq_min=freq_min, i0=i0, i1=i1))

        return flow_out, hist_out

    async def fetch(self, ctx: ExecutionContext, *, match_key: str = "nearest_sensor", topk: int = 8) -> Dict[str, Any]:
        await self._ensure_sensor_code_map()

        m = ctx.require(match_key)
        if not isinstance(m, dict) or "sensor_idx" not in m:
            raise RuntimeError("TrafficFlowAgent requires nearest_sensor output with sensor_idx")

        sensor_idx = int(m["sensor_idx"])
        if ctx.request.local_start is None or ctx.request.local_end is None:
            raise RuntimeError("TrafficFlowAgent requires ctx.request.local_start/local_end")

        start_dt = parse_local_datetime(ctx.request.local_start)
        end_dt = parse_local_datetime(ctx.request.local_end)

        main_flow, main_hist = await self._get_series_and_history_for_window(sensor_idx, start_dt, end_dt)

        neighbors = self._neighbors_1hop(sensor_idx, topk=int(topk))
        hop_lines: List[str] = []
        for nb in neighbors:
            nb_flow, _ = await self._get_series_and_history_for_window(nb, start_dt, end_dt)
            nb_code = self._sensor_code_by_idx.get(int(nb), str(nb))
            hop_lines.append(f"{nb_code}: {nb_flow}")

        ctx.update_request(
            {
                "traffic_flow": main_flow if getattr(ctx.request, "traffic_flow", None) is None else None,
                "history_average_flow": main_hist if getattr(ctx.request, "history_average_flow", None) is None else None,
            }
        )

        return {
            "sensor_idx": sensor_idx,
            "sensor_id": self._sensor_code_by_idx.get(sensor_idx, str(sensor_idx)),
            "neighbors": neighbors,
            "main_flow_len": len(main_flow),
            "history_avg_len": len(main_hist),
            "hop_sensor_lines": hop_lines,
        }
