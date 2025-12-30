# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from traffic_service.schemas import Plan
from traffic_service.core.context import ExecutionContext

logger = logging.getLogger(__name__)


def _get_by_path(obj: Any, path: str) -> Any:
    parts = path.split(".")
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            cur = getattr(cur, p)
    return cur


def resolve_refs(value: Any, ctx: ExecutionContext) -> Any:
    # {"$ref": "key"} or {"$ref":"key.field"}
    if isinstance(value, dict) and "$ref" in value and len(value) == 1:
        ref = str(value["$ref"])
        base_key = ref.split(".", 1)[0]
        base = ctx.get(base_key)
        if "." in ref:
            return _get_by_path(base, ref.split(".", 1)[1])
        return base

    # legacy "${key}" style
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        ref = value[2:-1].strip()
        base_key = ref.split(".", 1)[0]
        base = ctx.get(base_key)
        if "." in ref:
            return _get_by_path(base, ref.split(".", 1)[1])
        return base

    if isinstance(value, dict):
        return {k: resolve_refs(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, ctx) for v in value]
    return value


def topo_sort(steps: List[PlanStep]) -> List[PlanStep]:
    by_id: Dict[str, PlanStep] = {s.step_id: s for s in steps}
    indeg: Dict[str, int] = {s.step_id: 0 for s in steps}
    adj: Dict[str, Set[str]] = {s.step_id: set() for s in steps}

    for s in steps:
        deps = list(s.depends_on or [])
        for dep in deps:
            if dep not in by_id:
                raise ValueError(f"Step {s.step_id} depends_on unknown step_id: {dep}")
            adj[dep].add(s.step_id)
            indeg[s.step_id] += 1

    q = [sid for sid, d in indeg.items() if d == 0]
    out: List[PlanStep] = []
    while q:
        sid = q.pop(0)
        out.append(by_id[sid])
        for nxt in adj[sid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(out) != len(steps):
        remaining = [s.step_id for s in steps if s.step_id not in {x.step_id for x in out}]
        raise ValueError(f"Plan has a cycle. remaining={remaining}")
    return out


class Executor:
    def __init__(self, agents: Dict[str, Any]) -> None:
        self.agents = agents

    async def execute(self, plan: Plan, ctx: ExecutionContext) -> ExecutionContext:
        done = set()

        while len(done) < len(plan.steps):
            progressed = False

            for step in plan.steps:
                if step.step_id in done:
                    continue
                if any(dep not in done for dep in step.depends_on):
                    continue

                progressed = True

                if step.agent not in self.agents:
                    raise KeyError(f"Unknown agent '{step.agent}'. Available={list(self.agents.keys())}")

                agent = self.agents[step.agent]

                # Resolve $ref / ${} in step args.
                resolved_args = resolve_refs(step.args, ctx)
                if resolved_args is not step.args:
                    if hasattr(step, "model_copy"):      # pydantic v2
                        step = step.model_copy(update={"args": resolved_args})
                    elif hasattr(step, "copy"):          # pydantic v1
                        step = step.copy(update={"args": resolved_args})
                    else:
                        step.args = resolved_args

                t0 = time.perf_counter()
                logger.info(
                    f"STEP start | id={step.step_id} agent={step.agent} action={step.action} "
                    f"deps={step.depends_on} -> out={step.output_key}"
                )

                resolved_args = resolve_refs(step.args, ctx)

                # Replace step.args if resolved (pydantic v1/v2).
                if resolved_args is not step.args:
                    if hasattr(step, "model_copy"):       # pydantic v2
                        step = step.model_copy(update={"args": resolved_args})
                    elif hasattr(step, "copy"):           # pydantic v1
                        step = step.copy(update={"args": resolved_args})
                    else:
                        step.args = resolved_args

                try:
                    result = await agent.run(step, ctx)
                    ctx.set(step.output_key, result)

                    dt = (time.perf_counter() - t0) * 1000.0
                    logger.info(f"STEP ok    | id={step.step_id} ms={dt:.1f} result={_summarize_result(result)}")
                
                    soft_fail = bool((step.args or {}).get("soft_fail", False))

                except Exception as e:
                    dt = (time.perf_counter() - t0) * 1000.0
                    soft_fail = bool((step.args or {}).get("soft_fail", False))

                    if soft_fail:
                        logger.warning(
                            f"STEP soft-fail | id={step.step_id} ms={dt:.1f} err={type(e).__name__}: {e}"
                        )
                        ctx.set(step.output_key, None)     # Fill with None on soft-fail.
                    else:
                        logger.exception(
                            f"STEP fail  | id={step.step_id} ms={dt:.1f} err={type(e).__name__}: {e}"
                        )
                        raise

                done.add(step.step_id)

            if not progressed:
                remaining = [(s.step_id, s.depends_on) for s in plan.steps if s.step_id not in done]
                raise RuntimeError(f"Plan stuck (cycle or missing deps): {remaining}")

        return ctx

def _summarize_result(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, dict):
        keys = list(x.keys())[:8]
        return f"dict(keys={keys}, size={len(x)})"
    if isinstance(x, list):
        return f"list(len={len(x)})"
    t = type(x).__name__
    return t
