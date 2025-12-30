# traffic_service/agents/base.py
from __future__ import annotations

import inspect
from typing import Any, Dict

from traffic_service.core.context import ExecutionContext
from traffic_service.schemas import PlanStep


class ActionAgent:
    """
    Dispatcher: Executor calls agent.run(step, ctx), then dispatches to step.action.
    """

    name: str = "ActionAgent"

    async def run(self, step: PlanStep, ctx: ExecutionContext) -> Any:
        action = str(step.action or "").strip()
        if not action:
            raise ValueError(f"{self.__class__.__name__}: empty step.action")

        fn = getattr(self, action, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"{self.__class__.__name__} has no action method '{action}'. "
                f"Available={[m for m in dir(self) if not m.startswith('_')]}"
            )

        args: Dict[str, Any] = dict(step.args or {})

        # Drop args the method does not accept.
        sig = inspect.signature(fn)
        params = sig.parameters
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        call_kwargs: Dict[str, Any] = {}
        for k, v in args.items():
            if has_kwargs or k in params:
                call_kwargs[k] = v

        # Convention: action methods take ctx as the first argument.
        return await fn(ctx, **call_kwargs)


def _pick_float(v: Any, fallback: Any) -> float:
    x = v if v is not None else fallback
    if x is None:
        raise ValueError("missing float")
    return float(x)
