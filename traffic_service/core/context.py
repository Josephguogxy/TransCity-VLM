# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Any, Protocol

from traffic_service.schemas import PlanStep


class Agent(Protocol):
    name: str
    async def run(self, step: PlanStep, ctx: "ExecutionContext") -> Any: ...


_MISSING = object()


class ExecutionContext:
    def __init__(self, request: Any):
        self.request = request
        self.outputs: dict[str, Any] = {}

    def get(self, key: str, default: Any = _MISSING) -> Any:
        if default is _MISSING:
            return self.outputs[key]
        return self.outputs.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.outputs:
            raise KeyError(f"Missing ctx.outputs['{key}']. Available={list(self.outputs.keys())}")
        return self.outputs[key]

    def set(self, key: str, value: Any) -> None:
        self.outputs[key] = value

    def update_request(self, updates: dict[str, Any], *, drop_none: bool = True) -> None:
        if not updates:
            return
        if drop_none:
            updates = {k: v for k, v in updates.items() if v is not None}
            if not updates:
                return
        if hasattr(self.request, "model_copy"):
            self.request = self.request.model_copy(update=updates)
        else:
            for k, v in updates.items():
                setattr(self.request, k, v)
