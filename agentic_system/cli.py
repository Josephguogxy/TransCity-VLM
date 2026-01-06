# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentic_system.core.runner import QuestionRunner, RunOptions
from agentic_system.core.logging import setup_logging

async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--question", type=str, required=True)

    p.add_argument("--no-flow", dest="with_flow", action="store_false", default=True)
    p.add_argument("--no-satellite", dest="with_satellite", action="store_false", default=True)
    p.add_argument("--image-store", choices=["base64", "file", "none"], default="file")

    p.add_argument("--print-prompt", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=str, default=None)

    args = p.parse_args()
    setup_logging(verbose=bool(args.verbose))
    runner = QuestionRunner()
    opts = RunOptions(
        with_flow=bool(args.with_flow),
        with_satellite=bool(args.with_satellite),
        image_store=str(args.image_store),
        print_prompt=bool(args.print_prompt),
    )

    payload = await runner.run_question(args.question, opts=opts)

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
