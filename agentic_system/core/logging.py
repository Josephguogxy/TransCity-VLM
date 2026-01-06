# Copyright (c) 2024 torchtorch Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import sys


def setup_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    h = logging.StreamHandler(sys.stderr)
    h.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(fmt)
    root.addHandler(h)
