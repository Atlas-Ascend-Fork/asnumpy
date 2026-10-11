#!/usr/bin/env python3
# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""Benchmark public ndarray wrapping before and after the move-semantics change.

For the final performance conclusion, run this script against separately built
pre-change and post-change commits with identical hardware and software settings.
The ``core`` versus ``public`` modes are a same-build proxy for wrapper overhead.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import asnumpy as anp
from asnumpy import _core

logger = logging.getLogger("benchmark_move_wrapper")

Benchmark = Callable[[], object]


def _parse_shape(text: str) -> tuple[int, ...]:
    shape = tuple(int(part) for part in text.lower().split("x"))
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"invalid shape: {text!r}")
    return shape


def _percentile(samples: list[int], percentile: float) -> int:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _measure(operation: Benchmark, warmup: int, repeats: int) -> dict[str, float]:
    result: object | None = None
    for _ in range(warmup):
        result = operation()

    samples: list[int] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = operation()
            samples.append(time.perf_counter_ns() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
        del result

    return {
        "minimum_ms": min(samples) / 1_000_000,
        "median_ms": statistics.median(samples) / 1_000_000,
        "p95_ms": _percentile(samples, 0.95) / 1_000_000,
    }


def _build_cases(shape: tuple[int, ...]) -> dict[str, dict[str, Benchmark]]:
    host = np.linspace(-3.0, 3.0, num=math.prod(shape), dtype=np.float32).reshape(shape)
    x = anp.ndarray.from_numpy(host)
    dtype = np.dtype(np.float32)

    return {
        "zeros": {
            "public": lambda: anp.zeros(shape, dtype=dtype),
            "core": lambda: _core.array.zeros(shape, dtype),
        },
        "sin": {
            "public": lambda: anp.sin(x),
            "core": lambda: _core.math.sin(x),
        },
        "add": {
            "public": lambda: anp.add(x, x),
            "core": lambda: _core.math.add(x, x, None),
        },
        "sum-axis-0": {
            "public": lambda: anp.sum(x, axis=0),
            "core": lambda: _core.math.sum(x, 0, False, None),
        },
        "modf": {
            "public": lambda: anp.modf(x),
            "core": lambda: _core.math.modf(x),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", default="unlabelled", help="Build/commit label stored in the output"
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        dest="shapes",
        help="Tensor shape such as 1024 or 1024x1024; may be repeated",
    )
    parser.add_argument(
        "--mode", choices=("public", "core", "both"), default="both", help="Paths to benchmark"
    )
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        help="Operator to run; may be repeated (zeros, sin, add, sum-axis-0, modf)",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if not args.shapes:
        args.shapes = [(1024,), (1024, 1024), (4096, 4096)]
    return args


def main() -> None:
    args = _parse_args()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    modes = ("public", "core") if args.mode == "both" else (args.mode,)
    records: list[dict[str, object]] = []

    for shape in args.shapes:
        cases = _build_cases(shape)
        operators = args.operators or list(cases)
        unknown = sorted(set(operators) - set(cases))
        if unknown:
            raise SystemExit(f"unknown operator(s): {', '.join(unknown)}")

        for operator in operators:
            for mode in modes:
                metrics = _measure(cases[operator][mode], args.warmup, args.repeats)
                record: dict[str, object] = {
                    "label": args.label,
                    "operator": operator,
                    "mode": mode,
                    "shape": list(shape),
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    **metrics,
                }
                records.append(record)
                logger.info(
                    "%-10s %-6s %-18s median=%.6f ms p95=%.6f ms min=%.6f ms",
                    operator,
                    mode,
                    str(shape),
                    metrics["median_ms"],
                    metrics["p95_ms"],
                    metrics["minimum_ms"],
                )

    payload = {
        "metadata": {
            "label": args.label,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "asnumpy_version": getattr(anp, "__version__", "unknown"),
            "timer": "time.perf_counter_ns",
            "note": "Use separately built before/after commits for the final A/B conclusion.",
        },
        "results": records,
    }
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
