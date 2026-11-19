"""Measure cold event indexes with one or more index workers.

Run against a complete, prebuilt TFRecord from the project root:

    OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.benchmark_index_workers \
        /path/to/events.out.tfevents.mixed --repeat 3 --compare-workers 4

Each row is a new index, including process startup for parallel runs. The OS
page cache is not flushed. Raw timings do not instrument the indexer. The
optional comparison uses the native executor and ordered reducer for three
worker result types: stock objects, packed objects, and production packing
with source-backed tensors. Comparison phase totals can overlap.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import monotonic_ns
from typing import Literal
from unittest.mock import patch

from tboardfs import index_transport
from tboardfs import parallel_index as coordinator
from tboardfs.index_segment import Job, Segment
from tboardfs.indexer import _EventIndexer


type Payload = Literal["stock", "packed", "source"]
PAYLOADS: tuple[Payload, ...] = ("stock", "packed", "source")


class _Sample:
    """Measure one complete cold index and report JSON results."""

    @staticmethod
    def index(path: Path, workers: int, size: int) -> dict[str, object]:
        """Time the selected indexer and check that it consumed the file."""
        start = monotonic_ns()
        cache = (
            _EventIndexer.parse_event_file(path, strict_framing=True)
            if workers == 1
            else coordinator.parallel_index(path, workers)
        )
        elapsed = monotonic_ns() - start
        if cache is None:
            raise ValueError(
                f"{path}: parallel indexing declined with {workers} workers"
            )
        if cache.progress.processed_pos != size:
            raise ValueError(f"{path}: expected a complete index of {size} bytes")
        return {"elapsed_ns": elapsed, "binary_entries": len(cache.binaries)}

    @staticmethod
    def emit(row: dict[str, object]) -> None:
        """Write and flush a sample as one JSON line."""
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()

    @staticmethod
    def order(counts: list[int], round_index: int) -> list[int]:
        """Rotate and reverse worker counts between trial rounds."""
        shift = round_index // 2 * (len(counts) // 2) % len(counts)
        order = counts[shift:] + counts[:shift]
        return order if round_index % 2 == 0 else order[::-1]


def _timed[**P, R](fn: Callable[P, R], durations: list[int]) -> Callable[P, R]:
    """Wrap a phase without changing how callers pass its arguments."""

    def measured(*args: P.args, **kwargs: P.kwargs) -> R:
        """Record elapsed time even when the measured call fails."""
        start = monotonic_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            durations.append(monotonic_ns() - start)

    return measured


class _Comparison:
    """Time the native reducer with each supported worker result type."""

    @staticmethod
    def stock_segment(future: Future[Segment], job: Job) -> Segment:
        """Collect unmodified stock worker results."""
        del job
        return future.result()

    @staticmethod
    def stock_work(job: Job) -> Segment:
        """Return the stock result from the isolated worker process."""
        return Segment.work(job)

    @staticmethod
    def packed_objects(job: Job) -> index_transport._PackedResult:
        """Pack objects without enabling source-backed tensor transport."""
        return index_transport._PackedResult.pack(Segment.work(job))

    @classmethod
    def measure(
        cls, path: Path, workers: int, size: int, payload: Payload
    ) -> dict[str, object]:
        """Measure a fresh index using the selected transport."""
        phases = (
            ("header_scan", Segment, "plan"),
            ("seed", coordinator._Seed, "guess"),
            ("pool_and_reduce", coordinator._Reduction, "run"),
            ("worker_result", coordinator._Reduction, "worker_segment"),
            ("unpack", index_transport._PackedResult, "unpack"),
            ("merge", coordinator._Reduction, "merge"),
            ("replay", Segment, "work"),
            ("finish", coordinator._Tables, "finish"),
            ("restore_tensors", index_transport, "_restore_tensors"),
        )
        durations: dict[str, list[int]] = {phase: [] for phase, _, _ in phases}
        with ExitStack() as scope:
            if payload == "stock":
                scope.enter_context(
                    patch.object(coordinator, "packed_work", cls.stock_work)
                )
                scope.enter_context(
                    patch.object(
                        coordinator._Reduction, "worker_segment", cls.stock_segment
                    )
                )
            elif payload == "packed":
                scope.enter_context(
                    patch.object(coordinator, "packed_work", cls.packed_objects)
                )
            for phase, owner, symbol in phases:
                timed = _timed(getattr(owner, symbol), durations[phase])
                original = vars(owner)[symbol]
                wrapper = (
                    staticmethod(timed) if isinstance(original, staticmethod) else timed
                )
                scope.enter_context(patch.object(owner, symbol, wrapper))
            row = _Sample.index(path, workers, size)
        return {
            **row,
            "payload": payload,
            "phases_ns": {phase: sum(values) for phase, values in durations.items()},
            "calls": {phase: len(values) for phase, values in durations.items()},
        }


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="complete, prebuilt TFRecord file")
    parser.add_argument("-j", "--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument(
        "-c",
        "--compare-workers",
        type=int,
        help="also time each native payload mode at this worker count",
    )
    args = parser.parse_args()
    if min(args.workers) < 1 or args.repeat < 1:
        parser.error("workers and repeat must be positive")
    if args.compare_workers is not None and args.compare_workers < 2:
        parser.error("compare-workers must be at least 2")
    return args


def main() -> None:
    """Emit run metadata and one JSON line per completed index."""
    args = _options()
    path = args.data.resolve(strict=True)
    status = path.stat()
    size = status.st_size
    _Sample.emit(
        {
            "phase": "environment",
            "source": str(path),
            "source_bytes": size,
            "mtime_ns": status.st_mtime_ns,
            "python": sys.version,
            "available_cpus": coordinator._available_cpus(),
            "start_method": get_context("spawn").get_start_method(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        }
    )
    for trial in range(1, args.repeat + 1):
        for workers in _Sample.order(args.workers, trial - 1):
            row = _Sample.index(path, workers, size)
            _Sample.emit({"phase": "index", "trial": trial, "workers": workers, **row})
        if args.compare_workers is not None:
            workers = args.compare_workers
            payloads = PAYLOADS if trial % 2 else PAYLOADS[::-1]
            for payload in payloads:
                row = _Comparison.measure(path, workers, size, payload)
                _Sample.emit(
                    {
                        "phase": "index_breakdown",
                        "trial": trial,
                        "workers": workers,
                        **row,
                    }
                )


if __name__ == "__main__":
    main()
