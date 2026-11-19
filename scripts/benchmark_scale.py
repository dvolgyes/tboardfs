r"""Measure FUSE callbacks against a deterministic, prebuilt event log.

From the project root:
    uv run --locked python -m scripts.benchmark_scale --prepare --data /tmp/scale.tfevents
    uv run --locked python -m scripts.benchmark_scale --data /tmp/scale.tfevents

To compare a baseline source tree with this checkout, run outside both trees:
    repo=$PWD
    baseline=/path/to/baseline
    uv run --project "$repo" --locked --directory /tmp env \
        PYTHONPATH="$baseline:$repo" python -m scripts.benchmark_scale \
        --data /tmp/scale.tfevents

Repeat with PYTHONPATH="$repo"; alternate versions, one process per trial.
Each trial copies the input before timing: app caches are cold, OS caches aren't.
Timings exclude corpus copying, expected-value construction and checks; peak
RSS is the high-water mark of the whole measurement process. Use --profile PATH
for an instrumented trial, optionally restricting it with --profile-phase PHASE
(repeatable). Compare ordinary timing trials without --profile. Inspect a
profile with `python -m pstats PATH`.
"""

import argparse
import cProfile
import json
import os
import resource
import shutil
import sys
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

import tboardfs
from scripts import scale_corpus
from scripts.scale_corpus import EVENT_NAME, IMAGES, LARGE, LOSS, SIBLING, _ScaleCorpus
from tboardfs import TensorBoardFS
from tboardfs.tfrecord import iter_tfrecords

MIXED_TARGET_BYTES = 1 << 30
MIXED_PHASES = (
    "bulk_image",
    "bulk_audio",
    "bulk_numeric",
    "raw_record",
    "real_text",
    "real_numeric",
)
PHASES = (
    "cold_scalar",
    "cold_listing",
    "warm_listing_batch",
    "warm_late_read_batch",
    "large_uncached_offset_read",
    "small_sibling_read",
    *MIXED_PHASES,
    "append_and_read",
)


def _read_file(fs: TensorBoardFS, path: str, size: int, *, offset: int = 0) -> bytes:
    """Read through one mounted-file handle and release it even if a read fails."""
    handle = fs.open(path, os.O_RDONLY)
    try:
        return fs.read(path, size, offset, handle)
    finally:
        fs.release(path, handle)


class _Batch:
    """Group callbacks that form one measured operation."""

    @staticmethod
    def list_repeatedly(fs: TensorBoardFS, repeats: int) -> None:
        """List the same directory without retaining its entries."""
        for _ in range(repeats):
            fs.readdir(IMAGES)

    @staticmethod
    def read_late_images(fs: TensorBoardFS, paths: list[str]) -> list[bytes]:
        """Read the terminal images through separate handles."""
        return [_read_file(fs, path, 128) for path in paths]

    @staticmethod
    def append_and_read(
        fs: TensorBoardFS, source: Path, frame: bytes, path: str
    ) -> bytes:
        """Write the prepared frame and read its newly exposed image."""
        with source.open("ab") as stream:
            stream.write(frame)
        return _read_file(fs, path, 128)


class _Trial:
    """Collect timings and optionally profile only selected callbacks."""

    durations: dict[str, int]
    phases: set[str]
    profiler: cProfile.Profile | None

    def __init__(self, profile: Path | None, phases: list[str]) -> None:
        self.durations = {}
        self.phases = set(phases or PHASES)
        self.profiler = cProfile.Profile() if profile is not None else None

    def measure[T](self, phase: str, operation: Callable[[], T]) -> T:
        """Time an operation without including checks or fixture construction."""
        instrument = self.profiler if phase in self.phases else None
        start = perf_counter_ns()
        if instrument is not None:
            result = instrument.runcall(operation)
        else:
            result = operation()
        self.durations[f"{phase}_ns"] = perf_counter_ns() - start
        return result

    def cold_scalar(self, root: Path, count: int) -> None:
        """Measure a scalar aggregate with an independent cold application index."""
        fs = TensorBoardFS(root, refresh_age_seconds=0)
        data = self.measure("cold_scalar", partial(_read_file, fs, LOSS, count * 256))
        rows = json.loads(data)
        if len(rows) != count or any(
            row["step"] != step or row["value"] != step / 4
            for step, row in enumerate(rows)
        ):
            raise ValueError("cold scalar aggregate returned different steps or values")

    def images(self, fs: TensorBoardFS, count: int, large_slice: bytes) -> None:
        """Measure the small-entry workload and a large image's stored payload."""
        names = self.measure("cold_listing", partial(fs.readdir, IMAGES))
        if len(names) != count + 2:
            raise ValueError(f"expected {count} images but got {len(names) - 2}")
        del names

        repeats = 32
        self.measure("warm_listing_batch", partial(_Batch.list_repeatedly, fs, repeats))
        self.durations["warm_listing_count"] = repeats
        steps = range(max(0, count - repeats), count)
        paths = [_ScaleCorpus.image_path(step) for step in steps]
        blocks = self.measure(
            "warm_late_read_batch", partial(_Batch.read_late_images, fs, paths)
        )
        self.durations["warm_late_read_count"] = len(steps)
        if blocks != [_ScaleCorpus.small_png(step) for step in steps]:
            raise ValueError("warm late reads returned different bytes")

        block = self.measure(
            "large_uncached_offset_read",
            partial(_read_file, fs, LARGE, 4096, offset=8 << 20),
        )
        if block != large_slice:
            raise ValueError("the large offset read returned different bytes")
        sibling = self.measure(
            "small_sibling_read", partial(_read_file, fs, SIBLING, 128)
        )
        if sibling != _ScaleCorpus.small_png(0xABCDEF):
            raise ValueError("the small sibling returned different bytes")

    def mixed(self, fs: TensorBoardFS, source: Path, count: int) -> None:
        """Measure independent stored payload reads and a derived numeric export."""
        image, audio = scale_corpus._BulkSamples.first_media()
        for phase, path, expected in (
            ("bulk_image", scale_corpus.BULK_IMAGE, image),
            ("bulk_audio", scale_corpus.BULK_AUDIO, audio),
        ):
            block = self.measure(
                phase, partial(_read_file, fs, path, len(expected) + 1)
            )
            if block != expected:
                raise ValueError(f"{path} returned different bytes")

        path = scale_corpus.BULK_NUMERIC
        block = self.measure("bulk_numeric", partial(_read_file, fs, path, 1 << 20))
        data = np.load(BytesIO(block), allow_pickle=False)
        if (
            data.dtype != np.dtype("float32")
            or data.shape != (384, 512)
            or not np.array_equal(data[0, :4], [0.0, 0.25, 0.5, 0.75])
        ):
            raise ValueError(f"{path} returned a different array")

        if count == 1:
            with source.open("rb") as stream:
                record = next(iter_tfrecords(stream))
            offset = record.record_offset
            frame = record.header + record.payload + record.footer
        else:
            frame = _ScaleCorpus.event_frame(count - 1)
            offset = source.stat().st_size - len(frame)
        path = f"/raw/{EVENT_NAME}/{offset:020d}.tfrecord"
        block = self.measure("raw_record", partial(_read_file, fs, path, len(frame)))
        if block != frame:
            raise ValueError(f"{path} returned different framing")

    def real_exports(self, fs: TensorBoardFS) -> None:
        """Measure typed reads for non-bulk strings and complex tensors."""
        path = "/text/binary-strings/000004.json"
        text = self.measure("real_text", partial(_read_file, fs, path, 4096))
        payload = json.loads(text)
        if payload["shape"] != [3] or payload["values"] != ["", "A\0B", "\ufffd\ufffd"]:
            raise ValueError(f"{path} returned different string values")

        path = "/tensors/complex64/000004.npy"
        block = self.measure("real_numeric", partial(_read_file, fs, path, 4096))
        data = np.load(BytesIO(block), allow_pickle=False)
        if data.dtype != np.dtype("complex64") or not np.array_equal(
            data, [1 + 2j, -3.5 - 0.5j]
        ):
            raise ValueError(f"{path} returned a different array")

    def append(self, fs: TensorBoardFS, source: Path, count: int) -> None:
        """Measure completing an append and reading the new image."""
        frame = _ScaleCorpus.event_frame(count)
        path = _ScaleCorpus.image_path(count)
        block = self.measure(
            "append_and_read", partial(_Batch.append_and_read, fs, source, frame, path)
        )
        if block != _ScaleCorpus.small_png(count):
            raise ValueError("the appended image returned different bytes")


def main() -> None:
    """Prepare a corpus or emit one trial as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--target-bytes", type=int, default=MIXED_TARGET_BYTES)
    parser.add_argument(
        "--narrow", action="store_true", help="use the historical corpus"
    )
    parser.add_argument(
        "--profile", type=Path, help="write cProfile stats to this path"
    )
    parser.add_argument("--profile-phase", action="append", choices=PHASES, default=[])
    args = parser.parse_args()
    if not 0 < args.count < 1_000_000:
        parser.error("--count must be between 1 and 999999")
    if args.target_bytes <= 0:
        parser.error("--target-bytes must be positive")
    if args.profile_phase and args.profile is None:
        parser.error("--profile-phase requires --profile")
    if args.prepare and args.profile is not None:
        parser.error("--profile cannot be used with --prepare")
    if args.profile is not None and args.profile.resolve() == args.data.resolve():
        parser.error("--profile must differ from --data")
    if args.narrow and set(args.profile_phase).intersection(MIXED_PHASES):
        parser.error("--narrow does not support mixed profile phases")

    if args.prepare:
        if args.narrow:
            _ScaleCorpus.write(args.data, args.count)
        else:
            scale_corpus.write_mixed(args.data, args.count, args.target_bytes)
        return

    trial = _Trial(args.profile, args.profile_phase)
    large_slice = _ScaleCorpus.large_png()[8 << 20 : (8 << 20) + 4096]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / EVENT_NAME
        shutil.copyfile(args.data, source)
        trial.cold_scalar(root, args.count)
        fs = TensorBoardFS(root, refresh_age_seconds=0, cache_size_mb=0)
        trial.images(fs, args.count, large_slice)
        if not args.narrow:
            trial.mixed(fs, source, args.count)
            trial.real_exports(fs)
        trial.append(fs, source, args.count)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    profiled_phases = []
    if trial.profiler is not None:
        trial.profiler.dump_stats(str(args.profile))
        profiled_phases = sorted(
            phase for phase in trial.phases if f"{phase}_ns" in trial.durations
        )
    sys.stdout.write(
        json.dumps(
            {
                **trial.durations,
                "count": args.count,
                "input_bytes": args.data.stat().st_size,
                "peak_rss_bytes": peak_rss
                if sys.platform == "darwin"
                else peak_rss * 1024,
                "profiled_phases": profiled_phases,
                "source": tboardfs.__file__,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
