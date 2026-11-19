r"""Measure FUSE callbacks against a deterministic, prebuilt event log.

From the project root:
    uv run --locked python scripts/benchmark_scale.py --prepare --data /tmp/scale.tfevents
    uv run --locked python scripts/benchmark_scale.py --data /tmp/scale.tfevents

To compare a baseline source tree with this checkout, run outside both trees:
    repo=$PWD
    uv run --project "$repo" --locked --directory /tmp env PYTHONPATH=/path/to/baseline \
        python "$repo/scripts/benchmark_scale.py" --data /tmp/scale.tfevents

Repeat with PYTHONPATH="$repo"; alternate versions, one process per trial.
Each trial copies the input before timing: app caches are cold, OS caches aren't.
Each trial opens one app for cold scalar aggregation, drops it, then opens an
independent app for the remaining timings with the rendered-file cache disabled.
"""

import argparse
import json
import os
import resource
import shutil
import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zlib import compress, crc32

import tboardfs
from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2

EVENT_NAME = "events.out.tfevents.scale"
IMAGES = "/images/series"
LOSS = "/scalars/loss.json"
LARGE = "/images/large/000000.png"
SIBLING = "/images/sibling/000000.png"
LARGE_WIDTH, LARGE_HEIGHT = 2048, 1536


class _ScaleCorpus:
    """Generate deterministic checksummed scalar and image records."""

    @staticmethod
    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", crc32(kind + data))
        )

    @classmethod
    def _png(cls, width: int, height: int, rows: bytes) -> bytes:
        return (
            b"\x89PNG\r\n\x1a\n"
            + cls._chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + cls._chunk(b"IDAT", compress(rows, level=0))
            + cls._chunk(b"IEND", b"")
        )

    @classmethod
    def small_png(cls, pixel: int) -> bytes:
        """Encode a one-pixel RGB image."""
        return cls._png(1, 1, b"\0" + pixel.to_bytes(3, "big"))

    @classmethod
    def large_png(cls) -> bytes:
        """Encode an RGB gradient with uncompressed scanlines."""
        row = b"\0" + bytes(range(256)) * (LARGE_WIDTH * 3 // 256)
        return cls._png(LARGE_WIDTH, LARGE_HEIGHT, row * LARGE_HEIGHT)

    @classmethod
    def event_frame(
        cls, step: int, *, pixel: int | None = None, large: bytes | None = None
    ) -> bytes:
        """Write a scalar and image event as one checksummed frame."""
        event = getattr(compat_pb2, "Event")(step=step, wall_time=100.0 + step)
        event.summary.value.add(tag="loss", simple_value=step / 4)
        image = event.summary.value.add(tag="series").image
        image.width = image.height = 1
        image.colorspace = 3
        image.encoded_image_string = cls.small_png(step if pixel is None else pixel)
        if large is not None:
            image = event.summary.value.add(tag="large").image
            image.width, image.height, image.colorspace = LARGE_WIDTH, LARGE_HEIGHT, 3
            image.encoded_image_string = large
            image = event.summary.value.add(tag="sibling").image
            image.width = image.height = 1
            image.colorspace = 3
            image.encoded_image_string = cls.small_png(0xABCDEF)
        payload: bytes = event.SerializeToString()
        length = struct.pack("<Q", len(payload))
        return (
            length
            + struct.pack("<I", masked_crc32c(length))
            + payload
            + struct.pack("<I", masked_crc32c(payload))
        )

    @classmethod
    def write(cls, path: Path, count: int) -> None:
        """Write one image and scalar per step, plus a large image and sibling."""
        with path.open("xb") as stream:
            stream.write(cls.event_frame(0, large=cls.large_png()))
            for step in range(1, count):
                stream.write(cls.event_frame(step))

    @staticmethod
    def image_path(step: int) -> str:
        """Return the virtual image path for a series step."""
        return f"{IMAGES}/{step:06d}.png"


def _read_file(fs: TensorBoardFS, path: str, size: int, *, offset: int = 0) -> bytes:
    """Read through one mounted-file handle and release it even if a read fails."""
    handle = fs.open(path, os.O_RDONLY)
    try:
        return fs.read(path, size, offset, handle)
    finally:
        fs.release(path, handle)


def _run_trial(root: Path, count: int, large_slice: bytes) -> dict[str, int]:
    """Time listing, image reads and append in a fresh filesystem."""
    fs = TensorBoardFS(root, refresh_age_seconds=0, cache_size_mb=0)
    durations = {}
    start = perf_counter_ns()
    names = fs.readdir(IMAGES)
    durations["cold_listing_ns"] = perf_counter_ns() - start
    if len(names) != count + 2:
        raise ValueError(f"expected {count} images but got {len(names) - 2}")
    del names

    repeats = 32
    start = perf_counter_ns()
    for _ in range(repeats):
        fs.readdir(IMAGES)
    durations["warm_listing_batch_ns"] = perf_counter_ns() - start
    durations["warm_listing_count"] = repeats

    steps = range(max(0, count - repeats), count)
    start = perf_counter_ns()
    blocks = [_read_file(fs, _ScaleCorpus.image_path(step), 128) for step in steps]
    durations["warm_late_read_batch_ns"] = perf_counter_ns() - start
    durations["warm_late_read_count"] = len(steps)
    if blocks != [_ScaleCorpus.small_png(step) for step in steps]:
        raise ValueError("warm late reads returned different bytes")

    start = perf_counter_ns()
    block = _read_file(fs, LARGE, 4096, offset=8 << 20)
    durations["large_uncached_offset_read_ns"] = perf_counter_ns() - start
    if block != large_slice:
        raise ValueError("the large offset read returned different bytes")

    start = perf_counter_ns()
    sibling = _read_file(fs, SIBLING, 128)
    durations["small_sibling_read_ns"] = perf_counter_ns() - start
    if sibling != _ScaleCorpus.small_png(0xABCDEF):
        raise ValueError("the small sibling returned different bytes")

    append = _ScaleCorpus.event_frame(count)
    start = perf_counter_ns()
    with (root / EVENT_NAME).open("ab") as stream:
        stream.write(append)
    block = _read_file(fs, _ScaleCorpus.image_path(count), 128)
    durations["append_and_read_ns"] = perf_counter_ns() - start
    if block != _ScaleCorpus.small_png(count):
        raise ValueError("the appended image returned different bytes")
    return durations


def main() -> None:
    """Prepare a corpus or emit one trial as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not 0 < args.count < 1_000_000:
        parser.error("--count must be between 1 and 999999")
    if args.prepare:
        _ScaleCorpus.write(args.data, args.count)
        return
    large_slice = _ScaleCorpus.large_png()[8 << 20 : (8 << 20) + 4096]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        shutil.copyfile(args.data, root / EVENT_NAME)
        fs = TensorBoardFS(root, refresh_age_seconds=0)
        start = perf_counter_ns()
        data = _read_file(fs, LOSS, args.count * 256)
        cold_scalar_ns = perf_counter_ns() - start
        rows = json.loads(data)
        if len(rows) != args.count or rows[-1]["value"] != (args.count - 1) / 4:
            raise ValueError("cold scalar aggregate returned different values")
        del fs, data, rows
        result = _run_trial(root, args.count, large_slice)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(
        cold_scalar_ns=cold_scalar_ns,
        peak_rss_bytes=peak_rss if sys.platform == "darwin" else peak_rss * 1024,
        count=args.count,
        input_bytes=args.data.stat().st_size,
    )
    sys.stdout.write(
        json.dumps({**result, "source": tboardfs.__file__}, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
