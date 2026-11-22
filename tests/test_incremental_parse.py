"""A run read while it grows shows the same files as a fresh parse."""

import os
import stat
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2

DT_DOUBLE = 2
DT_INT64 = 9
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


def _frame(event: compat_pb2.Event) -> bytes:
    payload = event.SerializeToString()
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def _random_record(rng: np.random.Generator, step: int, wall_time: float) -> bytes:
    """Return one event with a random mix of scalar and non-scalar values."""
    event = compat_pb2.Event(wall_time=wall_time, step=step)
    kinds = rng.choice(
        ["loss", "lr", "count", "epoch", "weights", "img"],
        size=int(rng.integers(1, 4)),
        replace=False,
    )
    for kind in kinds:
        value = event.summary.value.add(tag=str(kind))
        if kind in {"loss", "epoch"}:
            value.simple_value = float(rng.normal())
        elif kind == "lr":
            value.tensor.dtype = DT_DOUBLE
            value.tensor.double_val.append(float(rng.random()))
        elif kind == "count":
            value.tensor.dtype = DT_INT64
            value.tensor.int64_val.append(int(rng.integers(-5, 1000)))
        elif kind == "weights":
            limits = np.sort(rng.normal(size=3))
            value.histo.min = float(limits[0] - 1)
            value.histo.bucket_limit.extend(limits.tolist())
            value.histo.bucket.extend(rng.integers(0, 9, size=3).tolist())
        else:
            value.image.height = 1
            value.image.width = 1
            value.image.encoded_image_string = PNG
    return _frame(event)


def _files(fs: TensorBoardFS, path: str = "/") -> Iterator[str]:
    for name in fs.readdir(path):
        if name in {".", ".."}:
            continue
        child = f"{path.rstrip('/')}/{name}"
        if stat.S_ISDIR(fs.getattr(child)["st_mode"]):
            yield from _files(fs, child)
        else:
            yield child


def _contents(fs: TensorBoardFS) -> dict[str, bytes]:
    return {path: fs.read(path, 1 << 30, 0) for path in _files(fs)}


@pytest.mark.parametrize("seed", range(40))
def test_incremental_refresh_equals_a_fresh_parse(tmp_path: Path, seed: int) -> None:
    """Appends split at any byte give the files of one full parse."""
    rng = np.random.default_rng(seed)
    data = b"".join(
        _random_record(rng, step, 100.0 + step + float(rng.random()))
        for step in range(int(rng.integers(5, 40)))
    )
    cuts = np.sort(rng.integers(1, len(data), size=int(rng.integers(1, 5))))
    event = tmp_path / "events.out.tfevents.host"
    growing = TensorBoardFS(tmp_path, refresh_age_seconds=0, source_check_seconds=0)
    start = 0
    for cut in [*cuts.tolist(), len(data)]:
        with event.open("ab") as handle:
            handle.write(data[start:cut])
        start = cut
        _contents(growing)

    assert _contents(growing) == _contents(TensorBoardFS(tmp_path))


@pytest.mark.parametrize("new_steps", [3, 6])
def test_replaced_event_file_is_parsed_again(tmp_path: Path, new_steps: int) -> None:
    """A new file at the same path never continues from the old offset."""
    event = tmp_path / "events.out.tfevents.host"
    event.write_bytes(b"".join(_loss(step, 1.0) for step in range(3)))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, source_check_seconds=0)
    fs.read("/scalars/loss.json", 1 << 20, 0)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"".join(_loss(step, 2.0) for step in range(new_steps)))

    os.replace(replacement, event)

    assert _contents(fs) == _contents(TensorBoardFS(tmp_path))


def _loss(step: int, value: float) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag="loss", simple_value=value)
    return _frame(event)
