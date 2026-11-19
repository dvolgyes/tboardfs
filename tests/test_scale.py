"""Exercise large live runs through the FUSE-facing operations.

Run the full scale case with:
    TBOARDFS_SCALE_COUNT=100000 uv run --locked pytest -q tests/test_scale.py
"""

import json
import os
from errno import ENOENT
from pathlib import Path

import pytest

from scripts.benchmark_scale import _read_file
from scripts.scale_corpus import EVENT_NAME, IMAGES, LARGE, LOSS, SIBLING
from scripts.scale_corpus import _ScaleCorpus as ScaleCorpus
from tboardfs import TensorBoardFS


def _row(step: int, first: int = 0) -> dict[str, int | float | None]:
    return {
        "epoch": None,
        "step": step,
        "wall_time": 100.0 + step,
        "relative_time": float(step - first),
        "value": step / 4,
    }


def _assert_append_and_truncate(
    fs: TensorBoardFS, source: Path, count: int, names: list[str], initial_loss: bytes
) -> None:
    old_end = source.stat().st_size
    append = ScaleCorpus.event_frame(count)
    with source.open("ab") as stream:
        stream.write(append[:-2])
    assert fs.readdir(IMAGES) == names
    with source.open("ab") as stream:
        stream.write(append[-2:])
    assert fs.readdir(IMAGES) == [*names, f"{count:06d}.png"]
    assert _read_file(fs, ScaleCorpus.image_path(count), 128) == ScaleCorpus.small_png(
        count
    )
    appended_loss = _read_file(fs, LOSS, (count + 1) * 256)
    assert json.loads(appended_loss) == [_row(step) for step in range(count + 1)]
    cold = TensorBoardFS(source.parent, refresh_age_seconds=0)
    assert cold.readdir(IMAGES) == fs.readdir(IMAGES)
    assert _read_file(cold, LOSS, (count + 1) * 256) == appended_loss
    del cold

    with source.open("r+b") as stream:
        stream.truncate(old_end)
    assert fs.readdir(IMAGES) == names
    assert _read_file(fs, LOSS, count * 256) == initial_loss
    with pytest.raises(OSError) as missing_append:
        fs.open(ScaleCorpus.image_path(count), os.O_RDONLY)
    assert missing_append.value.errno == ENOENT


def _assert_replace_and_delete(
    fs: TensorBoardFS, source: Path, last: int, raw_path: str
) -> None:
    replacement = source.with_name("replacement")
    frame = ScaleCorpus.event_frame(last, pixel=0xABCDEF)
    replacement.write_bytes(frame)
    replacement.replace(source)
    assert fs.readdir(IMAGES) == [".", "..", f"{last:06d}.png"]
    late_path = ScaleCorpus.image_path(last)
    assert _read_file(fs, late_path, 128) == ScaleCorpus.small_png(0xABCDEF)
    assert json.loads(_read_file(fs, LOSS, 256)) == [_row(last, first=last)]
    assert _read_file(fs, f"/raw/{EVENT_NAME}/{0:020d}.tfrecord", 512) == frame

    source.unlink()
    for path in (late_path, raw_path, LOSS, LARGE, SIBLING):
        with pytest.raises(OSError) as missing:
            fs.open(path, os.O_RDONLY)
        assert missing.value.errno == ENOENT


def test_large_live_run_generations(tmp_path: Path) -> None:
    count = int(os.environ.get("TBOARDFS_SCALE_COUNT", "20000"))
    assert 2 <= count < 1_000_000
    source = tmp_path / EVENT_NAME
    ScaleCorpus.write(source, count)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    initial_names = [".", "..", *(f"{step:06d}.png" for step in range(count))]
    assert fs.readdir(IMAGES) == initial_names
    assert fs.readdir(IMAGES) == initial_names
    for step in (count - 2, count - 1, 0):
        assert _read_file(
            fs, ScaleCorpus.image_path(step), 128
        ) == ScaleCorpus.small_png(step)

    initial_loss = _read_file(fs, LOSS, count * 256)
    assert json.loads(initial_loss) == [_row(step) for step in range(count)]
    assert _read_file(fs, SIBLING, 128) == ScaleCorpus.small_png(0xABCDEF)
    png = ScaleCorpus.large_png()
    offset = 8 << 20
    assert fs.getattr(LARGE)["st_size"] == len(png) > offset + 4096
    assert _read_file(fs, LARGE, 4096, offset=offset) == png[offset : offset + 4096]

    old_end = source.stat().st_size
    last = count - 1
    late_path = ScaleCorpus.image_path(last)
    frame = ScaleCorpus.event_frame(last)
    raw_path = f"/raw/{EVENT_NAME}/{old_end - len(frame):020d}.tfrecord"
    assert _read_file(fs, raw_path, 512) == frame

    late = fs.open(late_path, os.O_RDONLY)
    raw = fs.open(raw_path, os.O_RDONLY)
    loss = fs.open(LOSS, os.O_RDONLY)
    try:
        _assert_append_and_truncate(fs, source, count, initial_names, initial_loss)
        _assert_replace_and_delete(fs, source, last, raw_path)
        assert fs.read(late_path, 128, 0, late) == ScaleCorpus.small_png(last)
        assert fs.read(raw_path, 512, 12, raw) == frame[12:]
        assert fs.read(LOSS, 256, len(initial_loss) - 128, loss) == initial_loss[-128:]
    finally:
        fs.release(LOSS, loss)
        fs.release(raw_path, raw)
        fs.release(late_path, late)
