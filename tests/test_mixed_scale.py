"""Exercise a dense mixed TFRecord through strict framing and FUSE callbacks.

Run the same assertions on a full 1 GiB log with:
    TBOARDFS_MIXED_SCALE_BYTES=1073741824 uv run --locked pytest -q tests/test_mixed_scale.py
"""

import json
import os
import pstats
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from test_scale_breadth import check_breadth

from scripts import benchmark_scale
from scripts.benchmark_scale import _read_file
from scripts.scale_corpus import (
    BULK_AUDIO,
    BULK_IMAGE,
    BULK_NUMERIC,
    EVENT_NAME,
    IMAGES,
    LOSS,
    _BulkSamples,
    _ScaleCorpus,
    write_mixed,
)
from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords


def _check_framing(source: Path, count: int) -> tuple[int, bytes]:
    """Stream through every CRC and verify that the final frame reaches EOF."""
    frames = 0
    record = None
    with source.open("rb") as stream:
        for record in iter_tfrecords(stream):
            if frames == 0:
                assert record.record_offset == 0
                version = compat_pb2.Event.FromString(record.payload)
                assert version.file_version == "brain.Event:2"
            frames += 1
            assert record.record_offset + record.total_size == stream.tell()
    assert record is not None
    assert frames > count
    assert record.record_offset + record.total_size == source.stat().st_size
    frame = record.header + record.payload + record.footer
    assert frame == _ScaleCorpus.event_frame(count - 1)
    return record.record_offset, frame


def _check_bulk_exports(fs: TensorBoardFS) -> None:
    image, audio = _BulkSamples.first_media()
    for path, expected in ((BULK_IMAGE, image), (BULK_AUDIO, audio)):
        assert fs.getattr(path)["st_size"] == len(expected)
        assert _read_file(fs, path, len(expected) + 1) == expected
        offset = len(expected) - 4096
        assert _read_file(fs, path, 8192, offset=offset) == expected[offset:]
    data = np.load(BytesIO(_read_file(fs, BULK_NUMERIC, 1 << 20)), allow_pickle=False)
    assert data.shape == (384, 512)
    assert data.dtype == np.dtype("float32")
    rows, columns = np.ogrid[:384, :512]
    expected_data = (rows % 127) / 8 + ((rows + columns) % 256) / 4
    np.testing.assert_array_equal(data, expected_data.astype(np.float32), strict=True)


def test_mixed_log(tmp_path: Path) -> None:
    count = 256
    target = int(os.environ.get("TBOARDFS_MIXED_SCALE_BYTES", str(24 << 20)))
    source = tmp_path / EVENT_NAME
    write_mixed(source, count, target)
    assert source.stat().st_size >= target
    raw_offset, frame = _check_framing(source, count)

    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, cache_size_mb=0)
    assert fs.readdir(IMAGES) == [
        ".",
        "..",
        *(f"{step:06d}.png" for step in range(count)),
    ]
    rows = json.loads(_read_file(fs, LOSS, count * 256))
    assert [(row["step"], row["value"]) for row in rows] == [
        (step, step / 4) for step in range(count)
    ]
    last = count - 1
    assert _read_file(fs, _ScaleCorpus.image_path(last), 128) == (
        _ScaleCorpus.small_png(last)
    )
    raw_path = f"/raw/{EVENT_NAME}/{raw_offset:020d}.tfrecord"
    assert _read_file(fs, raw_path, len(frame) + 1) == frame
    _check_bulk_exports(fs)
    check_breadth(fs)

    with source.open("ab") as stream:
        stream.write(_ScaleCorpus.event_frame(count))
    assert _read_file(fs, _ScaleCorpus.image_path(count), 128) == (
        _ScaleCorpus.small_png(count)
    )
    assert _read_file(fs, raw_path, len(frame) + 1) == frame


def test_mixed_benchmark_profile_from_another_directory(tmp_path: Path) -> None:
    script = Path(benchmark_scale.__file__).resolve()
    checkout = script.parents[1]
    env = {**os.environ, "PYTHONPATH": str(checkout)}
    source = tmp_path / "input.tfevents"
    profile = tmp_path / "benchmark.pstats"
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_scale",
        "--data",
        str(source),
        "--count",
        "16",
    ]
    subprocess.run(
        [*command, "--prepare", "--target-bytes", str(14 << 20)],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    completed = subprocess.run(
        [*command, "--profile", str(profile), "--profile-phase", "append_and_read"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert Path(result["source"]).is_relative_to(checkout)
    assert result["input_bytes"] == source.stat().st_size >= 14 << 20
    assert result["profiled_phases"] == ["append_and_read"]
    assert all(result[f"{phase}_ns"] > 0 for phase in benchmark_scale.PHASES)
    stats = pstats.Stats(str(profile))
    functions = {key[2] for key in stats.stats}
    assert "append_and_read" in functions
    assert functions.isdisjoint({"list_repeatedly", "copyfile", "write_mixed"})
