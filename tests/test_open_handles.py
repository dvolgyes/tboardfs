"""An open file handle reads one fixed version of a growing virtual file."""

import json
import struct
import threading
import time
from errno import EISDIR, ENOENT
from pathlib import Path

import numpy as np
import pytest
from test_concurrency import _run_threads

from tboardfs import TensorBoardFS
from tboardfs.constants import DEFAULT_SOURCE_CHECK_SECONDS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2

LOSS = "/scalars/loss.json"


def _loss_record(step: int) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag="loss", simple_value=1.0 / (step + 1))
    payload = event.SerializeToString()
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def _append(path: Path, *steps: int) -> None:
    with path.open("ab") as handle:
        for step in steps:
            handle.write(_loss_record(step))


def _read_in_chunks(fs: TensorBoardFS, path: str, handle: int, chunk: int) -> bytes:
    parts = []
    offset = 0
    while part := fs.read(path, chunk, offset, handle):
        parts.append(part)
        offset += len(part)
    return b"".join(parts)


def _growing_fs(
    tmp_path: Path,
    *,
    source_check_seconds: float = 0,
    scalar_formats: str = "json,tsv",
) -> tuple[TensorBoardFS, Path]:
    event = tmp_path / "events.out.tfevents"
    _append(event, *range(3))
    fs = TensorBoardFS(
        tmp_path,
        refresh_age_seconds=0,
        source_check_seconds=source_check_seconds,
        scalar_formats=scalar_formats,
    )
    return fs, event


def test_handle_keeps_the_version_from_open(tmp_path: Path) -> None:
    """Chunks read after the run grows still belong to the opened version."""
    fs, event = _growing_fs(tmp_path)
    handle = fs.open(LOSS, 0)
    first = fs.read(LOSS, 50, 0, handle)
    _append(event, 3, 4)

    data = first + _read_in_chunks(fs, LOSS, handle, 50)[50:]

    assert [row["step"] for row in json.loads(data)] == [0, 1, 2]
    assert [row["step"] for row in json.loads(fs.read(LOSS, 1 << 20, 0))] == list(
        range(5)
    )


def test_handle_size_matches_its_version(tmp_path: Path) -> None:
    """Stat through a handle reports the size of the bytes that handle reads."""
    fs, event = _growing_fs(tmp_path)
    handle = fs.open(LOSS, 0)
    _append(event, 3)

    size = fs.getattr(LOSS, handle)["st_size"]

    assert size == len(_read_in_chunks(fs, LOSS, handle, 64))
    assert fs.getattr(LOSS)["st_size"] > size


def test_handles_opened_at_different_times_stay_apart(tmp_path: Path) -> None:
    """Each handle reads its own version while both are open."""
    fs, event = _growing_fs(tmp_path)
    old = fs.open(LOSS, 0)
    _append(event, 3)
    new = fs.open(LOSS, 0)

    old_rows = json.loads(_read_in_chunks(fs, LOSS, old, 40))
    new_rows = json.loads(_read_in_chunks(fs, LOSS, new, 40))

    assert len(old_rows) == 3
    assert len(new_rows) == 4


def test_binary_export_stays_loadable_while_the_run_grows(tmp_path: Path) -> None:
    """An npz read in chunks across a change is one complete archive."""
    fs, event = _growing_fs(tmp_path, scalar_formats="json,tsv,npz")
    path = "/scalars/loss.npz"
    handle = fs.open(path, 0)
    first = fs.read(path, 100, 0, handle)
    _append(event, *range(3, 50))
    rest = _read_in_chunks(fs, path, handle, 100)[100:]
    out = tmp_path / "loss.npz"
    out.write_bytes(first + rest)

    with np.load(out) as archive:
        assert archive["step"].tolist() == [0, 1, 2]


def test_released_handle_no_longer_pins_its_version(tmp_path: Path) -> None:
    """After release the old handle number reads the current version."""
    fs, event = _growing_fs(tmp_path)
    handle = fs.open(LOSS, 0)
    _append(event, 3)

    fs.release(LOSS, handle)

    rows = json.loads(fs.read(LOSS, 1 << 20, 0, handle))
    assert len(rows) == 4


def test_handle_survives_removal_of_its_run(tmp_path: Path) -> None:
    """A handle keeps reading after the run behind it disappears."""
    fs, event = _growing_fs(tmp_path)
    handle = fs.open(LOSS, 0)
    expected = fs.getattr(LOSS, handle)["st_size"]
    event.unlink()
    fs.readdir("/")

    assert len(_read_in_chunks(fs, LOSS, handle, 64)) == expected
    assert fs.getattr(LOSS, handle)["st_size"] == expected


@pytest.mark.parametrize(
    ("path", "error"), [("/scalars", EISDIR), ("/scalars/missing.json", ENOENT)]
)
def test_open_rejects_directories_and_missing_files(
    tmp_path: Path, path: str, error: int
) -> None:
    """Only existing files can be opened."""
    fs, _ = _growing_fs(tmp_path, source_check_seconds=DEFAULT_SOURCE_CHECK_SECONDS)

    with pytest.raises(OSError) as raised:
        fs.open(path, 0)

    assert raised.value.errno == error


def test_threads_with_handles_read_whole_versions_while_runs_grow(
    tmp_path: Path,
) -> None:
    """Concurrent open/read/release cycles always see one complete version."""
    fs, event = _growing_fs(tmp_path)
    stop = threading.Event()
    reads = [0]

    def writer() -> None:
        for step in range(3, 300):
            _append(event, step)
            time.sleep(0.001)
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            handle = fs.open(LOSS, 0)
            try:
                size = fs.getattr(LOSS, handle)["st_size"]
                data = _read_in_chunks(fs, LOSS, handle, 97)
            finally:
                fs.release(LOSS, handle)
            assert len(data) == size
            steps = [row["step"] for row in json.loads(data)]
            assert steps == list(range(len(steps)))
            reads[0] += 1

    errors = _run_threads([writer] + [reader] * 8)

    assert errors == []
    assert reads[0] > 0
    handle = fs.open(LOSS, 0)
    assert len(json.loads(_read_in_chunks(fs, LOSS, handle, 4096))) == 300
    fs.release(LOSS, handle)
