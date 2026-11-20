"""Cold mounted tensors defer verified source reads without changing typed files."""

import json
import os
from base64 import b64decode
from errno import EIO, ENOENT
from io import BytesIO
from pathlib import Path
from typing import Never

import numpy as np
import pytest
from test_incremental_parse import _frame, _loss

from tboardfs import TensorBoardFS, filesystem, parallel_index
from tboardfs.index_segment import Segment
from tboardfs.index_transport import _PackedResult
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache
from tboardfs.model.json_entry import JsonEntry
from tboardfs.proto import compat_pb2
from tboardfs.tensor_source import TensorSource

_SOURCE = "events.out.tfevents.host"
_SHAPE = (128, 256)
_READ_ALL = 1 << 20


def _values(step: int) -> np.ndarray:
    if step == 2:
        return np.resize(np.array([1.5, -2.25, 0.125], dtype="<f4"), _SHAPE)
    return np.arange(32768, dtype="<u2").reshape(_SHAPE) + step


def _tensor_frame(step: int, *, tag: str | None = None, shift: int = 0) -> bytes:
    values = _values(step) + shift
    event = compat_pb2.Event(step=step, wall_time=100.0 + step + shift)
    value = event.summary.value.add(
        tag=tag or {2: "bfloat", 3: "malformed"}.get(step, "main")
    )
    tensor = value.tensor
    tensor.dtype = 14 if step == 2 else 17
    tensor.tensor_content = (
        (values.view("<u4") >> 16).astype("<u2").tobytes()
        if step == 2
        else values.tobytes()
    )
    tensor.tensor_shape.dim.add(size=_SHAPE[0] + (step == 3))
    tensor.tensor_shape.dim.add(size=_SHAPE[1])
    if step != 2:
        tensor.int_val.append(7)
    return _frame(event)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    (tmp_path / "mnt").mkdir()
    event = root / _SOURCE
    event.write_bytes(b"".join(_tensor_frame(step) for step in range(1, 9)))
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(parallel_index, "_available_cpus", lambda: 4)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return event


def _mount(source: Path, *, disk_index: bool = False) -> TensorBoardFS:
    return TensorBoardFS(
        source.parent,
        mountpoint=source.parent.parent / "mnt",
        disk_index=disk_index,
        index_workers=4,
        refresh_age_seconds=0,
    )


def _cache(fs: TensorBoardFS, source: Path) -> EventFileCache:
    return fs._state.runs[()].files[source]


def _track_reads(monkeypatch: pytest.MonkeyPatch) -> list[TensorSource]:
    reads: list[TensorSource] = []
    read = TensorSource.read

    def track(extent: TensorSource) -> bytes:
        reads.append(extent)
        return read(extent)

    monkeypatch.setattr(TensorSource, "read", track)
    return reads


def _track_ranges(source: Path, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    unpack = _PackedResult.unpack
    starts: list[int] = []

    def track_range(
        result: _PackedResult, *, mountpoint: Path | None = None
    ) -> Segment:
        assert mountpoint == source.parent.parent / "mnt"
        segment = unpack(result, mountpoint=mountpoint)
        starts.append(segment.start)
        return segment

    monkeypatch.setattr(_PackedResult, "unpack", track_range)
    return starts


def _assert_cold_listing(
    fs: TensorBoardFS, serial: TensorBoardFS, starts: list[int]
) -> None:
    assert fs.virtual_tree()["type"] == "dir"
    assert len(starts) == len(set(starts)) == 4
    assert fs.readdir("/tensors") == serial.readdir("/tensors")
    for tag in ("main", "bfloat", "malformed"):
        directory = f"/tensors/{tag}"
        assert fs.readdir(directory) == serial.readdir(directory)


def _assert_source_backed(entry: JsonEntry, source: Path) -> TensorSource:
    extent = entry.tensor_source
    assert isinstance(extent, TensorSource)
    assert extent.source_path == source
    assert entry.payload["tensor"]["tensor_content"] == b""
    return extent


def _assert_first_reads(
    fs: TensorBoardFS,
    serial: TensorBoardFS,
    reads: list[TensorSource],
    first: TensorSource,
    bfloat: TensorSource,
) -> None:
    first_path = "/tensors/main/000001.npy"
    assert fs.getattr(first_path) == serial.getattr(first_path)
    assert len(reads) == 1 and reads[0] is first
    bfloat_path = "/tensors/bfloat/000002.npy"
    handle = fs.open(bfloat_path, os.O_RDONLY)
    try:
        assert len(reads) == 2 and reads[1] is bfloat
        assert fs.read(bfloat_path, 19, 9, handle) == serial.read(bfloat_path, 19, 9)
        assert fs.getattr(bfloat_path, handle) == serial.getattr(bfloat_path)
    finally:
        fs.release(bfloat_path, handle)


def _read_serial_parity(
    fs: TensorBoardFS, serial: TensorBoardFS, path: str, step: int
) -> bytes:
    data = fs.read(path, _READ_ALL, 0)
    assert data == serial.read(path, _READ_ALL, 0)
    info = fs.getattr(path)
    assert info == serial.getattr(path)
    assert info["st_size"] == len(data)
    assert info["st_mtime"] == 100.0 + step
    return data


def _assert_typed_parity(
    fs: TensorBoardFS, serial: TensorBoardFS, step: int, tag: str
) -> None:
    expected = _values(step)
    stem = f"/tensors/{tag}/{step:06}"
    npy = _read_serial_parity(fs, serial, f"{stem}.npy", step)
    np.testing.assert_array_equal(
        np.load(BytesIO(npy), allow_pickle=False), expected, strict=True
    )
    document = _read_serial_parity(fs, serial, f"{stem}.json", step)
    assert json.loads(document) == {
        "shape": list(_SHAPE),
        "dtype": str(expected.dtype),
        "values": expected.tolist(),
    }


def _assert_malformed_fallback(
    fs: TensorBoardFS, serial: TensorBoardFS, bad: JsonEntry
) -> None:
    content = _values(3).tobytes()
    assert bad.tensor_source is None
    assert bad.payload["tensor"]["tensor_content"] == content
    assert fs.readdir("/tensors/malformed") == [".", "..", "000003.bin", "000003.json"]
    binary = "/tensors/malformed/000003.bin"
    assert fs.read(binary, _READ_ALL, 0) == serial.read(binary, _READ_ALL, 0) == content
    path = "/tensors/malformed/000003.json"
    fallback = fs.read(path, _READ_ALL, 0)
    assert fallback == serial.read(path, _READ_ALL, 0)
    encoded = json.loads(fallback)["tensor"]["tensor_content"]["base64"]
    assert b64decode(encoded) == content


def test_cold_tree_defers_reads_and_preserves_typed_files(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = _track_ranges(source, monkeypatch)
    reads = _track_reads(monkeypatch)
    serial = TensorBoardFS(source.parent)
    fs = _mount(source)
    _assert_cold_listing(fs, serial, starts)

    entries = {entry.step: entry for entry in _cache(fs, source).json_entries}
    first = _assert_source_backed(entries[1], source)
    bfloat = _assert_source_backed(entries[2], source)
    unrequested = _assert_source_backed(entries[8], source)
    assert not reads
    _assert_first_reads(fs, serial, reads, first, bfloat)
    _assert_typed_parity(fs, serial, 1, "main")
    _assert_typed_parity(fs, serial, 2, "bfloat")
    assert not any(extent is unrequested for extent in reads)
    assert all(extent is first or extent is bfloat for extent in reads)

    _assert_malformed_fallback(fs, serial, entries[3])


def test_first_read_of_old_tensor_after_append(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _track_reads(monkeypatch)
    path = "/tensors/main/000001.npy"
    fs = _mount(source)
    fs.readdir("/tensors/main")
    entry = next(entry for entry in _cache(fs, source).json_entries if entry.step == 1)
    assert isinstance(entry.tensor_source, TensorSource)
    assert not reads

    with source.open("ab") as stream:
        stream.write(_loss(9, 42.0))
    data = fs.read(path, _READ_ALL, 0)
    assert _cache(fs, source).progress.resumed
    assert len(reads) == 1 and reads[0] is entry.tensor_source
    assert data == TensorBoardFS(source.parent).read(path, _READ_ALL, 0)


def test_rewrite_after_lookup_invalidates_even_an_unchanged_extent(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = _mount(source)
    fs.readdir("/tensors/main")
    entry = next(entry for entry in _cache(fs, source).json_entries if entry.step == 1)
    extent = entry.tensor_source
    assert isinstance(extent, TensorSource)
    assert entry.tensor_extent is not None
    offset, size, _ = entry.tensor_extent
    original = source.read_bytes()
    old_frame = _tensor_frame(1)
    assert original.startswith(old_frame)
    rewritten = _tensor_frame(1, tag="gone") + original[len(old_frame) :]
    assert len(rewritten) == len(original)
    assert rewritten != original
    assert rewritten[offset : offset + size] == original[offset : offset + size]
    read = TensorSource.read

    def rewrite_after_lookup(selected: TensorSource) -> bytes:
        assert selected is extent
        source.write_bytes(rewritten)
        return read(selected)

    path = "/tensors/main/000001.npy"
    with monkeypatch.context() as late:
        late.setattr(TensorSource, "read", rewrite_after_lookup)
        with pytest.raises(OSError) as failure:
            fs.open(path, os.O_RDONLY)
        assert failure.value.errno == EIO
    assert not fs._state.open_files

    with pytest.raises(OSError) as refreshed:
        fs.read(path, _READ_ALL, 0)
    assert refreshed.value.errno == ENOENT
    moved = "/tensors/gone/000001.npy"
    assert fs.read(moved, _READ_ALL, 0) == TensorBoardFS(source.parent).read(
        moved, _READ_ALL, 0
    )


def test_open_handle_pins_rendered_tensor_across_source_replacement(
    source: Path,
) -> None:
    fs = _mount(source)
    path = "/tensors/main/000001.npy"
    fs.readdir("/tensors/main")
    assert any(
        entry.step == 1 and isinstance(entry.tensor_source, TensorSource)
        for entry in _cache(fs, source).json_entries
    )
    expected = TensorBoardFS(source.parent).read(path, _READ_ALL, 0)
    handle = fs.open(path, os.O_RDONLY)
    try:
        before = fs.getattr(path, handle)
        replacement = source.with_name("replacement")
        replacement.write_bytes(_tensor_frame(1, shift=2))
        replacement.replace(source)
        current = fs.read(path, _READ_ALL, 0)
        assert current != expected
        assert current == TensorBoardFS(source.parent).read(path, _READ_ALL, 0)
        assert fs.getattr(path)["st_mtime"] == 103.0
        assert fs.read(path, _READ_ALL, 0, handle) == expected
        assert fs.getattr(path, handle) == before
    finally:
        fs.release(path, handle)


def test_disk_index_roundtrip_before_first_tensor_read(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_extent_read(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("saving or restoring an index used the lazy render read")

    def unexpected_parse(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("the complete cold source did not reopen from its saved index")

    monkeypatch.setattr(TensorSource, "read", unexpected_extent_read)
    written = _mount(source, disk_index=True)
    names = written.readdir("/tensors/main")
    original = next(
        entry for entry in _cache(written, source).json_entries if entry.step == 1
    )
    assert isinstance(original.tensor_source, TensorSource)
    assert original.payload["tensor"]["tensor_content"] == b""

    serial = TensorBoardFS(source.parent)
    expected = {}
    for path in ("/tensors/main/000001.npy", "/tensors/bfloat/000002.json"):
        expected[path] = (serial.read(path, _READ_ALL, 0), serial.getattr(path))

    monkeypatch.setattr(_EventIndexer, "parse_event_file", unexpected_parse)
    monkeypatch.setattr(parallel_index, "parallel_index", unexpected_parse)
    restored = _mount(source, disk_index=True)
    assert restored.readdir("/tensors/main") == names
    entry = next(
        entry for entry in _cache(restored, source).json_entries if entry.step == 1
    )
    assert entry.tensor_source is None
    assert entry.payload["tensor"]["tensor_content"] == _values(1).tobytes()
    for path, (data, info) in expected.items():
        assert restored.read(path, _READ_ALL, 0) == data
        assert restored.getattr(path) == info
