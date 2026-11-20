"""Parity and bounds of the persistent index body."""

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from tboardfs.crc32c import masked_crc32c
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.index_codec import IndexCodec
from tboardfs.index_wire import _IndexReader
from tboardfs.model.event_file_cache import EventFileCache
from tboardfs.model.graph_entry import GraphEntry
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.model.json_entry import JsonEntry
from tboardfs.model.meta_graph_entry import MetaGraphEntry
from tboardfs.model.restart_mark import RestartMark
from tboardfs.range_proof import range_proof
from tboardfs.tfrecord import DEFAULT_MAX_RECORD_SIZE


def _table(*, histogram: bool = False) -> dict[str, np.ndarray]:
    columns = {
        "step": np.array([2**60 + 1], dtype=np.int64),
        "epoch": np.array([np.nan]),
        "wall_time": np.array([12.0]),
        "relative_time": np.array([0.0]),
        "_inherits_epoch": np.array([True]),
    }
    if histogram:
        columns.update(
            _start=np.array([True]),
            bucket_left=np.array([1.0]),
            bucket_right=np.array([2.0]),
            count=np.array([3.0]),
        )
    else:
        columns["value"] = np.array([2**63 + 1], dtype=np.uint64)
    return columns


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[EventFileCache, bytes]:
    path = tmp_path / "original"
    content = bytes(range(256)) * 256
    graph = b"graph"
    payload = content + graph
    crc = masked_crc32c(payload)
    length = struct.pack("<Q", len(payload))
    source = length + struct.pack("<I", masked_crc32c(length)) + payload
    source += struct.pack("<I", crc)
    frame_size = len(source)
    empty_length = struct.pack("<Q", 0)
    empty_crc = masked_crc32c(b"")
    source += empty_length + struct.pack("<II", masked_crc32c(empty_length), empty_crc)
    path.write_bytes(source)
    cache = EventFileCache(path)
    progress = cache.progress
    progress.identity = FileIdentity.of(path.stat())
    progress.checked_status = FileIdentity.fingerprint(path.stat())
    progress.processed_pos = progress.last_checked_size = len(source)
    progress.first_time = 12.0
    progress.epoch = np.nan
    progress.initial_metadata = {
        "p": {"plugin_content": b"\x00\xff", "extra": {7: [1, None]}}
    }
    progress.effective_metadata = {
        **progress.initial_metadata,
        "new": {"plugin_name": "text"},
    }
    progress.frames.append(frame_size, crc)
    progress.frames.append(len(source), empty_crc)
    cache.scalars["value"] = _table()
    cache.histograms["h"] = _table(histogram=True)
    cache.scalar_mtimes["value"] = cache.histogram_mtimes["h"] = 12.0
    cache.histograms["empty"] = {
        name: column[:0] for name, column in cache.histograms["h"].items()
    }
    cache.histogram_mtimes["empty"] = 12.0
    frame = FrameRef(progress.identity, 0, frame_size, crc, progress.checked_status)
    blob = IndexedBlob(
        frame, len(graph), (len(content), len(graph)), range_proof(graph)
    )
    cache.graphs = [
        GraphEntry(blob, path, 12.0),
        MetaGraphEntry(blob, path, 13.0, True),
    ]
    cache.binaries = [
        GraphSummaryEntry(
            blob,
            path,
            "model",
            "graphs",
            2**60 + 1,
            13.0,
            "pb",
            0,
            converts=True,
            profiles=True,
        )
    ]
    cache.json_entries = [
        JsonEntry(
            "p",
            "tensors",
            2**60 + 1,
            12.0,
            {"tensor": {"tensor_content": content, "dtype": 4}},
            tensor_extent=(12, len(content), masked_crc32c(content)),
        )
    ]
    cache.restarts = [
        RestartMark("version", 2.0),
        RestartMark("start", 2**60 + 1, {("scalars", "value"): 1}),
    ]
    return cache, source


def test_mixed_index_round_trip(
    indexed: tuple[EventFileCache, bytes], tmp_path: Path
) -> None:
    cache, source = indexed
    entry = cache.json_entries[0]
    content = entry.payload["tensor"]["tensor_content"]
    body = IndexCodec.encode(cache)
    assert content not in body
    assert entry.payload["tensor"]["tensor_content"] is content
    moved = tmp_path / "copy"
    moved.write_bytes(source)
    with moved.open("rb") as handle:
        status = os.fstat(handle.fileno())
        result = IndexCodec.decode(
            body, moved, status, handle, cache.progress.initial_metadata
        )
    assert result is not None
    assert result.source_path == moved
    assert result.progress.checked_status == FileIdentity.fingerprint(status)
    assert result.progress.identity != cache.progress.identity
    assert list(result.progress.frames) == list(cache.progress.frames)
    _assert_rows(result, cache)
    _assert_graphs(result, moved)
    _assert_tables(result, cache)


def _assert_rows(result: EventFileCache, cache: EventFileCache) -> None:
    assert result.progress.effective_metadata == cache.progress.effective_metadata
    assert np.isnan(result.progress.epoch)
    assert result.progress.first_time == 12.0 and not result.progress.resumed
    assert result.json_entries == cache.json_entries
    assert result.json_entries[0].tensor_extent == cache.json_entries[0].tensor_extent
    assert result.restarts == cache.restarts


def _assert_graphs(result: EventFileCache, moved: Path) -> None:
    assert isinstance(result.graphs[1], MetaGraphEntry) and result.graphs[1].readable
    binary = result.binaries[0]
    assert isinstance(binary, GraphSummaryEntry)
    assert binary.converts and binary.profiles
    assert binary.step == 2**60 + 1
    assert binary.source_path == moved and binary.frame == result.graphs[0].frame
    for graph in result.graphs:
        assert graph.source_path == moved
        assert graph.frame.identity == result.progress.identity
        assert graph.frame.fingerprint == result.progress.checked_status


def _assert_tables(result: EventFileCache, cache: EventFileCache) -> None:
    assert result.scalar_mtimes == cache.scalar_mtimes
    assert result.histogram_mtimes == cache.histogram_mtimes
    for expected, actual in (
        (cache.scalars, result.scalars),
        (cache.histograms, result.histograms),
    ):
        for tag, columns in expected.items():
            assert list(actual[tag]) == list(columns)
            for name, values in columns.items():
                assert actual[tag][name].dtype == values.dtype
                np.testing.assert_array_equal(actual[tag][name], values)


def test_rejects_invalid_body_and_changed_tensor(
    indexed: tuple[EventFileCache, bytes],
) -> None:
    cache, source = indexed
    path = cache.source_path
    body = IndexCodec.encode(cache)
    with path.open("rb") as handle:
        status = os.fstat(handle.fileno())
        assert IndexCodec.decode(body, path, status, handle, {}) is None
        changed_types = {
            "p": {"plugin_content": b"\x00\xff", "extra": {7: [True, None]}}
        }
        assert IndexCodec.decode(body, path, status, handle, changed_types) is None
        for invalid in (body[:-1], body + b"x"):
            with pytest.raises(ValueError):
                IndexCodec.decode(
                    invalid, path, status, handle, cache.progress.initial_metadata
                )
    path.write_bytes(source[:12] + b"z" + source[13:])
    with path.open("rb") as handle, pytest.raises(ValueError, match="tensor"):
        IndexCodec.decode(
            body,
            path,
            os.fstat(handle.fileno()),
            handle,
            cache.progress.initial_metadata,
        )


def test_refuses_object_array(indexed: tuple[EventFileCache, bytes]) -> None:
    cache, _ = indexed
    cache.scalars["value"]["value"] = np.array([object()], dtype=object)
    with pytest.raises(ValueError, match="dtype"):
        IndexCodec.encode(cache)


def test_rejects_overlapping_tensor_extents(
    indexed: tuple[EventFileCache, bytes],
) -> None:
    cache, _ = indexed
    cache.json_entries.append(cache.json_entries[0])
    body = IndexCodec.encode(cache)
    path = cache.source_path
    with path.open("rb") as handle:
        with pytest.raises(ValueError, match="overlap"):
            IndexCodec.decode(
                body,
                path,
                os.fstat(handle.fileno()),
                handle,
                cache.progress.initial_metadata,
            )
        assert handle.tell() == 0


def test_empty_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.touch()
    cache = EventFileCache(path)
    body = IndexCodec.encode(cache)
    with path.open("rb") as handle:
        result = IndexCodec.decode(body, path, os.fstat(handle.fileno()), handle, {})
    assert result is not None
    assert list(result.progress.frames) == []
    assert result.progress.processed_pos == 0


def test_encoding_respects_byte_budget(indexed: tuple[EventFileCache, bytes]) -> None:
    cache, _ = indexed
    body = IndexCodec.encode(cache)
    assert IndexCodec.encode(cache, limit=len(body)) == body
    for limit in (0, 128, len(body) - 1):
        with pytest.raises(ValueError, match="budget"):
            IndexCodec.encode(cache, limit=limit)


def test_rejects_overlarge_cached_frame(tmp_path: Path) -> None:
    path = tmp_path / "oversized"
    cache = EventFileCache(path)
    progress = cache.progress
    progress.processed_pos = progress.last_checked_size = 16
    progress.frames.append(16, 0)
    body = bytearray(IndexCodec.encode(cache))
    reader = _IndexReader(bytes(body))
    reader.take(8)
    reader.value()
    source_size_pos = len(body) - len(reader.remaining)
    reader.count()
    for _ in range(3):
        reader.value()
    reader.count()
    frame_end_pos = len(body) - len(reader.remaining)
    oversized = DEFAULT_MAX_RECORD_SIZE + 17
    struct.pack_into("<Q", body, source_size_pos, oversized)
    struct.pack_into("<Q", body, frame_end_pos, oversized)
    with path.open("w+b") as handle:
        handle.truncate(oversized)
        status = os.fstat(handle.fileno())
        with pytest.raises(ValueError, match="maximum record size"):
            IndexCodec.decode(bytes(body), path, status, handle, {})
        assert handle.tell() == 0
