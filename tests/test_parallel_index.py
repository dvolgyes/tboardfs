"""Cold range indexing matches serial frame, table, and mounted output."""

from collections.abc import Iterator
from multiprocessing import active_children
import os
from pathlib import Path
import struct
from typing import Never

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs import parallel_index as cold
from tboardfs.crc32c import masked_crc32c
from tboardfs.index_segment import Job, Metadata, Segment
from tboardfs.index_transport import _PackedResult
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache

from test_incremental_parse import _contents, _frame, _loss
from test_payload_spans import _PNG as PNG
from test_restart_purge import _start, _version
from test_tf2_summaries import _images

_STEP = 2**53 + 1
_SOURCE = "events.out.tfevents.host"


@pytest.fixture(autouse=True)
def available_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cold, "_available_cpus", lambda: 3)


def _frames(*, padding: int = 0) -> Iterator[bytes]:
    yield _version()
    first = _images(_STEP - 1, PNG + bytes(padding))
    first.wall_time = 20.0
    first.summary.value.add(tag="epoch", simple_value=3.0)
    yield _frame(first)
    for index, step in enumerate((_STEP + 1, _STEP, _STEP, _STEP + 2)):
        if index == 1:
            yield _start(_STEP)
        event = _images(step, PNG + bytes([index]), PNG + b"second")
        event.wall_time = 19.0 + index
        event.summary.value[0].ClearField("metadata")
        value = event.summary.value.add(tag="count", simple_value=2.5)
        if index != 1:
            value.ClearField("simple_value")
            value.tensor.dtype = 9  # DT_INT64
            value.tensor.int64_val.append(_STEP + index)
        event.summary.value.add(tag="count", simple_value=float(index))
        exact = event.summary.value.add(tag="exact").tensor
        exact.dtype = 9  # DT_INT64
        exact.int64_val.append(_STEP + index)
        narrow = event.summary.value.add(tag="narrow").tensor
        narrow.dtype = 17  # DT_UINT16
        narrow.int_val.append(40_000 + index)
        histogram = event.summary.value.add(tag="weights").histo
        histogram.min = 0.0
        histogram.bucket_limit.extend([1.0, 2.0])
        histogram.bucket.extend([index + 1.0, 2.0])
        if index == 0:
            event.summary.value.add(tag="epoch", simple_value=4.0)
        yield _frame(event)


def _assert_index_equal(actual: EventFileCache, expected: EventFileCache) -> None:
    assert actual.progress == expected.progress
    assert actual.restarts == expected.restarts
    assert actual.binaries == expected.binaries
    assert actual.graphs == expected.graphs
    assert actual.json_entries == expected.json_entries
    assert actual.scalar_mtimes == expected.scalar_mtimes
    assert actual.histogram_mtimes == expected.histogram_mtimes
    for actual_table, expected_table in (
        (actual.scalars, expected.scalars),
        (actual.histograms, expected.histograms),
    ):
        assert actual_table.keys() == expected_table.keys()
        for tag, expected_columns in expected_table.items():
            assert actual_table[tag].keys() == expected_columns.keys()
            for name, values in expected_columns.items():
                np.testing.assert_array_equal(
                    actual_table[tag][name], values, strict=True
                )


def test_parallel_index_matches_serial_ranges(tmp_path: Path) -> None:
    path = tmp_path / _SOURCE
    inherited = _images(_STEP - 2, PNG + b"inherited")
    inherited.summary.value[0].tag = "carried"
    inherited.summary.value[0].ClearField("metadata")
    path.write_bytes(_frame(inherited) + b"".join(_frames()))
    metadata = {"carried": {"plugin_name": "images", "plugin_content": b""}}

    indexed = cold.parallel_index(path, 3, initial_metadata=metadata)
    assert indexed is not None
    expected = _EventIndexer.parse_event_file(
        path, ignore_truncated=True, initial_metadata=metadata
    )
    _assert_index_equal(indexed, expected)
    assert indexed.scalars["exact"]["value"].tolist() == list(range(_STEP, _STEP + 4))
    assert indexed.scalars["count"]["epoch"].tolist() == [3.0, 3.0] + [4.0] * 6
    assert indexed.binaries[0].tag == "carried"
    assert indexed.scalars["count"]["step"].tolist() == [
        _STEP + 1,
        _STEP + 1,
        _STEP,
        _STEP,
        _STEP,
        _STEP,
        _STEP + 2,
        _STEP + 2,
    ]
    assert indexed.scalars["narrow"]["value"].dtype == np.dtype("uint16")


@pytest.mark.parametrize(
    "damage", ["header CRC", "payload CRC", "incomplete", "malformed protobuf"]
)
def test_parallel_index_rejects_damaged_tail(tmp_path: Path, damage: str) -> None:
    path = tmp_path / _SOURCE
    *prefix, last = _frames()
    if damage == "malformed protobuf":
        payload = b"\x80"
        length = struct.pack("<Q", len(payload))
        last = (
            length
            + struct.pack("<I", masked_crc32c(length))
            + payload
            + struct.pack("<I", masked_crc32c(payload))
        )
    elif damage == "incomplete":
        last = last[:-1]
    else:
        offset = 8 if damage == "header CRC" else len(last) - 1
        last = last[:offset] + bytes([last[offset] ^ 1]) + last[offset + 1 :]
    path.write_bytes(b"".join([*prefix, last]))

    assert cold.parallel_index(path, 3) is None


@pytest.mark.parametrize("mutation", ["append", "replace"])
def test_parallel_index_discards_mutation_during_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / _SOURCE
    data = b"".join(_loss(step, 1.0) for step in range(8))
    path.write_bytes(data)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(data)
    unpack = _PackedResult.unpack
    merged = 0

    def mutate(result: _PackedResult, *, mountpoint: Path | None = None) -> Segment:
        nonlocal merged
        segment = unpack(result, mountpoint=mountpoint)
        merged += 1
        if merged == 2:
            if mutation == "append":
                with path.open("ab") as handle:
                    handle.write(_loss(42, 9.0))
            else:
                os.replace(replacement, path)
        return segment

    monkeypatch.setattr(_PackedResult, "unpack", mutate)
    assert cold.parallel_index(path, 2) is None
    assert merged == 2


def test_mount_cold_parallel_then_reuses_and_resumes_serially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / _SOURCE
    frames = list(_frames(padding=8 << 20))
    parallel_index = cold.parallel_index
    attempted: list[EventFileCache | None] = []

    def record_cold(
        source: Path,
        workers: int,
        *,
        initial_metadata: Metadata | None = None,
        mountpoint: Path | None = None,
    ) -> EventFileCache | None:
        index = parallel_index(
            source, workers, initial_metadata=initial_metadata, mountpoint=mountpoint
        )
        attempted.append(index)
        return index

    monkeypatch.setattr(cold, "parallel_index", record_cold)
    path.write_bytes(b"".join(frames))
    serial = TensorBoardFS(tmp_path)
    expected = _contents(serial)
    mounted = TensorBoardFS(tmp_path, index_workers=2, refresh_age_seconds=0)
    assert _contents(mounted) == expected
    assert len(attempted) == 1 and attempted[0] is not None
    original = mounted._state.runs[()].files[path]
    raw = f"/raw/{_SOURCE}"
    raw_names = sorted(mounted.readdir(raw)[2:])
    assert [expected[f"{raw}/{name}"] for name in raw_names] == frames

    def unexpected_cold(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("an already indexed file was dispatched to cold workers")

    monkeypatch.setattr(cold, "parallel_index", unexpected_cold)
    assert _contents(mounted) == expected
    assert mounted._state.runs[()].files[path] is original
    with path.open("ab") as handle:
        handle.write(_loss(_STEP + 3, 9.0))
    assert _contents(mounted) == _contents(TensorBoardFS(tmp_path))
    assert mounted._state.runs[()].files[path].progress.resumed


def _exit_worker(job: Job) -> Never:
    Path(job[0]).with_name(f".worker-exited-{os.getpid()}").touch()
    os._exit(1)


def test_mount_worker_loss_falls_back_to_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / _SOURCE
    path.write_bytes(b"".join(_frames(padding=8 << 20)))
    expected = _contents(TensorBoardFS(tmp_path))
    parallel_index = cold.parallel_index
    attempts: list[EventFileCache | None] = []

    def record_attempt(
        source: Path,
        workers: int,
        *,
        initial_metadata: Metadata | None = None,
        mountpoint: Path | None = None,
    ) -> EventFileCache | None:
        indexed = parallel_index(
            source, workers, initial_metadata=initial_metadata, mountpoint=mountpoint
        )
        attempts.append(indexed)
        return indexed

    monkeypatch.setattr(cold, "packed_work", _exit_worker)
    monkeypatch.setattr(cold, "parallel_index", record_attempt)
    assert _contents(TensorBoardFS(tmp_path, index_workers=2)) == expected
    assert attempts == [None]
    markers = tmp_path.glob(".worker-exited-*")
    exited = {int(marker.name.rsplit("-", 1)[1]) for marker in markers}
    assert exited
    assert exited.isdisjoint(child.pid for child in active_children())


def test_mount_serves_serially_with_one_available_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / _SOURCE
    path.write_bytes(b"".join(_frames(padding=8 << 20)))
    expected = _contents(TensorBoardFS(tmp_path))

    def unexpected_pool(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("parallel indexing created a pool on a single CPU")

    monkeypatch.setattr(cold, "_available_cpus", lambda: 1)
    monkeypatch.setattr(cold, "ProcessPoolExecutor", unexpected_pool)
    assert _contents(TensorBoardFS(tmp_path, index_workers=2)) == expected
