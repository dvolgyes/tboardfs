"""Cold range indexing matches serial frame, table, and mounted output."""

import json
import os
import struct
from collections.abc import Iterator
from multiprocessing import active_children
from pathlib import Path
from typing import Never

import numpy as np
import pytest
from test_incremental_parse import _contents, _frame, _loss
from test_payload_spans import _PNG as PNG
from test_restart_purge import _start, _version
from test_tf2_summaries import _images

from tboardfs import TensorBoardFS, filesystem
from tboardfs import parallel_index as cold
from tboardfs.crc32c import masked_crc32c
from tboardfs.index_segment import Job, Metadata, Segment
from tboardfs.index_transport import _PackedResult
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache, JsonEntry
from tboardfs.proto import compat_pb2

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


@pytest.mark.parametrize("workers", [1, 4])
def test_legacy_scalar_image_pairs_keep_order_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    monkeypatch.setattr(cold, "_available_cpus", lambda: 4)
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 0)
    first = -(2**63) + 1
    seed = compat_pb2.Event(step=first, wall_time=100.0)
    seed.summary.value.add(tag="epoch", simple_value=3)
    frames = [_frame(seed)]
    filler = _loss(0, 1.0)
    reset = compat_pb2.Event(step=first + 3, wall_time=104.0)
    reset.summary.value.add(tag="epoch", simple_value=3)

    cases = (
        (False, "plain", False),
        (True, "plain", False),
        (False, "annotated", True),
        (True, "annotated", False),
        (False, "annotated", False),
        (True, "annotated", True),
        (False, "fresh", True),
    )
    for offset, (image_first, tag, sets_metadata) in enumerate(cases):
        event = compat_pb2.Event(step=first + offset, wall_time=101.0 + offset)
        scalar = compat_pb2.Summary.Value(tag="metric", simple_value=offset + 1)
        image = compat_pb2.Summary.Value(tag=tag)
        image.image.encoded_image_string = PNG
        if sets_metadata:
            image.metadata.plugin_data.plugin_name = "annotations"
        if offset == 2:
            frames.extend([filler] * cold._PREFIX_RECORDS)
            epoch = compat_pb2.Summary.Value(tag="epoch", simple_value=7)
            event.summary.value.extend([scalar, epoch, image])
            frames.append(_frame(event))
            frames.append(_frame(reset))
            frames.extend([filler] * (4 * cold._PREFIX_RECORDS))
        else:
            event.summary.value.extend(
                [image, scalar] if image_first else [scalar, image]
            )
            frames.append(_frame(event))

    (tmp_path / _SOURCE).write_bytes(b"".join(frames))
    fs = TensorBoardFS(tmp_path, index_workers=workers)
    rows = json.loads(fs.read("/scalars/metric.json", 4096, 0))
    assert [
        (r["step"], r["value"], r["epoch"], r["wall_time"], r["relative_time"])
        for r in rows
    ] == [
        (first, 1, 3, 101, 1),
        (first + 1, 2, 3, 102, 2),
        (first + 2, 3, 7, 103, 3),
        (first + 3, 4, 3, 104, 4),
        (first + 4, 5, 3, 105, 5),
        (first + 5, 6, 3, 106, 6),
        (first + 6, 7, 3, 107, 7),
    ]
    epochs = json.loads(fs.read("/scalars/epoch.json", 4096, 0))
    assert [(r["step"], r["value"]) for r in epochs] == [
        (first, 3),
        (first + 2, 7),
        (first + 3, 3),
    ]
    assert fs.getattr("/scalars/metric.json")["st_mtime"] == 107.0
    for offset, (_, tag, _) in enumerate(cases):
        step = first + offset
        media = f"/images/{tag}/{step}.png"
        assert fs.read(media, 4096, 0) == PNG
        assert fs.getattr(media)["st_mtime"] == 101.0 + offset
        if offset >= 2:
            plugin = f"/plugins/annotations/{tag}/{step}.json"
            assert json.loads(fs.read(plugin, 4096, 0)) == {
                "plugin_name": "annotations",
                "tag": tag,
            }


def test_legacy_scalar_metadata_declarations_follow_value_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cold, "_available_cpus", lambda: 4)
    seed, custom, restored = (
        compat_pb2.Event(step=_STEP + offset, wall_time=100.0 + offset)
        for offset in range(3)
    )
    first = seed.summary.value.add(tag="metric", simple_value=1)
    first.metadata.plugin_data.plugin_name = "SCALARS"
    first.metadata.plugin_data.content = b"first"

    custom.summary.value.add(tag="metric", simple_value=2)
    declaration = custom.summary.value.add(tag="metric").metadata.plugin_data
    declaration.plugin_name = "annotations"
    declaration.content = b"custom"
    custom.summary.value.add(tag="metric", simple_value=3).metadata.SetInParent()
    empty = custom.summary.value.add(tag="metric", simple_value=4)
    empty.metadata.plugin_data.SetInParent()
    declaration = custom.summary.value.add(tag="metric").metadata.plugin_data
    declaration.plugin_name = "annotations"

    restored.summary.value.add(tag="metric", simple_value=5)
    declaration = restored.summary.value.add(tag="metric").metadata.plugin_data
    declaration.plugin_name = "ScAlArS"
    declaration.content = b"restored"
    restored.summary.value.add(tag="metric", simple_value=6)

    path = tmp_path / _SOURCE
    filler = _loss(0, 1.0) * cold._PREFIX_RECORDS
    path.write_bytes(
        _frame(seed) + filler + _frame(custom) + filler * 4 + _frame(restored)
    )
    serial = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    parallel = cold.parallel_index(path, 4)
    assert parallel is not None
    _assert_index_equal(parallel, serial)

    rows = serial.scalars["metric"]
    np.testing.assert_array_equal(
        rows["value"], np.arange(1, 7, dtype="float32"), strict=True
    )
    assert list(
        zip(rows["step"], rows["wall_time"], rows["relative_time"], strict=True)
    ) == [
        (_STEP, 100.0, 0.0),
        *[(_STEP + 1, 101.0, 1.0)] * 3,
        *[(_STEP + 2, 102.0, 2.0)] * 2,
    ]
    annotated = {"plugin_name": "annotations", "tag": "metric"}
    annotated_content = annotated | {"plugin_content": b"custom"}
    assert serial.json_entries == [
        JsonEntry("metric", "plugins", _STEP + 1, 101.0, annotated_content),
        JsonEntry("metric", "plugins", _STEP + 1, 101.0, annotated_content),
        JsonEntry("metric", "plugins", _STEP + 1, 101.0, annotated),
        JsonEntry("metric", "plugins", _STEP + 2, 102.0, annotated),
        JsonEntry(
            "metric",
            "scalars",
            _STEP + 2,
            102.0,
            {"plugin_name": "ScAlArS", "tag": "metric", "plugin_content": b"restored"},
        ),
    ]


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
