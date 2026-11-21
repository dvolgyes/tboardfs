"""Parallel workers own frame starts and validate complete crossing frames."""

import os
from itertools import accumulate, pairwise
from multiprocessing import active_children
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO

import pytest
from test_incremental_parse import _contents, _frame, _loss
from test_parallel_index import _SOURCE, PNG, _assert_index_equal
from test_tf2_summaries import _images

from tboardfs import TensorBoardFS, filesystem, index_segment
from tboardfs import parallel_index as cold
from tboardfs.disk_index import DiskIndex
from tboardfs.index_segment import Job, Metadata, Segment, _SegmentError
from tboardfs.index_transport import _PackedResult, packed_work
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache


def _mark_finished_range(job: Job) -> _PackedResult:
    source = Path(job[0])
    try:
        result = packed_work(job)
    except _SegmentError:
        source.with_name(f".failed-range-{os.getpid()}").touch()
        raise
    start, end = result.segment.start, result.segment.end
    source.with_name(f".range-{start}-{end}-{os.getpid()}").touch()
    return result


class _PlannerGate:
    """Hold the header that would reveal the first threshold's frame start."""

    root: Path
    starts: list[int]
    visited: list[int]
    handles: list[BinaryIO]

    def __init__(self, root: Path, starts: list[int]) -> None:
        self.root = root
        self.starts = starts
        self.visited = []
        self.handles = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        read_header = index_segment._read_header

        def gated_header(
            handle: BinaryIO,
            offset: int,
            end: int,
            max_record_size: int,
            previous: tuple[bytes, int] | None,
        ) -> tuple[bytes, int]:
            if handle not in self.handles:
                self.handles.append(handle)
            if offset == self.starts[2]:
                self._await_first_range()
            self.visited.append(offset)
            return read_header(handle, offset, end, max_record_size, previous)

        monkeypatch.setattr(index_segment, "_read_header", gated_header)

    def _await_first_range(self) -> None:
        marker = f".range-0-{self.starts[3]}-*"
        deadline = monotonic() + 30
        while not any(self.root.glob(marker)):
            if monotonic() >= deadline:
                pytest.fail(
                    "the first worker did not finish before the first cut was known"
                )
            sleep(0.01)

    def assert_finished(self) -> None:
        assert self.starts[2] in self.visited
        assert set(self.visited) <= set(self.starts[:6])
        assert all(handle.closed for handle in self.handles)


def _guard_mount_paths(monkeypatch: pytest.MonkeyPatch, bad_header: bool) -> None:
    def unexpected_save(*args: object, **kwargs: object) -> None:
        pytest.fail("the damaged snapshot must not be saved to the disk index")

    def unexpected_serial(*args: object, **kwargs: object) -> None:
        pytest.fail("the complete snapshot must use its parallel index")

    if bad_header:
        monkeypatch.setattr(DiskIndex, "save", unexpected_save)
    else:
        monkeypatch.setattr(_EventIndexer, "parse_event_file", unexpected_serial)


def _assert_mount_matches_serial(
    mounted: TensorBoardFS,
    expected: dict[str, bytes],
    serial_index: EventFileCache,
    prefix: list[bytes],
    bad_header: bool,
) -> None:
    assert (mounted._state.index.disk is not None) is bad_header
    contents = _contents(mounted)
    assert contents == expected
    raw = f"/raw/{_SOURCE}/"
    assert [
        contents[name] for name in sorted(contents) if name.startswith(raw)
    ] == prefix
    actual = mounted._state.runs[()].files[serial_index.source_path]
    _assert_index_equal(actual, serial_index)
    assert actual.progress.processed_pos == sum(map(len, prefix))
    assert actual.progress.provenance.cacheable is not bad_header


def _assert_workers_finished(
    root: Path, expected_ranges: list[tuple[int, int]], bad_header: bool
) -> None:
    finished = [
        tuple(map(int, marker.name.split("-")[1:])) for marker in root.glob(".range-*")
    ]
    assert finished
    finished_ranges = sorted((start, end) for start, end, _ in finished)
    failed_pids = {
        int(marker.name.rsplit("-", 1)[1]) for marker in root.glob(".failed-range-*")
    }
    assert len(failed_pids) == int(bad_header)
    if bad_header:
        assert set(finished_ranges) <= set(expected_ranges[:-1])
        assert expected_ranges[0] in finished_ranges
    else:
        assert finished_ranges == expected_ranges
    child_pids = {pid for _, _, pid in finished}
    child_pids.update(failed_pids)
    assert os.getpid() not in child_pids
    assert child_pids.isdisjoint(child.pid for child in active_children())


@pytest.mark.parametrize("bad_header", [False, True], ids=["complete", "bad-tail"])
def test_real_worker_finishes_before_first_cut_is_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_header: bool
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    source = root / _SOURCE
    frames = [_loss(7, float(value)) for value in range(1, 10)]
    starts = [0, *accumulate(map(len, frames[:-1]))]
    expected_ranges = list(pairwise([0, starts[3], starts[6], sum(map(len, frames))]))
    prefix = frames[:-1] if bad_header else frames
    tail = frames[-1]
    if bad_header:
        tail = tail[:8] + bytes([tail[8] ^ 1]) + tail[9:]
    source.write_bytes(b"".join(frames[:-1]) + tail)
    serial = TensorBoardFS(root, index_workers=1)
    expected = _contents(serial)
    serial_index = serial._state.runs[()].files[source]

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(cold, "_available_cpus", lambda: 3)
    monkeypatch.setattr(cold, "packed_work", _mark_finished_range)
    gate = _PlannerGate(root, starts)
    gate.install(monkeypatch)
    parallel_index = cold.parallel_index
    attempts: list[EventFileCache | None] = []

    def observe(
        path: Path,
        workers: int,
        *,
        initial_metadata: Metadata | None = None,
        mountpoint: Path | None = None,
    ) -> EventFileCache | None:
        index = parallel_index(
            path, workers, initial_metadata=initial_metadata, mountpoint=mountpoint
        )
        attempts.append(index)
        return index

    monkeypatch.setattr(cold, "parallel_index", observe)
    _guard_mount_paths(monkeypatch, bad_header)

    mounted = TensorBoardFS(root, index_workers=3, disk_index=bad_header)
    _assert_mount_matches_serial(mounted, expected, serial_index, prefix, bad_header)
    gate.assert_finished()
    assert len(attempts) == 1
    assert (attempts[0] is None) is bad_header
    _assert_workers_finished(root, expected_ranges, bad_header)


@pytest.mark.parametrize(
    ("layout", "workers", "ownership"),
    [
        ("exact", 3, [(0, 0), (2, 1), (4, 2)]),
        ("straddling", 2, [(0, 0), (2, 1)]),
        ("spanning", 8, [(0, 0), (2, 7)]),
        ("spanning-tail", 8, [(0, 0), (1, 4)]),
    ],
)
def test_plan_and_workers_own_frame_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    workers: int,
    ownership: list[tuple[int, int]],
) -> None:
    source = tmp_path / _SOURCE
    small = [_loss(7, float(value)) for value in range(1, 7)]
    large = _frame(_images(7, PNG + b"a" * 4096))
    trailing = _frame(_images(7, PNG + b"b" * 3072))
    frames = {
        "exact": small,
        "straddling": small[:3],
        "spanning": [small[0], large, *small[1:3]],
        "spanning-tail": [large, small[0], trailing],
    }[layout]
    source.write_bytes(b"".join(frames))
    size = source.stat().st_size
    starts = [0, *accumulate(map(len, frames[:-1]))]
    planned = [
        (starts[frame], ((bucket + 1) * size + workers - 1) // workers)
        for frame, bucket in ownership
    ]
    assert list(Segment.plan(source, source.stat(), workers)) == planned

    serial = TensorBoardFS(tmp_path, index_workers=1)
    expected = _contents(serial)
    serial_index = serial._state.runs[()].files[source]
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 1)
    monkeypatch.setattr(cold, "_available_cpus", lambda: workers)
    monkeypatch.setattr(cold, "packed_work", _mark_finished_range)
    _guard_mount_paths(monkeypatch, False)

    mounted = TensorBoardFS(tmp_path, index_workers=workers)
    _assert_mount_matches_serial(mounted, expected, serial_index, frames, False)
    actual_ranges = list(pairwise([start for start, _ in planned] + [size]))
    _assert_workers_finished(tmp_path, actual_ranges, False)


@pytest.mark.parametrize("data", [b"", _loss(7, 1.0)], ids=["empty", "single-frame"])
def test_plan_declines_without_starting_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes
) -> None:
    source = tmp_path / _SOURCE
    source.write_bytes(data)

    def unexpected_pool(*args: object, **kwargs: object) -> None:
        pytest.fail("an empty or single-frame file must not start a pool")

    monkeypatch.setattr(cold, "_available_cpus", lambda: 3)
    monkeypatch.setattr(cold, "ProcessPoolExecutor", unexpected_pool)
    assert cold.parallel_index(source, 3) is None
