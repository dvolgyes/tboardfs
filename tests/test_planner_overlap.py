"""Cold workers can finish a range before the planner validates the tail."""

import os
from itertools import accumulate, pairwise
from multiprocessing import active_children
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO

import pytest
from test_incremental_parse import _contents, _loss
from test_parallel_index import _SOURCE, _assert_index_equal

from tboardfs import TensorBoardFS, filesystem, index_segment
from tboardfs import parallel_index as cold
from tboardfs.disk_index import DiskIndex
from tboardfs.index_segment import Job, Segment
from tboardfs.index_transport import _PackedResult, packed_work
from tboardfs.indexer import _EventIndexer, _ParseState
from tboardfs.model import EventFileCache


def _mark_finished_range(job: Job) -> _PackedResult:
    result = packed_work(job)
    source, start, end, *_ = job
    Path(source).with_name(f".range-{start}-{end}-{os.getpid()}").touch()
    return result


class _PlannerGate:
    """Hold the final header until the first worker has parsed its range."""

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
            if offset == self.starts[-1]:
                self.handles.append(handle)
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
                    "the first range did not finish while the last header was unread"
                )
            sleep(0.01)

    def assert_finished(self) -> None:
        assert self.visited == self.starts
        assert len(self.handles) == 1 and self.handles[0].closed


def _guard_reduction(
    monkeypatch: pytest.MonkeyPatch, gate: _PlannerGate, bad_header: bool
) -> list[tuple[int, int]]:
    unpack = _PackedResult.unpack
    merge = cold._Reduction.merge
    merged: list[tuple[int, int]] = []

    def checked_unpack(
        result: _PackedResult, *, mountpoint: Path | None = None
    ) -> Segment:
        assert not bad_header
        gate.assert_finished()
        return unpack(result, mountpoint=mountpoint)

    def checked_merge(
        cache: EventFileCache, state: _ParseState, segment: Segment
    ) -> None:
        assert not bad_header
        gate.assert_finished()
        merged.append((segment.start, segment.end))
        merge(cache, state, segment)

    monkeypatch.setattr(_PackedResult, "unpack", checked_unpack)
    monkeypatch.setattr(cold._Reduction, "merge", staticmethod(checked_merge))
    return merged


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
    if bad_header:
        assert finished_ranges == expected_ranges[: len(finished_ranges)]
    else:
        assert finished_ranges == expected_ranges
    child_pids = {pid for _, _, pid in finished}
    assert os.getpid() not in child_pids
    assert child_pids.isdisjoint(child.pid for child in active_children())


@pytest.mark.parametrize("bad_header", [False, True], ids=["complete", "bad-tail"])
def test_real_worker_finishes_before_later_header(
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
    merged = _guard_reduction(monkeypatch, gate, bad_header)
    _guard_mount_paths(monkeypatch, bad_header)

    mounted = TensorBoardFS(root, index_workers=3, disk_index=bad_header)
    _assert_mount_matches_serial(mounted, expected, serial_index, prefix, bad_header)
    gate.assert_finished()
    assert merged == ([] if bad_header else expected_ranges)
    _assert_workers_finished(root, expected_ranges, bad_header)


@pytest.mark.parametrize(
    ("frame_count", "workers", "cuts"),
    [(3, 2, [0, 1, 3]), (4, 12, [0, 1, 2, 3, 4])],
    ids=["midpoint-tie-rounds-down", "duplicate-cuts-are-suppressed"],
)
def test_plan_preserves_exact_cuts(
    tmp_path: Path, frame_count: int, workers: int, cuts: list[int]
) -> None:
    source = tmp_path / _SOURCE
    frame = _loss(7, 1.0)
    source.write_bytes(frame * frame_count)
    ranges = list(Segment.plan(source, source.stat(), workers))
    assert ranges == [(lo * len(frame), hi * len(frame)) for lo, hi in pairwise(cuts)]


@pytest.mark.parametrize("data", [b"", _loss(7, 1.0)], ids=["empty", "one-range"])
def test_plan_declines_without_starting_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes
) -> None:
    source = tmp_path / _SOURCE
    source.write_bytes(data)

    def unexpected_pool(*args: object, **kwargs: object) -> None:
        pytest.fail("an empty or single-range plan must not start a pool")

    monkeypatch.setattr(cold, "_available_cpus", lambda: 3)
    monkeypatch.setattr(cold, "ProcessPoolExecutor", unexpected_pool)
    assert cold.parallel_index(source, 3) is None
