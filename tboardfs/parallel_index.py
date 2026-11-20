"""Index complete cold event files in isolated workers and reduce in file order."""

import os
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from errno import EAGAIN, EMFILE, ENFILE, ENOMEM
from multiprocessing import current_process, get_context
from pathlib import Path
from struct import pack
from typing import Any, BinaryIO

import numpy as np
from loguru import logger

from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.index_segment import (
    _SOURCE_RACE_ERRNOS,
    Fingerprint,
    Job,
    Metadata,
    Segment,
    _check_snapshot,
    _index_record,
    _SegmentError,
)
from tboardfs.index_transport import _PackedResult, packed_work
from tboardfs.indexer import _ParseState
from tboardfs.model import EventFileCache, TFRecordError
from tboardfs.restarts import _offset_restarts
from tboardfs.tables import _TableRows
from tboardfs.tfrecord import _iter_tfrecords

_PREFIX_RECORDS = 128
_PREFIX_BYTES = 4 << 20
_FLOAT64 = np.dtype("float64")
_POOL_RESOURCE_ERRNOS = frozenset({EAGAIN, ENOMEM, EMFILE, ENFILE})


class _PoolUnavailableError(Exception):
    """The executor could not allocate resources to launch work."""


class _Seed:
    """Calibrate speculative inputs and validate inherited dependencies."""

    @staticmethod
    def guess(
        path: Path,
        end: int,
        metadata: Metadata,
        fingerprint: Fingerprint,
        mountpoint: Path | None,
    ) -> tuple[float | None, object, Metadata]:
        """Estimate the clock and inherited metadata from a validated prefix."""
        state = _ParseState()
        cache = EventFileCache(path)
        cache.progress.effective_metadata = dict(metadata)
        with EventSource.open(path, mountpoint=mountpoint) as handle:
            status = os.fstat(handle.fileno())
            _check_snapshot(path, status, fingerprint)
            cache.progress.identity = FileIdentity.of(status)
            cache.progress.checked_status = fingerprint
            _Seed._scan_prefix(handle, end, cache, state)
            _check_snapshot(path, os.fstat(handle.fileno()), fingerprint)
        return state.first_time, state.epoch, cache.progress.effective_metadata

    @staticmethod
    def _scan_prefix(
        handle: BinaryIO, end: int, cache: EventFileCache, state: _ParseState
    ) -> None:
        limit = min(end, _PREFIX_BYTES)
        records = _iter_tfrecords(handle, end=end)
        count = 0
        while count < _PREFIX_RECORDS and handle.tell() < limit:
            _index_record(cache, next(records), state)
            count += 1

    @staticmethod
    def _same_clock(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(left, float) and isinstance(right, float):
            return pack("!d", left) == pack("!d", right)
        if isinstance(left, np.generic) and isinstance(right, np.generic):
            return bool(left.tobytes() == right.tobytes())
        return bool(left == right)

    @staticmethod
    def matches(segment: Segment, state: _ParseState, metadata: Metadata) -> bool:
        """Check inputs that the segment used against the actual prefix."""
        clock = segment.clock
        return (
            (
                not clock.reads_first_time
                or _Seed._same_clock(clock.first_time, state.first_time)
            )
            and (not clock.reads_epoch or _Seed._same_clock(clock.epoch, state.epoch))
            and all(
                metadata.get(tag) == seed
                for tag, seed in segment.metadata_reads.items()
            )
        )


class _Tables:
    """Accumulate mutable columns and materialize them after ordered reduction."""

    @staticmethod
    def merge_scalars(state: _ParseState, segment: Segment) -> None:
        """Append scalar columns, promoting mixed value dtypes."""
        for tag, rows in segment.scalar_rows.items():
            if tag not in state.scalar_rows:
                state.scalar_rows[tag] = rows
                continue
            dest = state.scalar_rows[tag]
            if dest.dtype != rows.dtype:
                dest.dtype = _FLOAT64
            dest.epoch.extend(rows.epoch)
            dest.step.extend(rows.step)
            dest.wall_time.extend(rows.wall_time)
            dest.relative_time.extend(rows.relative_time)
            dest.value.extend(rows.value)

    @staticmethod
    def merge_histograms(state: _ParseState, segment: Segment) -> None:
        """Append histogram columns in encounter order."""
        for tag, rows in segment.histogram_rows.items():
            if tag not in state.histogram_rows:
                state.histogram_rows[tag] = rows
                continue
            _Tables._extend_histogram(state.histogram_rows[tag], rows)

    @staticmethod
    def _extend_histogram(
        target: dict[str, list[Any]], source: dict[str, list[Any]]
    ) -> None:
        for key, column in source.items():
            target[key].extend(column)

    @staticmethod
    def finish(cache: EventFileCache, state: _ParseState) -> None:
        """Build the same column arrays and modification times as serial indexing."""
        for tag, rows in state.scalar_rows.items():
            cache.scalars[tag] = rows.arrays()
            cache.scalar_mtimes[tag] = rows.wall_time[-1]
        cache.histograms = _TableRows.arrays_from_histogram_rows(state.histogram_rows)


def _available_cpus() -> int:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        with suppress(OSError):
            return len(get_affinity(0))
    return os.cpu_count() or 1


class _Reduction:
    """Consume worker ranges strictly in order and repair incompatible seeds."""

    @staticmethod
    def run(cache: EventFileCache, state: _ParseState, jobs: list[Job]) -> None:
        """Reduce all jobs, releasing each result and closing the executor."""
        metadata = cache.progress.effective_metadata
        executor = _Reduction._new_pool(len(jobs))
        pending: deque[Future[_PackedResult]] = deque()
        try:
            pending.extend(_Reduction._submit(executor, job) for job in jobs)
            for job in jobs:
                _Reduction.merge(
                    cache,
                    state,
                    _Reduction._corrected(
                        _Reduction.worker_segment(pending.popleft(), job),
                        job,
                        state,
                        metadata,
                    ),
                )
        finally:
            pending.clear()
            executor.shutdown(wait=True, cancel_futures=True)
        cache.progress.first_time = state.first_time
        cache.progress.epoch = state.epoch
        _Tables.finish(cache, state)

    @staticmethod
    def worker_segment(future: Future[_PackedResult], job: Job) -> Segment:
        """Read a completed result and reconstruct its compact fields."""
        return future.result().unpack(mountpoint=job[-1])

    @staticmethod
    def merge(cache: EventFileCache, state: _ParseState, segment: Segment) -> None:
        """Append one segment to the settled prefix."""
        _Reduction._validate_segment(cache, segment)
        chunk = segment.cache
        progress = cache.progress
        frames = chunk.progress.frames
        if chunk.restarts:
            _Reduction._merge_restarts(cache, state, segment)
        progress.processed_pos = segment.end
        progress.frames.ends.extend(frames.ends)
        progress.frames.checksums.extend(frames.checksums)
        progress.effective_metadata.update(segment.metadata_writes)
        if segment.clock.writes_first_time:
            state.first_time = chunk.progress.first_time
        if segment.clock.writes_epoch:
            state.epoch = chunk.progress.epoch
        cache.binaries.extend(chunk.binaries)
        cache.json_entries.extend(chunk.json_entries)
        cache.graphs.extend(chunk.graphs)
        cache.histogram_mtimes.update(chunk.histogram_mtimes)
        _Tables.merge_scalars(state, segment)
        _Tables.merge_histograms(state, segment)

    @staticmethod
    def _merge_restarts(
        cache: EventFileCache, state: _ParseState, segment: Segment
    ) -> None:
        marks = segment.cache.restarts
        if not any(mark.kind == "start" for mark in marks):
            cache.restarts.extend(marks)
            return
        counts = {
            ("scalars", tag): len(rows.step) for tag, rows in state.scalar_rows.items()
        }
        counts.update(
            (("histograms", tag), len(rows["step"]))
            for tag, rows in state.histogram_rows.items()
        )
        counts["binaries", ""] = len(cache.binaries)
        counts["json_entries", ""] = len(cache.json_entries)
        cache.restarts.extend(_offset_restarts(marks, counts))

    @staticmethod
    def _validate_segment(cache: EventFileCache, segment: Segment) -> None:
        chunk = segment.cache
        progress = cache.progress
        frames = chunk.progress.frames
        if (
            segment.start != progress.processed_pos
            or segment.end != chunk.progress.processed_pos
            or not frames.ends
            or frames.ends[0] <= segment.start
            or frames.ends[-1] != segment.end
            or len(frames.ends) != len(frames.checksums)
            or chunk.progress.identity != progress.identity
            or chunk.progress.checked_status != progress.checked_status
        ):
            raise _SegmentError(
                f"{cache.source_path}: discontinuous or mismatched segment"
            )

    @staticmethod
    def _corrected(
        segment: Segment, job: Job, state: _ParseState, metadata: Metadata
    ) -> Segment:
        path, start, end, fingerprint, _, _, _, mountpoint = job
        if (segment.start, segment.end) != (start, end):
            raise _SegmentError(f"{path}: unexpected worker range")
        if _Seed.matches(segment, state, metadata):
            return segment
        del segment
        segment = Segment.work(
            (
                path,
                start,
                end,
                fingerprint,
                state.first_time,
                state.epoch,
                metadata,
                mountpoint,
            )
        )
        if not _Seed.matches(segment, state, metadata):
            raise _SegmentError(f"{path}: corrective parse used a mismatched seed")
        return segment

    @staticmethod
    def _new_pool(workers: int) -> ProcessPoolExecutor:
        try:
            return ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            )
        except OSError as error:
            _Reduction._raise_if_pool_unavailable(error)
            raise

    @staticmethod
    def _submit(executor: ProcessPoolExecutor, job: Job) -> Future[_PackedResult]:
        try:
            return executor.submit(packed_work, job)
        except OSError as error:
            _Reduction._raise_if_pool_unavailable(error)
            raise

    @staticmethod
    def _raise_if_pool_unavailable(error: OSError) -> None:
        if error.errno in _POOL_RESOURCE_ERRNOS:
            raise _PoolUnavailableError(str(error)) from error


def _index_snapshot(
    path: Path, workers: int, metadata: Metadata, mountpoint: Path | None
) -> EventFileCache | None:
    status = EventSource.stat(path, mountpoint=mountpoint)
    fingerprint = FileIdentity.fingerprint(status)
    ranges = Segment.plan(path, status, workers, mountpoint=mountpoint)
    if len(ranges) < 2 or ranges[0][0] != 0 or ranges[-1][1] != status.st_size:
        return None
    try:
        guess = _Seed.guess(path, ranges[0][1], metadata, fingerprint, mountpoint)
    except TFRecordError:
        return None

    cache = EventFileCache(path)
    state = _ParseState()
    progress = cache.progress
    progress.identity = FileIdentity.of(status)
    progress.checked_status = fingerprint
    progress.initial_metadata = metadata
    progress.effective_metadata = dict(metadata)
    progress.last_checked_size = status.st_size
    jobs: list[Job] = [
        (
            path,
            start,
            end,
            fingerprint,
            *(guess if start else (None, None, metadata)),
            mountpoint,
        )
        for start, end in ranges
    ]
    _Reduction.run(cache, state, jobs)
    _check_snapshot(path, EventSource.stat(path, mountpoint=mountpoint), fingerprint)
    logger.debug("parallel cold index of {} used {} ranges", path, len(ranges))
    return cache


def parallel_index(
    path: Path,
    workers: int,
    *,
    initial_metadata: Metadata | None = None,
    mountpoint: Path | None = None,
) -> EventFileCache | None:
    """Index a stable, complete snapshot or return None to select the serial path."""
    workers = min(workers, _available_cpus())
    if workers < 2 or current_process().daemon:
        return None
    try:
        return _index_snapshot(path, workers, dict(initial_metadata or {}), mountpoint)
    except (BrokenProcessPool, _PoolUnavailableError) as error:
        logger.warning(
            "parallel cold index of {} failed; using serial: {}", path, error
        )
        return None
    except _SegmentError:
        return None
    except OSError as error:
        if error.errno not in _SOURCE_RACE_ERRNOS:
            raise
        return None
