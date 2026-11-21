"""Plan and index bounded TFRecord ranges from a checked source snapshot."""

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from errno import EINVAL, EISDIR, ELOOP, ENOENT, ENOTDIR, ENXIO
from pathlib import Path
from stat import S_ISREG
from typing import Any, BinaryIO

from tboardfs.event_source import SCAN_BUFFER, EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.indexer import (
    _MALFORMED_RECORD_ERRORS,
    _ClockDependencies,
    _EventIndexer,
    _ParseState,
)
from tboardfs.model import EventFileCache, Record, TFRecordError
from tboardfs.skip_failure import _ReplayRecordError, _speculative_record
from tboardfs.tables import _ScalarRows
from tboardfs.tfrecord import (
    _FOOTER_SIZE,
    _HEADER_SIZE,
    DEFAULT_MAX_RECORD_SIZE,
    _iter_tfrecords,
    _read_header,
)

type Metadata = dict[str, dict[str, Any]]
type Fingerprint = tuple[int | None, ...]
type Job = tuple[
    str | Path,
    int,
    int,
    Fingerprint,
    float | None,
    object,
    Metadata | None,
    Path | None,
]

_SOURCE_RACE_ERRNOS = frozenset({EINVAL, EISDIR, ELOOP, ENOENT, ENOTDIR, ENXIO})


class _SegmentError(Exception):
    """Reject a parallel snapshot so the caller can use the serial parser."""


@dataclass(slots=True)
class Segment:
    """One indexed range, its clock seeds, and its inherited dependencies."""

    start: int
    end: int
    cache: EventFileCache
    clock: _ClockDependencies
    metadata_reads: dict[str, dict[str, Any] | None]
    metadata_writes: Metadata
    scalar_rows: dict[str, _ScalarRows]
    histogram_rows: dict[str, dict[str, list[Any]]]
    deferred_tensor_content: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )

    @staticmethod
    def plan(
        path: str | Path,
        status: os.stat_result,
        n: int,
        *,
        mountpoint: Path | None = None,
    ) -> Generator[tuple[int, int], None, None]:
        """Yield each occupied bucket's first frame start and exclusive start limit."""
        if n < 1:
            raise ValueError("range count must be positive")
        path = Path(path)
        size = status.st_size
        expected = FileIdentity.fingerprint(status)
        _check_snapshot(path, status, expected)
        try:
            with EventSource.open(
                path, mountpoint=mountpoint, buffering=4 << 10
            ) as handle:
                fd = handle.fileno()
                _check_snapshot(path, os.fstat(fd), expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
                yield from Segment._ranges(handle, size, n)
                _check_snapshot(path, os.fstat(fd), expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
        except TFRecordError as error:
            raise _SegmentError(f"{path}: could not plan snapshot: {error}") from error
        except OSError as error:
            if error.errno not in _SOURCE_RACE_ERRNOS:
                raise
            raise _SegmentError(f"{path}: could not plan snapshot: {error}") from error

    @staticmethod
    def _ranges(
        handle: BinaryIO, size: int, n: int
    ) -> Generator[tuple[int, int], None, None]:
        """Follow frame lengths until every occupied byte range has a known start."""
        if not size:
            return
        previous = _read_header(handle, 0, size, DEFAULT_MAX_RECORD_SIZE, None)
        offset = _HEADER_SIZE + previous[1] + _FOOTER_SIZE
        if offset == size:
            return
        limit = (size + n - 1) // n
        yield 0, limit
        while limit < size:
            while offset < limit:
                handle.seek(offset)
                previous = _read_header(
                    handle, offset, size, DEFAULT_MAX_RECORD_SIZE, previous
                )
                offset += _HEADER_SIZE + previous[1] + _FOOTER_SIZE
            if offset == size:
                return
            bucket = offset * n // size
            limit = ((bucket + 1) * size + n - 1) // n
            yield offset, limit

    @staticmethod
    def work(job: Job, *, defer_tensor_content: bool = False) -> "Segment":
        """Index frames starting in the assigned range using one verified descriptor."""
        (
            path_arg,
            start,
            limit,
            expected,
            first_time,
            epoch,
            initial_metadata,
            mountpoint,
        ) = job
        path = Path(path_arg)
        cache = EventFileCache(source_path=path)
        progress = cache.progress
        progress.initial_metadata = dict(initial_metadata or {})
        metadata = _TrackedMetadata(progress.initial_metadata)
        progress.effective_metadata = metadata
        deferred: list[tuple[int, int, int, int]] = []
        clock = _ClockDependencies(first_time, epoch)
        state = _ParseState(
            first_time=first_time,
            epoch=epoch,
            clock=clock,
            deferred_tensor_content=deferred if defer_tensor_content else None,
            retain_tensor_extents=True,
        )
        position = start
        try:
            with EventSource.open(
                path, mountpoint=mountpoint, buffering=SCAN_BUFFER
            ) as handle:
                fd = handle.fileno()
                status = os.fstat(fd)
                _check_snapshot(path, status, expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
                if not 0 <= start <= limit <= status.st_size:
                    raise _SegmentError(f"{path}: invalid range [{start}, {limit})")
                progress.identity = FileIdentity.of(status)
                progress.checked_status = expected
                progress.last_checked_size = status.st_size
                handle.seek(start)
                for record in _iter_tfrecords(handle, end=status.st_size, stop=limit):
                    _index_record(cache, record, state)
                    position = record.record_offset + record.total_size
                    progress.frames.append(position, record.payload_crc)
                if position < limit:
                    raise _SegmentError(f"{path}: range did not reach offset {limit}")
                _check_snapshot(path, os.fstat(fd), expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
        except TFRecordError as error:
            raise _SegmentError(
                f"{path}: could not index range [{start}, {limit}): {error}"
            ) from error
        except OSError as error:
            if error.errno not in _SOURCE_RACE_ERRNOS:
                raise
            raise _SegmentError(
                f"{path}: could not index range [{start}, {limit}): {error}"
            ) from error
        progress.processed_pos = position
        progress.first_time = state.first_time
        progress.epoch = state.epoch
        progress.effective_metadata = dict(metadata)
        cache.scalar_mtimes = {
            tag: rows.wall_time[-1] for tag, rows in state.scalar_rows.items()
        }
        return Segment(
            start,
            position,
            cache,
            clock,
            metadata.reads,
            metadata.writes,
            state.scalar_rows,
            state.histogram_rows,
            deferred,
        )


class _TrackedMetadata(dict[str, dict[str, Any]]):
    """Record the inherited metadata looked up during summary classification."""

    reads: dict[str, dict[str, Any] | None]
    writes: Metadata

    def __init__(self, seed: Metadata) -> None:
        super().__init__(seed)
        self.reads = {}
        self.writes = {}

    def __contains__(self, tag: object) -> bool:
        if isinstance(tag, str) and tag not in self.writes and tag not in self.reads:
            self.reads[tag] = self.get(tag)
        return super().__contains__(tag)

    def __setitem__(self, tag: str, value: dict[str, Any]) -> None:
        self.writes[tag] = value
        super().__setitem__(tag, value)


def _check_snapshot(path: Path, status: os.stat_result, expected: Fingerprint) -> None:
    """Reject a nonregular source or a changed snapshot fingerprint."""
    if not S_ISREG(status.st_mode) or FileIdentity.fingerprint(status) != expected:
        raise _SegmentError(f"{path}: expected an unchanged regular-file snapshot")


def _index_record(cache: EventFileCache, record: Record, state: _ParseState) -> None:
    """Reject a speculative frame when the serial parser must issue diagnostics."""
    token = _speculative_record.set(True)
    try:
        _EventIndexer.index_record(cache, record, state, strict=True)
    except (_ReplayRecordError, *_MALFORMED_RECORD_ERRORS) as error:
        raise _SegmentError(
            f"{cache.source_path}: could not index frame at offset {record.record_offset}: {error}"
        ) from error
    finally:
        _speculative_record.reset(token)
