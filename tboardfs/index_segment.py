"""Plan and index bounded TFRecord ranges from a checked source snapshot."""

from dataclasses import dataclass, field
from errno import EINVAL, EISDIR, ELOOP, ENOENT, ENOTDIR, ENXIO
from itertools import pairwise
import os
from pathlib import Path
from stat import S_ISREG
from typing import Any

from tboardfs.event_source import EventSource
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
    DEFAULT_MAX_RECORD_SIZE,
    _FOOTER_SIZE,
    _HEADER_SIZE,
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
    ) -> list[tuple[int, int]]:
        """Check frame headers and cut at the boundaries nearest byte targets."""
        if n < 1:
            raise ValueError("range count must be positive")
        path = Path(path)
        size = status.st_size
        expected = FileIdentity.fingerprint(status)
        _check_snapshot(path, status, expected)
        boundaries = [0]
        offset = 0
        target_index = 1
        try:
            with EventSource.open(path, mountpoint=mountpoint) as handle:
                fd = handle.fileno()
                _check_snapshot(path, os.fstat(fd), expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
                while offset < size:
                    _, length = _read_header(
                        handle, offset, size, DEFAULT_MAX_RECORD_SIZE
                    )
                    following = offset + _HEADER_SIZE + length + _FOOTER_SIZE
                    if target_index < n and target_index * size <= following * n:
                        last_target = min(n - 1, following * n // size)
                        midpoint = (offset + following) * n
                        if (
                            2 * target_index * size <= midpoint
                            and boundaries[-1] < offset
                        ):
                            boundaries.append(offset)
                        if (
                            2 * last_target * size > midpoint
                            and boundaries[-1] < following < size
                        ):
                            boundaries.append(following)
                        target_index = last_target + 1
                    offset = handle.seek(following)
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
        if size:
            boundaries.append(size)
        return list(pairwise(boundaries))

    @staticmethod
    def work(job: Job, *, defer_tensor_content: bool = False) -> "Segment":
        """Strictly index the assigned range using one verified descriptor."""
        (
            path_arg,
            start,
            end,
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
        )
        position = start
        try:
            with EventSource.open(path, mountpoint=mountpoint) as handle:
                fd = handle.fileno()
                status = os.fstat(fd)
                _check_snapshot(path, status, expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
                if not 0 <= start <= end <= status.st_size:
                    raise _SegmentError(f"{path}: invalid range [{start}, {end})")
                progress.identity = FileIdentity.of(status)
                progress.checked_status = expected
                progress.last_checked_size = status.st_size
                handle.seek(start)
                for record in _iter_tfrecords(handle, end):
                    _index_record(cache, record, state)
                    position = record.record_offset + record.total_size
                    progress.frames.append(position, record.payload_crc)
                if position != end:
                    raise _SegmentError(f"{path}: range did not reach offset {end}")
                _check_snapshot(path, os.fstat(fd), expected)
                _check_snapshot(
                    path, EventSource.stat(path, mountpoint=mountpoint), expected
                )
        except TFRecordError as error:
            raise _SegmentError(
                f"{path}: could not index range [{start}, {end}): {error}"
            ) from error
        except OSError as error:
            if error.errno not in _SOURCE_RACE_ERRNOS:
                raise
            raise _SegmentError(
                f"{path}: could not index range [{start}, {end}): {error}"
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
            end,
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
