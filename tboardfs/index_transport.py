"""Pack lazy binary entries and source-backed tensor descriptors for worker IPC."""

import os
import struct
from array import array
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Self

from tboardfs.crc32c import masked_crc32c
from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.index_frames import FrameSection
from tboardfs.index_json import _MIN_EXTENT
from tboardfs.index_segment import (
    _SOURCE_RACE_ERRNOS,
    Job,
    Segment,
    _check_snapshot,
    _SegmentError,
)
from tboardfs.model import EventFileCache
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob, _BlobRef
from tboardfs.model.packed_binaries import PackedBinaries
from tboardfs.model.packed_source import BinaryKey, BinaryProjection
from tboardfs.tensor import _TensorDType
from tboardfs.tensor_source import TensorSource
from tboardfs.tfrecord import _FOOTER_SIZE, _HEADER_SIZE

_FLOAT_STEP = 1
_HAS_SPAN = 2
_HAS_PART = 4
_ROW_FIELDS = "dIQIIIIIIII"
_INT_ROW = struct.Struct("<Bq" + _ROW_FIELDS)
_FLOAT_ROW = struct.Struct("<Bd" + _ROW_FIELDS)
_ROW_SIZE = _INT_ROW.size
_INT_HEADER = struct.Struct("<BqdI")
_FLOAT_HEADER = struct.Struct("<BddI")
_ORDINAL_PART = struct.Struct("<II")
_UINT32 = struct.Struct("<I")
_PROOF_SIZE_OFFSET = _ROW_SIZE - _UINT32.size
_ORDINAL_PART_OFFSET = _PROOF_SIZE_OFFSET - _ORDINAL_PART.size
_BLOB_SIZE_OFFSET = _INT_HEADER.size + struct.calcsize("<QII")


def _restore_tensors(segment: Segment, mountpoint: Path | None) -> None:
    if not segment.deferred_tensor_content:
        return
    cache = segment.cache
    path = cache.source_path
    expected = cache.progress.checked_status
    try:
        with EventSource.open(path, mountpoint=mountpoint) as handle:
            fd = handle.fileno()
            _check_snapshot(path, os.fstat(fd), expected)
            _check_snapshot(
                path, EventSource.stat(path, mountpoint=mountpoint), expected
            )
            _read_tensors(handle, segment, mountpoint)
            _check_snapshot(path, os.fstat(fd), expected)
            _check_snapshot(
                path, EventSource.stat(path, mountpoint=mountpoint), expected
            )
    except OSError as error:
        if error.errno not in _SOURCE_RACE_ERRNOS:
            raise
        raise _SegmentError(
            f"{path}: could not restore indexed tensors: {error}"
        ) from error
    segment.deferred_tensor_content.clear()


def _read_tensors(handle: BinaryIO, segment: Segment, mountpoint: Path | None) -> None:
    cache = segment.cache
    path = cache.source_path
    progress = cache.progress
    for position, offset, size, checksum in segment.deferred_tensor_content:
        try:
            start, end, frame_crc = FrameSection.bounds(
                progress.frames, offset, range_start=segment.start
            )
        except ValueError as error:
            raise _SegmentError(
                f"{path}: tensor is outside its indexed range"
            ) from error
        if not (
            segment.start <= start
            and start + _HEADER_SIZE <= offset < offset + size <= end - _FOOTER_SIZE
            and end <= segment.end
        ):
            raise _SegmentError(f"{path}: tensor is outside its indexed frame payload")
        entry = cache.json_entries[position]
        tensor = entry.payload["tensor"]
        if (
            mountpoint is not None
            and entry.tab == "tensors"
            and size >= _MIN_EXTENT
            and _TensorDType.accepts_extent(tensor, size)
        ):
            extent = (offset, size, checksum)
            if extent != entry.tensor_extent or tensor["tensor_content"] != b"":
                raise _SegmentError(
                    f"{path}: indexed tensor changed at offset {offset}"
                )
            frame = FrameRef(
                progress.identity,
                start,
                end - start,
                frame_crc,
                progress.checked_status,
            )
            entry.tensor_source = TensorSource(path, mountpoint, frame, extent)
            continue
        handle.seek(offset)
        data = handle.read(size)
        if len(data) != size or masked_crc32c(data) != checksum:
            raise _SegmentError(f"{path}: indexed tensor changed at offset {offset}")
        tensor["tensor_content"] = data


@dataclass(slots=True)
class _WorkerRows:
    """Project worker rows and materialize entries against their original snapshot."""

    rows: bytearray
    proofs: bytearray
    labels: tuple[BinaryKey, ...]
    exceptions: dict[int, BinaryEntry]
    source_path: Path
    identity: FileIdentity
    fingerprint: tuple[int | None, ...]
    exception_positions: tuple[int, ...] = field(init=False)
    proof_starts: "array[int]" = field(init=False, default_factory=lambda: array("Q"))
    row_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.exception_positions = tuple(sorted(self.exceptions))
        self.row_count = len(self.rows) // _ROW_SIZE
        if self.proofs:
            proof_offset = 0
            for offset in range(0, len(self.rows), _ROW_SIZE):
                (proof_size,) = _UINT32.unpack_from(
                    self.rows, offset + _PROOF_SIZE_OFFSET
                )
                self.proof_starts.append(proof_offset)
                proof_offset += proof_size

    def __len__(self) -> int:
        return self.row_count + len(self.exception_positions)

    def project(self) -> Iterator[BinaryProjection]:
        """Yield virtual-file metadata in the original row order."""
        logical_start = packed_start = 0
        for position in self.exception_positions:
            packed_end = packed_start + position - logical_start
            yield from self._project_rows(packed_start, packed_end, logical_start)
            entry = self.exceptions[position]
            labels = (entry.tag, entry.kind, entry.extension, entry.suffix)
            yield labels, entry.step, entry.wall_time, position
            logical_start = position + 1
            packed_start = packed_end
        yield from self._project_rows(packed_start, self.row_count, logical_start)

    def _project_rows(
        self, packed_start: int, packed_end: int, logical_start: int
    ) -> Iterator[BinaryProjection]:
        rows = self.rows
        labels = self.labels
        offsets = range(packed_start * _ROW_SIZE, packed_end * _ROW_SIZE, _ROW_SIZE)
        for position, offset in enumerate(offsets, logical_start):
            header = _FLOAT_HEADER if rows[offset] & _FLOAT_STEP else _INT_HEADER
            _, step, wall_time, label_id = header.unpack_from(rows, offset)
            yield labels[label_id], step, wall_time, position

    def entry_at(self, index: int) -> BinaryEntry:
        """Restore one entry by its position before exception separation."""
        if self.exception_positions:
            exception = self.exceptions.get(index)
            if exception is not None:
                return exception
            index -= bisect_left(self.exception_positions, index)
        offset = index * _ROW_SIZE
        header = _FLOAT_HEADER if self.rows[offset] & _FLOAT_STEP else _INT_HEADER
        flags, step, wall_time, label_id = header.unpack_from(self.rows, offset)
        ordinal, part = _ORDINAL_PART.unpack_from(
            self.rows, offset + _ORDINAL_PART_OFFSET
        )
        tag, kind, extension, suffix = self.labels[label_id]
        return BinaryEntry(
            _BlobRef(self, offset),
            self.source_path,
            tag,
            kind,
            step,
            wall_time,
            extension,
            ordinal,
            part if flags & _HAS_PART else None,
            suffix,
        )

    def summaries(self) -> Iterator[GraphSummaryEntry]:
        """Yield graph companions, which the worker always stores as exceptions."""
        for position in self.exception_positions:
            entry = self.exceptions[position]
            if isinstance(entry, GraphSummaryEntry):
                yield entry

    def size_at(self, index: int) -> int:
        """Read the size from a row identified by its byte offset."""
        (size,) = _UINT32.unpack_from(self.rows, index + _BLOB_SIZE_OFFSET)
        return int(size)

    def blob_at(self, index: int) -> IndexedBlob:
        """Materialize one row and copy only its proof bytes."""
        flags, _, _, _, offset, frame_size, checksum, size, start, length, _, _, n = (
            _INT_ROW.unpack_from(self.rows, index)
        )
        frame = FrameRef(self.identity, offset, frame_size, checksum, self.fingerprint)
        span = (start, length) if flags & _HAS_SPAN else None
        proof = b""
        if n:
            begin = self.proof_starts[index // _ROW_SIZE]
            proof = bytes(memoryview(self.proofs)[begin : begin + n])
        return IndexedBlob(frame, size, span, proof)


@dataclass(slots=True)
class _PackedResult:
    """A worker segment with its binary entries stored separately in row order."""

    segment: Segment
    rows: bytearray
    proofs: bytearray
    labels: tuple[BinaryKey, ...]
    exceptions: dict[int, BinaryEntry]

    @classmethod
    def pack(cls, segment: Segment) -> Self:
        """Consume a segment and encode its large fields for worker transport."""
        cache = segment.cache
        rows = bytearray(len(cache.binaries) * _ROW_SIZE)
        proofs = bytearray()
        labels: dict[BinaryKey, int] = {}
        exceptions: dict[int, BinaryEntry] = {}
        offset = 0
        for position, entry in enumerate(cache.binaries):
            if cls._pack_row(entry, cache, rows, offset, labels):
                offset += _ROW_SIZE
                proofs.extend(entry.blob.range_checksums)
            else:
                exceptions[position] = entry

        del rows[offset:]
        cache.binaries = []
        return cls(segment, rows, proofs, tuple(labels), exceptions)

    @staticmethod
    def _pack_row(
        entry: BinaryEntry,
        cache: EventFileCache,
        rows: bytearray,
        offset: int,
        labels: dict[BinaryKey, int],
    ) -> bool:
        if type(entry) is not BinaryEntry:
            return False
        blob = entry.blob
        frame = blob.frame
        progress = cache.progress
        step_type = type(entry.step)
        if (
            entry.source_path is not cache.source_path
            or frame.identity is not progress.identity
            or frame.fingerprint is not progress.checked_status
            or step_type not in (int, float)
        ):
            return False

        flags = 0
        row = _INT_ROW
        if step_type is float:
            flags |= _FLOAT_STEP
            row = _FLOAT_ROW
        span_start = span_size = part = 0
        if blob.span is not None:
            flags |= _HAS_SPAN
            span_start, span_size = blob.span
        if entry.part is not None:
            flags |= _HAS_PART
            part = entry.part
        label = (entry.tag, entry.kind, entry.extension, entry.suffix)
        label_id = labels.setdefault(label, len(labels))
        try:
            row.pack_into(
                rows,
                offset,
                flags,
                entry.step,
                entry.wall_time,
                label_id,
                frame.offset,
                frame.size,
                frame.checksum,
                blob.size,
                span_start,
                span_size,
                entry.value_ordinal,
                part,
                len(blob.range_checksums),
            )
        except struct.error:
            return False
        return True

    def unpack(self, *, mountpoint: Path | None = None) -> Segment:
        """Restore the segment from its locally generated worker representation."""
        segment = self.segment
        _restore_tensors(segment, mountpoint)
        cache = segment.cache
        progress = cache.progress
        cache.binaries = PackedBinaries(
            _WorkerRows(
                self.rows,
                self.proofs,
                self.labels,
                self.exceptions,
                cache.source_path,
                progress.identity,
                progress.checked_status,
            )
        )
        return segment


def packed_work(job: Job) -> _PackedResult:
    """Index and pack one segment before IPC."""
    return _PackedResult.pack(Segment.work(job, defer_tensor_content=True))
