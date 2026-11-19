"""Pack lazy binary entries and source-backed tensor descriptors for worker IPC."""

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import BinaryIO, Self

from tboardfs.crc32c import masked_crc32c
from tboardfs.event_source import EventSource
from tboardfs.frame_ref import FrameRef
from tboardfs.index_segment import (
    _SOURCE_RACE_ERRNOS,
    Job,
    Segment,
    _SegmentError,
    _check_snapshot,
)
from tboardfs.model import EventFileCache
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.indexed_blob import IndexedBlob


type _Labels = tuple[str, str, str, str]

_FLOAT_STEP = 1
_HAS_SPAN = 2
_HAS_PART = 4
_ROW_FIELDS = "dIQIIIIIIII"
_INT_ROW = struct.Struct("<Bq" + _ROW_FIELDS)
_FLOAT_ROW = struct.Struct("<Bd" + _ROW_FIELDS)
_ROW_SIZE = _INT_ROW.size


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
            _read_tensors(handle, segment)
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


def _read_tensors(handle: BinaryIO, segment: Segment) -> None:
    cache = segment.cache
    path = cache.source_path
    for position, offset, size, checksum in segment.deferred_tensor_content:
        if not segment.start <= offset < offset + size <= segment.end:
            raise _SegmentError(f"{path}: tensor is outside its indexed range")
        handle.seek(offset)
        data = handle.read(size)
        if len(data) != size or masked_crc32c(data) != checksum:
            raise _SegmentError(f"{path}: indexed tensor changed at offset {offset}")
        cache.json_entries[position].payload["tensor"]["tensor_content"] = data


@dataclass(slots=True)
class _PackedResult:
    """A worker segment with its binary entries stored separately in row order."""

    segment: Segment
    rows: bytearray
    proofs: bytearray
    labels: tuple[_Labels, ...]
    exceptions: dict[int, BinaryEntry]

    @classmethod
    def pack(cls, segment: Segment) -> Self:
        """Consume a segment and encode its large fields for worker transport."""
        cache = segment.cache
        rows = bytearray(len(cache.binaries) * _ROW_SIZE)
        proofs = bytearray()
        labels: dict[_Labels, int] = {}
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
        labels: dict[_Labels, int],
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
        entries: list[BinaryEntry] = []
        offset = proof_offset = 0
        proof_data = memoryview(self.proofs)
        count = len(self.rows) // _ROW_SIZE + len(self.exceptions)
        for position in range(count):
            exception = self.exceptions.get(position)
            if exception is not None:
                entries.append(exception)
                continue

            entry, proof_offset = self._unpack_row(offset, proof_data, proof_offset)
            offset += _ROW_SIZE
            entries.append(entry)
        cache.binaries = entries
        return segment

    def _unpack_row(
        self, offset: int, proofs: memoryview, proof_offset: int
    ) -> tuple[BinaryEntry, int]:
        row = _FLOAT_ROW if self.rows[offset] & _FLOAT_STEP else _INT_ROW
        (
            flags,
            step,
            wall_time,
            label_id,
            frame_offset,
            frame_size,
            checksum,
            size,
            span_start,
            span_size,
            ordinal,
            part,
            proof_size,
        ) = row.unpack_from(self.rows, offset)
        proof_end = proof_offset + proof_size
        proof = bytes(proofs[proof_offset:proof_end]) if proof_size else b""
        tag, kind, extension, suffix = self.labels[label_id]
        cache = self.segment.cache
        progress = cache.progress
        frame = FrameRef(
            progress.identity,
            frame_offset,
            frame_size,
            checksum,
            progress.checked_status,
        )
        span = (span_start, span_size) if flags & _HAS_SPAN else None
        blob = IndexedBlob(frame, size, span, proof)
        entry = BinaryEntry(
            blob,
            cache.source_path,
            tag,
            kind,
            step,
            wall_time,
            extension,
            ordinal,
            part if flags & _HAS_PART else None,
            suffix,
        )
        return entry, proof_end


def packed_work(job: Job) -> _PackedResult:
    """Index and pack one segment before IPC."""
    return _PackedResult.pack(Segment.work(job, defer_tensor_content=True))
