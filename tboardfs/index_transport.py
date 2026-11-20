"""Pack lazy binary entries and source-backed tensor descriptors for worker IPC."""

import os
import struct
from dataclasses import dataclass
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
from tboardfs.model.indexed_blob import IndexedBlob, _BlobRef, _Evidence, _FrameItem
from tboardfs.tensor import _TensorDType
from tboardfs.tensor_source import TensorSource
from tboardfs.tfrecord import _FOOTER_SIZE, _HEADER_SIZE

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
class _PackedEvidence:
    """Own the checked row buffers without retaining the segment or its entries."""

    rows: bytearray
    proofs: bytearray
    identity: FileIdentity
    fingerprint: tuple[int | None, ...]
    proof_starts: dict[int, int]

    def size_at(self, index: int) -> int:
        """Read the size from a row identified by its byte offset."""
        return int(_INT_ROW.unpack_from(self.rows, index)[7])

    def blob_at(self, index: int) -> IndexedBlob:
        """Materialize one row and copy only its proof bytes."""
        flags, _, _, _, offset, frame_size, checksum, size, start, length, _, _, n = (
            _INT_ROW.unpack_from(self.rows, index)
        )
        frame = FrameRef(self.identity, offset, frame_size, checksum, self.fingerprint)
        span = (start, length) if flags & _HAS_SPAN else None
        proof = b""
        if n:
            begin = self.proof_starts[index]
            proof = bytes(memoryview(self.proofs)[begin : begin + n])
        return IndexedBlob(frame, size, span, proof)


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
        progress = cache.progress
        identity = progress.identity
        fingerprint = progress.checked_status
        source_path = cache.source_path
        rows = self.rows
        labels = self.labels
        entries: list[BinaryEntry] = []
        append = entries.append
        evidence: _PackedEvidence | None = None
        offset = proof_offset = 0
        count = len(rows) // _ROW_SIZE + len(self.exceptions)
        for position in range(count):
            exception = self.exceptions.get(position)
            if exception is not None:
                append(exception)
                continue

            row = _FLOAT_ROW if rows[offset] & _FLOAT_STEP else _INT_ROW
            (
                flags,
                step,
                wall_time,
                label_id,
                frame_offset,
                frame_size,
                checksum,
                size,
                _,
                _,
                ordinal,
                part,
                proof_size,
            ) = row.unpack_from(rows, offset)
            blob: _Evidence
            if not (flags & _HAS_SPAN or proof_size):
                blob = _FrameItem(
                    identity, fingerprint, frame_offset, frame_size, checksum, size
                )
            else:
                if evidence is None:
                    evidence = _PackedEvidence(
                        rows, self.proofs, identity, fingerprint, {}
                    )
                if proof_size:
                    evidence.proof_starts[offset] = proof_offset
                    proof_offset += proof_size
                blob = _BlobRef(evidence, offset)
            tag, kind, extension, suffix = labels[label_id]
            append(
                BinaryEntry(
                    blob,
                    source_path,
                    tag,
                    kind,
                    step,
                    wall_time,
                    extension,
                    ordinal,
                    part if flags & _HAS_PART else None,
                    suffix,
                )
            )
            offset += _ROW_SIZE
        cache.binaries = entries
        return segment


def packed_work(job: Job) -> _PackedResult:
    """Index and pack one segment before IPC."""
    return _PackedResult.pack(Segment.work(job, defer_tensor_content=True))
