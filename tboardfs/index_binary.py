"""Encode data-only binary entries bound to a whole-file frame index."""

import struct
import sys
from bisect import bisect_left
from operator import index

import numpy as np

from tboardfs.frame_ref import FrameRef
from tboardfs.index_utf8 import encode_utf8
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.range_proof import BLOCK_SIZE


type _Label = tuple[str, str, str, str]

# Header, fixed-size rows, length-prefixed labels, then unprefixed proof bytes.
_HEADER = struct.Struct("<III")  # row count, label count, label section size
_LABEL = struct.Struct("<IIII")  # tag, kind, extension, suffix lengths
_FIELDS = "dIIIIIIII"  # time, label, frame ID/size/CRC, blob size/start, ordinal, part
_SIGNED = struct.Struct("<Bq" + _FIELDS)
_UNSIGNED = struct.Struct("<BQ" + _FIELDS)
_FLOAT = struct.Struct("<Bd" + _FIELDS)
_SPAN, _PART, _GRAPH, _CONVERTS, _PROFILES, _UINT, _REAL = (1, 2, 4, 8, 16, 32, 64)
_STEP_MASK = _UINT | _REAL
_BINARY_KINDS = frozenset({"images", "audio", "videos", "tensors", "graphs"})
_TFRECORD_OVERHEAD = 16


def _proof_length(size: int, has_span: bool) -> int:
    return ((size + BLOCK_SIZE - 1) // BLOCK_SIZE) * 4 if has_span else 0


def _check_extent(
    flags: int, payload_size: int, size: int, start: int, ordinal: int, part: int
) -> None:
    if not 0 <= size <= payload_size or not 0 <= ordinal < payload_size:
        raise ValueError("invalid binary blob size or value ordinal")
    if not 0 <= part < payload_size or (not flags & _PART and part):
        raise ValueError("invalid binary part")
    if not 0 <= start <= payload_size - size or (not flags & _SPAN and start):
        raise ValueError("invalid binary span")


class BinarySection:
    """Translate fixed rows and validate their references to the source index."""

    @staticmethod
    def minimum_size(cache: EventFileCache) -> int:
        """Return the section size before labels and proofs."""
        return _HEADER.size + len(cache.binaries) * _SIGNED.size

    @staticmethod
    def encode(cache: EventFileCache, *, limit: int = sys.maxsize) -> bytes:
        """Pack binary entries without serializing source paths or payload bytes."""
        remaining = limit - BinarySection.minimum_size(cache)
        if remaining < 0:
            raise ValueError("binary section exceeds index budget")
        rows = bytearray(len(cache.binaries) * _SIGNED.size)
        labels: dict[_Label, int] = {}
        label_data = bytearray()
        proofs = bytearray()
        try:
            for position, entry in enumerate(cache.binaries):
                label = (entry.tag, entry.kind, entry.extension, entry.suffix)
                next_id = len(labels)
                label_id = labels.setdefault(label, next_id)
                if label_id == next_id:
                    remaining -= _LABEL.size
                    remaining = BinarySection._write_label(label_data, label, remaining)
                proof = entry.blob.range_checksums
                remaining -= len(proof)
                if remaining < 0:
                    raise ValueError("binary section exceeds index budget")
                BinarySection._pack_row(
                    rows, position * _SIGNED.size, cache, entry, label_id
                )
                proofs.extend(proof)
            header = _HEADER.pack(len(cache.binaries), len(labels), len(label_data))
        except struct.error as error:
            raise ValueError("binary index field is out of range") from error
        return b"".join((header, rows, label_data, proofs))

    @staticmethod
    def _write_label(data: bytearray, label: _Label, remaining: int) -> int:
        # Check the whole label's lower bound before encoding its fields.
        if sum(map(len, label)) > remaining:
            raise ValueError("binary section exceeds index budget")
        BinarySection._check_label(label)
        fields: list[bytes] = []
        for field in label:
            encoded = encode_utf8(field, remaining)
            remaining -= len(encoded)
            fields.append(encoded)
        data.extend(_LABEL.pack(*(len(field) for field in fields)))
        data.extend(b"".join(fields))
        return remaining

    @staticmethod
    def decode(data: bytes | memoryview, cache: EventFileCache) -> list[BinaryEntry]:
        """Validate rows against the restored frame index and bind to live status."""
        if len(data) < _HEADER.size:
            raise ValueError("truncated binary index header")
        row_count, label_count, label_size = _HEADER.unpack_from(data)
        rows_end = _HEADER.size + row_count * _SIGNED.size
        labels_end = rows_end + label_size
        if labels_end > len(data) or label_count > min(
            row_count, label_size // _LABEL.size
        ):
            raise ValueError("invalid binary section bounds")
        labels = BinarySection._read_labels(data, label_count, rows_end, labels_end)
        proof_offset = labels_end
        rows: list[BinaryEntry] = []
        for offset in range(_HEADER.size, rows_end, _SIGNED.size):
            entry, proof_offset = BinarySection._decode_row(
                data, offset, proof_offset, labels, cache
            )
            rows.append(entry)
        if proof_offset != len(data):
            raise ValueError("trailing binary range proof data")
        return rows

    @staticmethod
    def _step_layout(step: int | float) -> tuple[int, struct.Struct, int | float]:
        if type(step) is int or isinstance(step, np.integer):
            number = index(step)
            if number >= 2**63:
                return _UINT, _UNSIGNED, number
            return 0, _SIGNED, number
        if isinstance(step, (float, np.float16, np.float32, np.float64)):
            return _REAL, _FLOAT, float(step)
        raise ValueError("unsupported binary step type")

    @staticmethod
    def _encode_flags(entry: BinaryEntry) -> int:
        if type(entry) is BinaryEntry:
            flags = 0
        elif type(entry) is GraphSummaryEntry:
            flags = _GRAPH
            flags |= _CONVERTS if entry.converts else 0
            flags |= _PROFILES if entry.profiles else 0
        else:
            raise ValueError("unsupported binary entry type")
        flags |= _SPAN if entry.blob.span is not None else 0
        flags |= _PART if entry.part is not None else 0
        return flags

    @staticmethod
    def _check_label(label: _Label) -> None:
        _, kind, extension, suffix = label
        if kind not in _BINARY_KINDS or not extension:
            raise ValueError("invalid binary label")
        if any(char in field for field in (extension, suffix) for char in "/\x00"):
            raise ValueError("invalid binary filename label")

    @staticmethod
    def _frame_id(cache: EventFileCache, frame: FrameRef) -> int:
        progress = cache.progress
        if (
            frame.identity != progress.identity
            or frame.fingerprint != progress.checked_status
        ):
            raise ValueError("binary entry refers to another source snapshot")
        end = frame.offset + frame.size
        frames = progress.frames
        frame_id = bisect_left(frames.ends, end)
        if (
            frame_id >= len(frames.ends)
            or frame_id >= len(frames.checksums)
            or frames.ends[frame_id] != end
            or (frames.ends[frame_id - 1] if frame_id else 0) != frame.offset
            or frames.checksums[frame_id] != frame.checksum
            or end > progress.processed_pos
        ):
            raise ValueError("binary entry refers to an unindexed frame")
        return frame_id

    @staticmethod
    def _pack_row(
        rows: bytearray,
        offset: int,
        cache: EventFileCache,
        entry: BinaryEntry,
        label: int,
    ) -> None:
        if entry.source_path != cache.source_path:
            raise ValueError("binary entry refers to another source path")
        flags, layout, step = BinarySection._step_layout(entry.step)
        flags |= BinarySection._encode_flags(entry)
        if bool(flags & _GRAPH) != (entry.kind == "graphs"):
            raise ValueError("invalid graph binary kind")
        blob = entry.blob
        frame = blob.frame
        start, size = blob.span if blob.span is not None else (0, blob.size)
        if size != blob.size:
            raise ValueError("binary span and blob sizes disagree")
        if len(blob.range_checksums) != _proof_length(size, bool(flags & _SPAN)):
            raise ValueError("invalid binary range proof length")
        part = entry.part if entry.part is not None else 0
        _check_extent(flags, frame.payload_size, size, start, entry.value_ordinal, part)
        layout.pack_into(
            rows,
            offset,
            flags,
            step,
            entry.wall_time,
            label,
            BinarySection._frame_id(cache, frame),
            frame.size,
            frame.checksum,
            size,
            start,
            entry.value_ordinal,
            part,
        )

    @staticmethod
    def _read_labels(
        data: bytes | memoryview, count: int, offset: int, end: int
    ) -> list[_Label]:
        labels: list[_Label] = []
        for _ in range(count):
            if offset + _LABEL.size > end:
                raise ValueError("truncated binary label header")
            tag, kind, ext, suffix = _LABEL.unpack_from(data, offset)
            start = offset + _LABEL.size
            tag_end = start + tag
            kind_end = tag_end + kind
            ext_end = kind_end + ext
            offset = ext_end + suffix
            if offset > end:
                raise ValueError("truncated binary label")
            label = (
                str(data[start:tag_end], "utf-8"),
                str(data[tag_end:kind_end], "utf-8"),
                str(data[kind_end:ext_end], "utf-8"),
                str(data[ext_end:offset], "utf-8"),
            )
            BinarySection._check_label(label)
            labels.append(label)
        if offset != end:
            raise ValueError("trailing binary label data")
        return labels

    @staticmethod
    def _live_frame(
        cache: EventFileCache, position: int, size: int, checksum: int
    ) -> FrameRef:
        progress = cache.progress
        frames = progress.frames
        if position >= len(frames.ends) or position >= len(frames.checksums):
            raise ValueError("binary row frame index is out of bounds")
        start = frames.ends[position - 1] if position else 0
        end = frames.ends[position]
        if (
            size < _TFRECORD_OVERHEAD
            or end > progress.processed_pos
            or size != end - start
            or checksum != frames.checksums[position]
        ):
            raise ValueError("binary row disagrees with its indexed frame")
        return FrameRef(
            progress.identity, start, size, checksum, progress.checked_status
        )

    @staticmethod
    def _row_layout(flags: int) -> struct.Struct:
        if flags & 128 or flags & _STEP_MASK == _STEP_MASK:
            raise ValueError("invalid binary row flags")
        if flags & (_CONVERTS | _PROFILES) and not flags & _GRAPH:
            raise ValueError("graph flags on a non-graph binary row")
        if flags & _REAL:
            return _FLOAT
        return _UNSIGNED if flags & _UINT else _SIGNED

    @staticmethod
    def _decode_row(
        data: bytes | memoryview,
        offset: int,
        proof_offset: int,
        labels: list[_Label],
        cache: EventFileCache,
    ) -> tuple[BinaryEntry, int]:
        flags = data[offset]
        layout = BinarySection._row_layout(flags)
        (
            _,
            step,
            wall_time,
            label,
            frame_id,
            frame_size,
            checksum,
            size,
            start,
            ordinal,
            part,
        ) = layout.unpack_from(data, offset)
        if label >= len(labels):
            raise ValueError("binary row label index is out of bounds")
        tag, kind, extension, suffix = labels[label]
        if bool(flags & _GRAPH) != (kind == "graphs"):
            raise ValueError("invalid graph binary kind")
        frame = BinarySection._live_frame(cache, frame_id, frame_size, checksum)
        _check_extent(flags, frame.payload_size, size, start, ordinal, part)
        has_span = bool(flags & _SPAN)
        proof_end = proof_offset + _proof_length(size, has_span)
        if proof_end > len(data):
            raise ValueError("truncated binary range proof")
        blob = IndexedBlob(
            frame,
            size,
            (start, size) if has_span else None,
            bytes(data[proof_offset:proof_end]) if proof_end != proof_offset else b"",
        )
        fields = (
            blob,
            cache.source_path,
            tag,
            kind,
            step,
            wall_time,
            extension,
            ordinal,
            part if flags & _PART else None,
            suffix,
        )
        if flags & _GRAPH:
            return (
                GraphSummaryEntry(
                    *fields, bool(flags & _CONVERTS), bool(flags & _PROFILES)
                ),
                proof_end,
            )
        return BinaryEntry(*fields), proof_end
