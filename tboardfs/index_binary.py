"""Encode data-only binary entries bound to a whole-file frame index."""

import struct
import sys
from bisect import bisect_left
from operator import index

import numpy as np

from tboardfs.frame_ref import FrameRef
from tboardfs.index_binary_layout import (
    _CONVERTS,
    _FLOAT,
    _GRAPH,
    _HEADER,
    _LABEL,
    _PART,
    _PROFILES,
    _REAL,
    _SIGNED,
    _SPAN,
    _UINT,
    _UNSIGNED,
    Label,
)
from tboardfs.index_binary_rows import BinaryRows
from tboardfs.index_utf8 import encode_utf8
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.range_proof import BLOCK_SIZE

_BINARY_KINDS = frozenset({"images", "audio", "videos", "tensors", "graphs"})


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
        labels: dict[Label, int] = {}
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
    def _write_label(data: bytearray, label: Label, remaining: int) -> int:
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
        """Validate every row before publishing entries with deferred read evidence."""
        data = bytes(data)
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
        return BinaryRows(data, row_count, labels, labels_end, cache).entries()

    @staticmethod
    def _step_layout(step: float) -> tuple[int, struct.Struct, int | float]:
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
    def _check_label(label: Label) -> None:
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
    ) -> list[Label]:
        labels: list[Label] = []
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
