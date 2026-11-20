"""Validate packed binary rows and materialize their source evidence on demand."""

from io import BytesIO
from pathlib import Path
from struct import Struct

import numpy as np

from tboardfs.file_status import FileIdentity
from tboardfs.frame_index import FrameIndex
from tboardfs.frame_ref import FrameRef
from tboardfs.index_binary_layout import (
    _CONVERTS,
    _FLOAT,
    _GRAPH,
    _HEADER,
    _PART,
    _PROFILES,
    _REAL,
    _SIGNED,
    _SPAN,
    _TFRECORD_OVERHEAD,
    _UINT,
    _UNSIGNED,
    Label,
)
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob, _BlobRef
from tboardfs.range_proof import BLOCK_SIZE

_ROW = np.dtype(
    [
        ("flags", "u1"),
        ("step", "<u8"),
        ("time", "<f8"),
        ("label", "<u4"),
        ("frame", "<u4"),
        ("frame_size", "<u4"),
        ("checksum", "<u4"),
        ("size", "<u4"),
        ("start", "<u4"),
        ("ordinal", "<u4"),
        ("part", "<u4"),
    ]
)

_STEP_MASK = _UINT | _REAL
_UINT64 = Struct("=Q")
_BATCH_SIZE = 64 * 1024


def _validate_batch(
    rows: np.ndarray,
    graph_labels: np.ndarray,
    frame_ends: np.ndarray,
    frame_checksums: np.ndarray,
    processed_pos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate addresses and extents before returning frame starts and proof lengths."""
    flags = rows["flags"]
    graph = (flags & _GRAPH) != 0
    has_span = (flags & _SPAN) != 0
    if (((flags & 128) != 0) | ((flags & _STEP_MASK) == _STEP_MASK)).any():
        raise ValueError("invalid binary row flags")
    if (((flags & (_CONVERTS | _PROFILES)) != 0) & ~graph).any():
        raise ValueError("graph flags on a non-graph binary row")

    label_ids = rows["label"]
    if (label_ids >= len(graph_labels)).any():
        raise ValueError("binary row label index is out of bounds")
    if (graph != graph_labels[label_ids]).any():
        raise ValueError("invalid graph binary kind")

    positions = rows["frame"]
    frame_count = min(len(frame_ends), len(frame_checksums))
    if (positions >= frame_count).any():
        raise ValueError("binary row frame index is out of bounds")
    ends = frame_ends[positions]
    starts = frame_ends[np.maximum(positions, 1) - 1]
    starts[positions == 0] = 0
    frame_sizes = rows["frame_size"]
    if (
        (frame_sizes < _TFRECORD_OVERHEAD)
        | (ends > processed_pos)
        | (ends < starts)
        | (frame_sizes != ends - np.minimum(ends, starts))
        | (rows["checksum"] != frame_checksums[positions])
    ).any():
        raise ValueError("binary row disagrees with its indexed frame")

    payload = frame_sizes.astype(np.int64) - _TFRECORD_OVERHEAD
    sizes = rows["size"]
    parts = rows["part"]
    offsets = rows["start"]
    if ((sizes > payload) | (rows["ordinal"] >= payload)).any():
        raise ValueError("invalid binary blob size or value ordinal")
    if ((parts >= payload) | (((flags & _PART) == 0) & (parts != 0))).any():
        raise ValueError("invalid binary part")
    if ((offsets > payload - sizes) | (~has_span & (offsets != 0))).any():
        raise ValueError("invalid binary span")
    lengths = sizes.astype(np.uint64)
    lengths += BLOCK_SIZE - 1
    lengths //= BLOCK_SIZE
    lengths *= 4
    lengths *= has_span
    return starts, lengths


def _validate(
    rows: np.ndarray,
    labels: tuple[Label, ...],
    frames: FrameIndex,
    processed_pos: int,
    proof_size: int,
) -> tuple[bytes, bytes]:
    """Validate all rows in bounded batches and freeze the two derived columns."""
    if not len(rows):
        if proof_size:
            raise ValueError("trailing binary range proof data")
        return b"", b""

    graph_labels = np.fromiter(
        (label[1] == "graphs" for label in labels), dtype=np.bool_, count=len(labels)
    )
    frame_ends, frame_checksums = np.asarray(frames.ends), np.asarray(frames.checksums)
    starts = BytesIO()
    proof_ends = BytesIO()
    proof_end = 0
    for start in range(0, len(rows), _BATCH_SIZE):
        batch = slice(start, start + _BATCH_SIZE)
        offsets, lengths = _validate_batch(
            rows[batch], graph_labels, frame_ends, frame_checksums, processed_pos
        )
        np.cumsum(lengths, dtype=np.uint64, out=lengths)
        lengths += proof_end
        proof_end = int(lengths[-1])
        if proof_end > proof_size:
            raise ValueError("truncated binary range proof")
        starts.write(memoryview(offsets).cast("B"))
        proof_ends.write(memoryview(lengths).cast("B"))
    if proof_end != proof_size:
        raise ValueError("trailing binary range proof data")
    return starts.getvalue(), proof_ends.getvalue()


class BinaryRows:
    """Own validated wire rows, proof bytes, and the bound source snapshot."""

    _data: bytes
    _count: int
    _proof_start: int
    _frame_starts: bytes
    _proof_ends: bytes
    _labels: tuple[Label, ...]
    _source_path: Path
    _identity: FileIdentity
    _status: tuple[int | None, ...]

    def __init__(
        self,
        data: bytes,
        count: int,
        labels: list[Label],
        proofs: int,
        cache: EventFileCache,
    ) -> None:
        self._data = data
        self._count = count
        self._proof_start = proofs
        self._labels = tuple(labels)
        progress = cache.progress
        self._source_path = cache.source_path
        self._identity = progress.identity
        self._status = progress.checked_status
        self._frame_starts, self._proof_ends = _validate(
            np.frombuffer(data, dtype=_ROW, count=count, offset=_HEADER.size),
            self._labels,
            progress.frames,
            progress.processed_pos,
            len(data) - proofs,
        )

    def entries(self) -> list[BinaryEntry]:
        """Build ordinary entries without retaining the list or resolving blobs."""
        entries: list[BinaryEntry] = []
        append = entries.append
        rows_end = _HEADER.size + self._count * _SIGNED.size
        for index, offset in enumerate(range(_HEADER.size, rows_end, _SIGNED.size)):
            flags = self._data[offset]
            layout = (
                _FLOAT if flags & _REAL else _UNSIGNED if flags & _UINT else _SIGNED
            )
            _, step, wall_time, label, _, _, _, _, _, ordinal, part = (
                layout.unpack_from(self._data, offset)
            )
            tag, kind, extension, suffix = self._labels[label]
            fields = (
                _BlobRef(self, index),
                self._source_path,
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
                append(
                    GraphSummaryEntry(
                        *fields, bool(flags & _CONVERTS), bool(flags & _PROFILES)
                    )
                )
            else:
                append(BinaryEntry(*fields))
        return entries

    def size_at(self, index: int) -> int:
        """Return the validated blob length without resolving its evidence."""
        offset = _HEADER.size + index * _SIGNED.size
        return int(_SIGNED.unpack_from(self._data, offset)[7])

    def blob_at(self, index: int) -> IndexedBlob:
        """Materialize a single blob from validated evidence."""
        row_offset = _HEADER.size + index * _SIGNED.size
        flags, _, _, _, _, frame_size, checksum, size, start, _, _ = (
            _SIGNED.unpack_from(self._data, row_offset)
        )
        offset = index * _UINT64.size
        (frame_start,) = _UINT64.unpack_from(self._frame_starts, offset)
        frame = FrameRef(
            self._identity, frame_start, frame_size, checksum, self._status
        )
        if not flags & _SPAN:
            return IndexedBlob(frame, size)
        (end,) = _UINT64.unpack_from(self._proof_ends, offset)
        begin = (
            _UINT64.unpack_from(self._proof_ends, offset - _UINT64.size)[0]
            if index
            else 0
        )
        proof_start = self._proof_start
        return IndexedBlob(
            frame,
            size,
            (start, size),
            self._data[proof_start + begin : proof_start + end],
        )
