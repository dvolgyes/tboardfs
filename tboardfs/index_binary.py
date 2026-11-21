"""Encode data-only binary entries bound to a whole-file frame index."""

import struct
import sys
from bisect import bisect_left
from collections.abc import Iterable
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
from tboardfs.index_binary_rows import _BATCH_SIZE, _ROW, BinaryRows, _validate_batch
from tboardfs.index_transport import _FLOAT_STEP, _HAS_PART, _HAS_SPAN, _WorkerRows
from tboardfs.index_utf8 import encode_utf8
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.packed_binaries import PackedBinaries
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
        writer = _BinaryWriter(cache, limit)
        try:
            if isinstance(cache.binaries, PackedBinaries):
                writer.write_packed(cache.binaries)
            else:
                writer.write_entries(cache.binaries)
            header = _HEADER.pack(
                len(cache.binaries), len(writer.labels), len(writer.label_data)
            )
        except struct.error as error:
            raise ValueError("binary index field is out of range") from error
        return b"".join((header, writer.rows, writer.label_data, writer.proofs))

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
    def decode(data: bytes | memoryview, cache: EventFileCache) -> PackedBinaries:
        """Validate every row before publishing the packed source."""
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
        return PackedBinaries(BinaryRows(data, row_count, labels, labels_end, cache))

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


class _BinaryWriter:
    """Write entry chunks into a single ordered durable binary section."""

    cache: EventFileCache
    rows: bytearray
    labels: dict[Label, int]
    label_data: bytearray
    proofs: bytearray
    remaining: int
    position: int

    def __init__(self, cache: EventFileCache, limit: int) -> None:
        self.remaining = limit - BinarySection.minimum_size(cache)
        if self.remaining < 0:
            raise ValueError("binary section exceeds index budget")
        self.cache = cache
        self.rows = bytearray(len(cache.binaries) * _SIGNED.size)
        self.labels = {}
        self.label_data = bytearray()
        self.proofs = bytearray()
        self.position = 0

    def write_entries(self, entries: Iterable[BinaryEntry]) -> None:
        """Encode eager entries and worker exceptions through the object path."""
        cache, rows = self.cache, self.rows
        labels, label_data, proofs = self.labels, self.label_data, self.proofs
        remaining = self.remaining
        position = self.position - 1
        for position, entry in enumerate(entries, self.position):
            label = (entry.tag, entry.kind, entry.extension, entry.suffix)
            next_id = len(labels)
            label_id = labels.setdefault(label, next_id)
            if label_id == next_id:
                remaining = BinarySection._write_label(
                    label_data, label, remaining - _LABEL.size
                )
            proof = entry.blob.range_checksums
            remaining -= len(proof)
            if remaining < 0:
                raise ValueError("binary section exceeds index budget")
            BinarySection._pack_row(
                rows, position * _SIGNED.size, cache, entry, label_id
            )
            proofs.extend(proof)
        self.remaining = remaining
        self.position = position + 1

    def write_packed(self, binaries: PackedBinaries) -> None:
        """Encode each chunk according to its in-memory representation."""
        packed = _PackedWriter(self)
        for chunk in binaries.chunks():
            if isinstance(chunk, _WorkerRows):
                packed.write_worker(chunk)
            elif isinstance(chunk, list):
                self.write_entries(chunk)
            else:
                self.write_entries(chunk.entry_at(i) for i in range(len(chunk)))


class _PackedWriter:
    """Translate worker rows against one durable output and frame table."""

    writer: _BinaryWriter
    dest: np.ndarray
    frame_ends: np.ndarray
    frame_checksums: np.ndarray
    processed_pos: int

    def __init__(self, writer: _BinaryWriter) -> None:
        self.writer = writer
        self.dest = np.frombuffer(writer.rows, dtype=_ROW)
        progress = writer.cache.progress
        self.frame_ends = np.asarray(progress.frames.ends)
        self.frame_checksums = np.asarray(progress.frames.checksums)
        self.processed_pos = progress.processed_pos

    def _check_snapshot(self, chunk: _WorkerRows) -> None:
        if not chunk.row_count:
            return
        cache = self.writer.cache
        if chunk.source_path != cache.source_path:
            raise ValueError("binary entry refers to another source path")
        if (
            chunk.identity != cache.progress.identity
            or chunk.fingerprint != cache.progress.checked_status
        ):
            raise ValueError("binary entry refers to another source snapshot")

    def write_worker(self, chunk: _WorkerRows) -> None:
        """Write one chunk, interleaving its exceptional rows and range proofs."""
        self._check_snapshot(chunk)
        remap = np.full(len(chunk.labels), -1, dtype=np.int64)
        graph_labels = np.fromiter(
            (label[1] == "graphs" for label in chunk.labels),
            dtype=np.bool_,
            count=len(chunk.labels),
        )
        writer = self.writer
        proof_offset = 0
        proofs = memoryview(chunk.proofs)
        for batch in chunk.batches(_BATCH_SIZE):
            if isinstance(batch, BinaryEntry):
                writer.write_entries((batch,))
                continue
            out = self.dest[writer.position : writer.position + len(batch)]
            size = self._convert_batch(batch, out, graph_labels)
            proof_end = proof_offset + size
            if proof_end > len(proofs):
                raise ValueError("truncated binary range proof")
            out["label"] = self._label_ids(batch["label"], chunk.labels, remap)
            writer.remaining -= size
            if writer.remaining < 0:
                raise ValueError("binary section exceeds index budget")
            if size:
                writer.proofs.extend(proofs[proof_offset:proof_end])
            proof_offset = proof_end
            writer.position += len(batch)
        if proof_offset != len(proofs):
            raise ValueError("trailing binary range proof data")

    def _frame_ids(self, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        ends = starts + sizes
        positions = np.searchsorted(self.frame_ends, ends)
        if (ends < starts).any() or (
            positions >= min(len(self.frame_ends), 2**32)
        ).any():
            raise ValueError("binary entry refers to an unindexed frame")
        return positions

    def _convert_batch(
        self, batch: np.ndarray, out: np.ndarray, graph_labels: np.ndarray
    ) -> int:
        """Convert and validate a batch before remapping its worker-local labels."""
        flags = batch["flags"]
        if (flags & (255 ^ (_FLOAT_STEP | _HAS_SPAN | _HAS_PART))).any():
            raise ValueError("invalid worker binary row flags")
        has_span = (flags & _HAS_SPAN) != 0
        out["flags"] = (
            ((flags & _FLOAT_STEP) != 0) * np.uint8(_REAL)
            | has_span * np.uint8(_SPAN)
            | ((flags & _HAS_PART) != 0) * np.uint8(_PART)
        )
        frame_starts = batch["offset"]
        out["frame"] = self._frame_ids(frame_starts, batch["frame_size"])
        for name in (
            "step",
            "time",
            "label",
            "frame_size",
            "checksum",
            "size",
            "start",
            "ordinal",
            "part",
        ):
            out[name] = batch[name]
        starts, lengths = _validate_batch(
            out, graph_labels, self.frame_ends, self.frame_checksums, self.processed_pos
        )
        if (starts != frame_starts).any():
            raise ValueError("binary entry refers to an unindexed frame")
        if (has_span & (batch["span_size"] != batch["size"])).any():
            raise ValueError("binary span and blob sizes disagree")
        if (lengths != batch["proof_size"]).any():
            raise ValueError("invalid binary range proof length")
        return int(lengths.sum())

    def _label_ids(
        self, ids: np.ndarray, labels: tuple[Label, ...], remap: np.ndarray
    ) -> np.ndarray:
        """Intern referenced labels in first-use order."""
        mapped = remap[ids]
        missing = mapped < 0
        if not missing.any():
            return mapped
        writer = self.writer
        fresh, first = np.unique(ids[missing], return_index=True)
        for local_id in fresh[np.argsort(first)]:
            label = labels[local_id]
            next_id = len(writer.labels)
            label_id = writer.labels.setdefault(label, next_id)
            if label_id == next_id:
                writer.remaining = BinarySection._write_label(
                    writer.label_data, label, writer.remaining - _LABEL.size
                )
            remap[local_id] = label_id
        return remap[ids]
