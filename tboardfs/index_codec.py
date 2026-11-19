"""Encode and rebind data-only persistent indexes of complete event files."""

import os
from pathlib import Path
from typing import Any, BinaryIO

from tboardfs.constants import MAX_INDEX_BYTES
from tboardfs.file_status import FileIdentity
from tboardfs.index_binary import BinarySection
from tboardfs.index_frames import FrameSection
from tboardfs.index_json import PluginSection
from tboardfs.index_tables import TableSection
from tboardfs.index_wire import _IndexReader, _IndexWriter, exact_value
from tboardfs.model.event_file_cache import EventFileCache
from tboardfs.model.restart_mark import RestartMark

_MAGIC = b"TBFSIDX\x01"
_BLOCK_PREFIX_BYTES = 8
type _RestartRow = tuple[object, list[object]]


class IndexCodec:
    """Encode and restore the complete index body for one source."""

    @staticmethod
    def encode(cache: EventFileCache, *, limit: int = MAX_INDEX_BYTES) -> bytes:
        """Serialize a complete parse without persisting the old path or identity."""
        progress = cache.progress
        if progress.processed_pos != progress.last_checked_size:
            raise ValueError("cannot persist a partial event file index")
        writer = _IndexWriter(min(limit, MAX_INDEX_BYTES))
        writer.ensure(BinarySection.minimum_size(cache) + _BLOCK_PREFIX_BYTES)
        writer.write(_MAGIC)
        writer.value(progress.initial_metadata)
        writer.count(progress.processed_pos)
        writer.value(progress.first_time)
        writer.value(progress.epoch)
        writer.value(progress.effective_metadata)
        FrameSection.write(writer, progress.frames, progress.processed_pos)
        TableSection.write(writer, cache.scalars, cache.scalar_mtimes, histogram=False)
        TableSection.write(
            writer, cache.histograms, cache.histogram_mtimes, histogram=True
        )
        FrameSection.write_graphs(writer, cache.graphs)
        PluginSection.write(writer, cache.json_entries)
        IndexCodec._write_restarts(writer, cache.restarts)
        writer.block(
            BinarySection.encode(cache, limit=writer.remaining - _BLOCK_PREFIX_BYTES)
        )
        return writer.finish()

    @staticmethod
    def decode(
        body: bytes,
        path: Path,
        status: os.stat_result,
        source: BinaryIO,
        metadata: dict[str, dict[str, Any]],
    ) -> EventFileCache | None:
        """Decode one structurally valid index using the caller's verified source."""
        if len(body) > MAX_INDEX_BYTES:
            raise ValueError("index body exceeds byte budget")
        reader = _IndexReader(body)
        if reader.take(len(_MAGIC)) != _MAGIC:
            raise ValueError("invalid index body version")
        initial_metadata = IndexCodec._metadata(reader)
        if not exact_value(initial_metadata, metadata):
            return None
        cache = EventFileCache(path)
        cache.progress.initial_metadata = initial_metadata
        IndexCodec._read_progress(reader, cache, status)
        size = status.st_size
        cache.scalars, cache.scalar_mtimes = TableSection.read(
            reader, size, histogram=False
        )
        cache.histograms, cache.histogram_mtimes = TableSection.read(
            reader, size, histogram=True
        )
        cache.graphs = FrameSection.read_graphs(reader, cache)
        cache.json_entries = PluginSection.read(reader, size)
        restart_rows = IndexCodec._read_restarts(reader, cache)
        binaries = reader.block()
        if reader.remaining:
            raise ValueError("unexpected data after index body")
        cache.binaries = BinarySection.decode(binaries, cache)
        cache.restarts = IndexCodec._restore_restarts(restart_rows, cache)
        PluginSection.restore_tensors(cache, source)
        return cache

    @staticmethod
    def _metadata(reader: _IndexReader) -> dict[str, dict[str, Any]]:
        """Read a mapping of summary tags to parser fields."""
        metadata = reader.value()
        if not isinstance(metadata, dict) or any(
            not isinstance(tag, str) or not isinstance(fields, dict)
            for tag, fields in metadata.items()
        ):
            raise ValueError("invalid index parser metadata")
        return metadata

    @staticmethod
    def _read_progress(
        reader: _IndexReader, cache: EventFileCache, status: os.stat_result
    ) -> None:
        """Restore a snapshot-bound complete-file parser position."""
        size = reader.count()
        first_time = reader.value()
        epoch = reader.value()
        if size != status.st_size:
            raise ValueError("index length differs from source file size")
        if first_time is not None and type(first_time) is not float:
            raise ValueError("invalid index first time")
        if epoch is not None and type(epoch) not in (int, float, bool):
            raise ValueError("invalid index epoch")
        progress = cache.progress
        progress.identity = FileIdentity.of(status)
        progress.checked_status = FileIdentity.fingerprint(status)
        progress.processed_pos = progress.last_checked_size = size
        progress.first_time = first_time
        progress.epoch = epoch
        progress.effective_metadata = IndexCodec._metadata(reader)
        progress.frames = FrameSection.read(reader, size)

    @staticmethod
    def _write_restarts(writer: _IndexWriter, marks: list[RestartMark]) -> None:
        """Write restart events with their prefix row counts."""
        writer.count(len(marks))
        for mark in marks:
            writer.value((mark.kind, mark.value))
            writer.count(len(mark.counts))
            for (table, tag), count in mark.counts.items():
                writer.value((table, tag, count))

    @staticmethod
    def _restart_counts(
        counts: list[object], lengths: dict[tuple[str, str], int]
    ) -> dict[tuple[str, str], int]:
        """Validate counts against the decoded table and list lengths."""
        result: dict[tuple[str, str], int] = {}
        for row in counts:
            if not isinstance(row, tuple) or len(row) != 3:
                raise ValueError("invalid index restart count row")
            table, tag, count = row
            if (
                not isinstance(table, str)
                or not isinstance(tag, str)
                or type(count) is not int
            ):
                raise ValueError("invalid index restart count fields")
            key = (table, tag)
            if key in result or key not in lengths or not 0 <= count <= lengths[key]:
                raise ValueError("invalid index restart count")
            result[key] = count
        return result

    @staticmethod
    def _restart_mark(
        row: _RestartRow, lengths: dict[tuple[str, str], int]
    ) -> RestartMark:
        """Validate one restart marker and its table boundaries."""
        marker, counts = row
        if not isinstance(marker, tuple) or len(marker) != 2:
            raise ValueError("invalid index restart row")
        kind, value = marker
        if kind not in ("version", "start") or type(value) not in (int, float):
            raise ValueError("invalid index restart event")
        if kind == "version" and counts:
            raise ValueError("invalid index restart counts")
        return RestartMark(kind, value, IndexCodec._restart_counts(counts, lengths))

    @staticmethod
    def _read_restarts(
        reader: _IndexReader, cache: EventFileCache
    ) -> list[_RestartRow]:
        """Read no more restart events than source frames."""
        count = reader.items()
        if count > len(cache.progress.frames.ends):
            raise ValueError("too many index restarts for source frames")
        rows: list[_RestartRow] = []
        for _ in range(count):
            marker = reader.value()
            counts = [reader.value() for _ in range(reader.items(width=36))]
            rows.append((marker, counts))
        return rows

    @staticmethod
    def _restore_restarts(
        rows: list[_RestartRow], cache: EventFileCache
    ) -> list[RestartMark]:
        """Validate markers using complete decoded row totals."""
        if not rows:
            return []
        lengths = {
            (table, tag): len(columns["step"])
            for table, tags in (
                ("scalars", cache.scalars),
                ("histograms", cache.histograms),
            )
            for tag, columns in tags.items()
        }
        lengths["binaries", ""] = len(cache.binaries)
        lengths["json_entries", ""] = len(cache.json_entries)
        return [IndexCodec._restart_mark(row, lengths) for row in rows]
