"""Frame and graph evidence in a whole-file persistent index."""

import sys
from array import array
from bisect import bisect_left

from tboardfs.frame_index import FrameIndex
from tboardfs.frame_ref import FrameRef
from tboardfs.index_wire import _IndexReader, _IndexWriter
from tboardfs.model.event_file_cache import EventFileCache
from tboardfs.model.graph_entry import GraphEntry
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.model.meta_graph_entry import MetaGraphEntry
from tboardfs.range_proof import BLOCK_SIZE
from tboardfs.tfrecord import DEFAULT_MAX_RECORD_SIZE


class FrameSection:
    """Encode frame boundaries and graph evidence."""

    @staticmethod
    def write(writer: _IndexWriter, frames: FrameIndex, size: int) -> None:
        """Write the two compact columns in little-endian order."""
        writer.ensure(8 + len(frames.ends) * 12)
        FrameSection._validate_frames(frames, size)
        writer.count(len(frames.ends))
        for column in (frames.ends, frames.checksums):
            wire_column = column
            if sys.byteorder != "little":
                wire_column = column[:]
                wire_column.byteswap()
            writer.write(memoryview(wire_column).cast("B"))

    @staticmethod
    def read(reader: _IndexReader, size: int) -> FrameIndex:
        """Reject out-of-source and truncated frame tables before allocation."""
        count = reader.items(width=12)
        if count > size // 16:
            raise ValueError("too many index frames for source size")
        ends: array[int] = array("Q")
        checksums: array[int] = array("I")
        ends.frombytes(reader.take(count * 8))
        checksums.frombytes(reader.take(count * 4))
        if sys.byteorder != "little":
            ends.byteswap()
            checksums.byteswap()
        frames = FrameIndex(ends, checksums)
        FrameSection._validate_frames(frames, size)
        return frames

    @staticmethod
    def _validate_frames(frames: FrameIndex, size: int) -> None:
        """Require a contiguous whole-file sequence of complete frames."""
        if len(frames.ends) != len(frames.checksums):
            raise ValueError("index frame columns differ in length")
        if (frames.ends[-1] if frames.ends else 0) != size:
            raise ValueError("index frames do not cover the source")
        previous = 0
        for end in frames.ends:
            if end - previous < 16:
                raise ValueError("invalid index frame boundary")
            if end - previous - 16 > DEFAULT_MAX_RECORD_SIZE:
                raise ValueError("index frame payload exceeds the maximum record size")
            previous = end

    @staticmethod
    def bounds(
        frames: FrameIndex, position: int, *, range_start: int = 0
    ) -> tuple[int, int, int]:
        """Find the frame covering a position without expanding the table."""
        index = bisect_left(frames.ends, position + 1)
        if index >= len(frames.ends) or position < range_start:
            raise ValueError("index offset is outside source frames")
        start = frames.ends[index - 1] if index else range_start
        return start, frames.ends[index], frames.checksums[index]

    @staticmethod
    def write_graphs(writer: _IndexWriter, graphs: list[GraphEntry]) -> None:
        """Write graph subclasses and source evidence, not old source identities."""
        writer.count(len(graphs))
        for graph in graphs:
            if type(graph) not in (GraphEntry, MetaGraphEntry):
                raise ValueError("unsupported index graph type")
            readable = graph.readable if isinstance(graph, MetaGraphEntry) else None
            blob = graph.blob
            writer.value(
                (graph.wall_time, readable, blob.frame.offset, blob.size, blob.span)
            )
            writer.block(blob.range_checksums)

    @staticmethod
    def _graph_span(
        span: object, payload_size: int, size: int
    ) -> tuple[int, int] | None:
        """Check the bounds of graph bytes inside the payload."""
        if span is None:
            return None
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError("invalid index graph span")
        start, length = span
        if type(start) is not int or type(length) is not int or length != size:
            raise ValueError("invalid index graph span length")
        if not 0 <= start <= payload_size - size:
            raise ValueError("index graph is outside frame payload")
        return start, length

    @staticmethod
    def _graph_blob(
        cache: EventFileCache, offset: int, size: int, span: object, proof: memoryview
    ) -> IndexedBlob:
        """Restore graph evidence from a validated frame boundary."""
        progress = cache.progress
        start, end, crc = FrameSection.bounds(progress.frames, offset)
        payload_size = end - start - 16
        if start != offset or not 0 <= size <= payload_size:
            raise ValueError("invalid index graph frame evidence")
        validated_span = FrameSection._graph_span(span, payload_size, size)
        expected = (
            0 if validated_span is None else 4 * ((size + BLOCK_SIZE - 1) // BLOCK_SIZE)
        )
        if len(proof) != expected:
            raise ValueError("invalid index graph proof length")
        frame = FrameRef(
            progress.identity, start, end - start, crc, progress.checked_status
        )
        return IndexedBlob(frame, size, validated_span, proof.tobytes())

    @staticmethod
    def _read_graph(reader: _IndexReader, cache: EventFileCache) -> GraphEntry:
        """Read a graph or meta-graph row."""
        row = reader.value()
        if not isinstance(row, tuple) or len(row) != 5:
            raise ValueError("invalid index graph row")
        time, readable, offset, size, span = row
        if type(time) is not float or type(offset) is not int or type(size) is not int:
            raise ValueError("invalid index graph fields")
        blob = FrameSection._graph_blob(cache, offset, size, span, reader.block())
        if readable is None:
            return GraphEntry(blob, cache.source_path, time)
        if type(readable) is not bool:
            raise ValueError("invalid index meta-graph flag")
        return MetaGraphEntry(blob, cache.source_path, time, readable)

    @staticmethod
    def read_graphs(reader: _IndexReader, cache: EventFileCache) -> list[GraphEntry]:
        """Rebind graph evidence to the verified source and current path."""
        count = reader.items(width=10)
        if count > 2 * len(cache.progress.frames.ends):
            raise ValueError("too many index graphs for source frames")
        return [FrameSection._read_graph(reader, cache) for _ in range(count)]
