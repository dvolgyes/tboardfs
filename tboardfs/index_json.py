"""Persistent plugin rows and verified lazy source tensor extents."""

from typing import Any, BinaryIO

from tboardfs.crc32c import masked_crc32c
from tboardfs.index_frames import FrameSection
from tboardfs.index_wire import _IndexReader, _IndexWriter
from tboardfs.model.event_file_cache import EventFileCache
from tboardfs.model.json_entry import JsonEntry

_MIN_EXTENT = 64 << 10


class PluginSection:
    """Encode plugin rows and hydrate deferred source spans."""

    @staticmethod
    def _stored_payload(entry: JsonEntry) -> dict[str, Any]:
        """Replace a backed tensor buffer with a wire placeholder."""
        payload = entry.payload
        extent = entry.tensor_extent
        source = entry.tensor_source
        if source is not None:
            if extent is None or source.extent != PluginSection._extent(
                extent, payload
            ):
                raise ValueError("indexed tensor source disagrees with extent")
            return payload
        if extent is None:
            return payload
        tensor = payload.get("tensor")
        if not isinstance(tensor, dict):
            raise ValueError("indexed tensor extent has no tensor")
        content = tensor.get("tensor_content")
        if not isinstance(content, bytes) or len(content) != extent[1]:
            raise ValueError("indexed tensor content disagrees with extent")
        return {**payload, "tensor": {**tensor, "tensor_content": b""}}

    @staticmethod
    def write(writer: _IndexWriter, entries: list[JsonEntry]) -> None:
        """Write plugin rows without mutating the live index or copying tensors."""
        writer.count(len(entries))
        for entry in entries:
            writer.value(
                (
                    entry.tag,
                    entry.tab,
                    entry.step,
                    entry.wall_time,
                    PluginSection._stored_payload(entry),
                    entry.tensor_extent,
                )
            )

    @staticmethod
    def _extent(value: object, payload: dict[str, Any]) -> tuple[int, int, int] | None:
        """Validate an optional large tensor descriptor."""
        if value is None:
            return None
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("invalid index tensor extent")
        offset, size, crc = value
        if (
            any(type(field) is not int for field in value)
            or size < _MIN_EXTENT
            or not 0 <= crc <= 0xFFFFFFFF
        ):
            raise ValueError("invalid index tensor extent fields")
        tensor = payload.get("tensor")
        if not isinstance(tensor, dict) or tensor.get("tensor_content") != b"":
            raise ValueError("missing index tensor extent placeholder")
        return offset, size, crc

    @staticmethod
    def _json_entry(reader: _IndexReader) -> JsonEntry:
        """Restore a structurally valid plugin row."""
        row = reader.value()
        if not isinstance(row, tuple) or len(row) != 6:
            raise ValueError("invalid index plugin row")
        tag, tab, step, time, payload, extent = row
        if not isinstance(tag, str) or not isinstance(tab, str):
            raise ValueError("invalid index plugin labels")
        if type(step) not in (int, float) or type(time) is not float:
            raise ValueError("invalid index plugin times")
        if not isinstance(payload, dict) or any(
            not isinstance(key, str) for key in payload
        ):
            raise ValueError("invalid index plugin payload")
        PluginSection._validate_payload(payload)
        return JsonEntry(
            tag, tab, step, time, payload, PluginSection._extent(extent, payload)
        )

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        """Check types of fields used by plugin readers."""
        for name, kind in (
            ("plugin_name", str),
            ("plugin_content", bytes),
            ("tensor", dict),
        ):
            if name in payload and not isinstance(payload[name], kind):
                raise ValueError("invalid index plugin content")
        PluginSection._validate_tensor(payload.get("tensor", {}))

    @staticmethod
    def _validate_tensor(tensor: dict[str, Any]) -> None:
        """Preserve parser field types required by lazy tensor readers."""
        fields = (
            ("dtype", int),
            ("tensor_content", bytes),
            ("shape", list),
            ("string_val", list),
            ("unknown_rank", bool),
            ("values", dict),
        )
        for name, kind in fields:
            if name in tensor and type(tensor[name]) is not kind:
                raise ValueError("invalid index tensor content")
        if any(type(dim) is not int for dim in tensor.get("shape", ())):
            raise ValueError("invalid index tensor shape")
        if any(type(value) is not bytes for value in tensor.get("string_val", ())):
            raise ValueError("invalid index tensor strings")
        for key, values in tensor.get("values", {}).items():
            PluginSection._validate_values(key, values)

    @staticmethod
    def _validate_values(key: object, values: object) -> None:
        """Check the repeated numeric fields without materializing an array."""
        if not isinstance(key, int) or not isinstance(values, list):
            raise ValueError("invalid index tensor repeated field")
        if any(type(value) not in (int, float, bool) for value in values):
            raise ValueError("invalid index tensor repeated value")

    @staticmethod
    def read(reader: _IndexReader, source_size: int) -> list[JsonEntry]:
        """Parse plugin rows independently of any source reads."""
        count = reader.items()
        if count > source_size:
            raise ValueError("too many index plugin rows for source size")
        return [PluginSection._json_entry(reader) for _ in range(count)]

    @staticmethod
    def _tensor_reads(cache: EventFileCache) -> list[tuple[int, int, int, JsonEntry]]:
        """Plan non-overlapping reads within source payloads."""
        reads = [
            (*entry.tensor_extent, entry)
            for entry in cache.json_entries
            if entry.tensor_extent is not None
        ]
        reads.sort(key=PluginSection._extent_offset)
        previous = 0
        for offset, size, _, _entry in reads:
            start, end, _ = FrameSection.bounds(cache.progress.frames, offset)
            if offset < max(previous, start + 12) or offset + size > end - 4:
                raise ValueError("index tensor extents overlap or exceed frame payload")
            previous = offset + size
        return reads

    @staticmethod
    def _extent_offset(read: tuple[int, int, int, JsonEntry]) -> int:
        """Return the source offset used for ordering reads."""
        return read[0]

    @staticmethod
    def restore_tensors(cache: EventFileCache, source: BinaryIO) -> None:
        """Hydrate disjoint spans only, bounding their total length by source size."""
        for offset, size, checksum, entry in PluginSection._tensor_reads(cache):
            source.seek(offset)
            content = source.read(size)
            if len(content) != size or masked_crc32c(content) != checksum:
                raise ValueError(f"index tensor changed at source offset {offset}")
            entry.payload["tensor"]["tensor_content"] = content
