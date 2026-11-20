"""Index binary summary values and memoize exact-byte classifications."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tboardfs.classify import _Classifier
from tboardfs.frame_ref import FrameRef
from tboardfs.model import BinaryEntry, EventFileCache, GraphSummaryEntry, Record
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.payload_spans import PayloadSpans, Span
from tboardfs.range_proof import range_proof
from tboardfs.run_metadata_text import run_metadata_text
from tboardfs.summary_graph import summary_graph_def
from tboardfs.tensor_summary import GRAPH_PLUGINS, _TensorSummary

_LARGE_AUDIO_MEMO_LIMIT = 1 << 20
_IMAGE_FIELD = 4


def _indexed_blob(frame: FrameRef, data: bytes, span: Span | None) -> IndexedBlob:
    """Record the verified span and its range proofs, if present."""
    return IndexedBlob(
        frame, len(data), span, range_proof(data) if span is not None else b""
    )


@dataclass(slots=True)
class _MediaIndex:
    """Insert lazy binary entries with extension caches scoped to one parse."""

    binary_extensions: dict[tuple[str, bytes], str | None] = field(default_factory=dict)
    large_audio_extension: tuple[bytes, str | None] | None = None

    def add_legacy_image(
        self,
        cache: EventFileCache,
        ordinal: int,
        record: Record,
        data: bytes,
        identity: tuple[str, int, float],
        spans: PayloadSpans | None,
    ) -> None:
        """Index an image without metadata using the ordinary media policies."""
        kind = _Classifier.legacy_image_kind(data)
        if kind is None:
            return
        extension = self._cached_extension(kind, data)
        if extension is None:
            return
        span = None
        if spans is not None:
            candidates = self._binary_candidates(spans, ordinal, _IMAGE_FIELD, None, 0)
            if candidates:
                span = spans.match(candidates[0], data)
        frame = record.reference(
            cache.progress.identity, fingerprint=cache.progress.checked_status
        )
        tag, step, wall_time = identity
        cache.binaries.append(
            BinaryEntry(
                source_path=cache.source_path,
                tag=tag,
                kind=kind,
                step=step,
                wall_time=wall_time,
                value_ordinal=ordinal,
                blob=_indexed_blob(frame, data, span),
                extension=extension,
            )
        )

    def add_binary(
        self,
        cache: EventFileCache,
        ordinal: int,
        record: Record,
        value: dict[str, Any],
        identity: tuple[str, int, float],
        spans: PayloadSpans | None,
    ) -> None:
        """Index a binary summary value lazily by source record position."""
        source_field = (
            _IMAGE_FIELD if "image" in value else 6 if "audio" in value else 8
        )
        tensor = value.get("tensor", {})
        strings = tensor.get("string_val", ())
        parts = self._binary_parts(value, tensor, len(strings))
        if parts is None:
            return
        frame = record.reference(
            cache.progress.identity, fingerprint=cache.progress.checked_status
        )
        candidates: list[Span] | None = None
        tag, step, wall_time = identity
        for part_kind, part, suffix in parts:
            data = _Classifier.binary_blob(value) if part is None else strings[part]
            if data is None:
                continue
            extension = self._binary_extension(value, part_kind, data)
            if extension is None:
                continue
            if spans is not None and candidates is None:
                candidates = self._binary_candidates(
                    spans, ordinal, source_field, part, len(strings)
                )
            span = (
                spans.match(candidates[0 if part is None else part], data)
                if spans is not None and candidates
                else None
            )
            entry_type = GraphSummaryEntry if part_kind == "graphs" else BinaryEntry
            entry = entry_type(
                source_path=cache.source_path,
                tag=tag,
                kind=part_kind,
                step=step,
                wall_time=wall_time,
                value_ordinal=ordinal,
                blob=_indexed_blob(frame, data, span),
                extension=extension,
                part=part,
                suffix=suffix,
            )
            if isinstance(entry, GraphSummaryEntry):
                graph, profile = _TensorSummary.graph_views(value)
                entry.converts = graph and summary_graph_def(entry, data) is not None
                entry.profiles = profile and run_metadata_text(entry, data) is not None
            cache.binaries.append(entry)

    @staticmethod
    def _binary_parts(
        value: dict[str, Any], tensor: dict[str, Any], strings: int
    ) -> Iterable[tuple[str, int | None, str]] | None:
        """Select tensor samples or the bytes of one legacy summary value."""
        if tensor:
            parts = _TensorSummary.media_parts(value)
            if parts is not None:
                return parts
            plugin = _Classifier.plugin_name(value)
            if plugin in {"images", "audio"} or plugin in GRAPH_PLUGINS:
                return None
        kind = _Classifier.binary_kind(value)
        if kind is None:
            return None
        if strings and not tensor.get("tensor_content"):
            width = len(str(strings - 1))
            return (
                (kind, i, f"_{i:0{width}d}" if strings > 1 else "")
                for i in range(strings)
            )
        return [(kind, None, "")]

    def _binary_extension(
        self, value: dict[str, Any], kind: str, data: bytes
    ) -> str | None:
        """Select special graph and packed-tensor extensions."""
        if kind == "graphs":
            return GRAPH_PLUGINS[_Classifier.plugin_name(value)][1]
        if kind == "tensors" and value.get("packed_float32"):
            return "bin"
        return self._cached_extension(kind, data)

    def _cached_extension(self, kind: str, data: bytes) -> str | None:
        """Reuse exact-byte media classifications within this parse."""
        size = len(data)
        if size <= 4096:
            cached = self.binary_extensions
            key = (kind, data)
            if key in cached:
                return cached[key]
            if len(cached) == 128:
                cached.clear()
            cached[key] = extension = _TensorSummary.part_extension(kind, data)
            return extension
        if kind == "audio" and size <= _LARGE_AUDIO_MEMO_LIMIT:
            previous = self.large_audio_extension
            if previous is not None and data == previous[0]:
                return previous[1]
            extension = _TensorSummary.part_extension(kind, data)
            self.large_audio_extension = (data, extension)
            return extension
        return _TensorSummary.part_extension(kind, data)

    @staticmethod
    def _binary_candidates(
        spans: PayloadSpans,
        ordinal: int,
        source_field: int,
        part: int | None,
        string_count: int,
    ) -> list[Span]:
        """Reject ambiguous field merges and mismatched repeated-field counts."""
        if part is not None:
            found = spans.value_fields(ordinal, 8, 8)
            return found if len(found) == string_count else []
        found = spans.value_fields(ordinal, source_field, 4)
        return found if len(found) == 1 else []
