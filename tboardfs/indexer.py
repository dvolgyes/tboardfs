from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO

from google.protobuf.message import DecodeError
from loguru import logger
import numpy as np

from tboardfs.classify import _Classifier
from tboardfs.constants import PLUGIN_TAB
from tboardfs.crc32c import masked_crc32c
from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.meta_graph import meta_graph_graph_def
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
    GraphSummaryEntry,
    GraphEntry,
    JsonEntry,
    MetaGraphEntry,
    ParseProgress,
    Record,
    RestartMark,
    TFRecordError,
)
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.model.parse_progress import _ParseProvenance
from tboardfs.payload_spans import PayloadSpans, Span
from tboardfs.proto_schema import _ProtoParser
from tboardfs.range_proof import range_proof
from tboardfs.restarts import _RestartPurge, _offset_restarts
from tboardfs.run_metadata_text import run_metadata_text
from tboardfs.skip_failure import _diagnostic_provenance, _skip_error, _warn_skipped
from tboardfs.summary import _SummaryParser
from tboardfs.summary_graph import summary_graph_def
from tboardfs.tensor_summary import GRAPH_PLUGINS, _TensorSummary
from tboardfs.tables import (
    _ScalarRows,
    _TableMerge,
    _TableRows,
)
from tboardfs.tfrecord import iter_tfrecords

# A framed record whose payload cannot be decoded or indexed; framing is intact.
_MALFORMED_RECORD_ERRORS = (DecodeError, ValueError, struct.error)
_FLOAT32 = np.dtype("float32")
_FLOAT64 = np.dtype("float64")
_LARGE_AUDIO_MEMO_LIMIT = 1 << 20


def parse_file(filename: str | Path) -> dict[str, Any]:
    """Return parser data for one TensorBoard event file."""
    cache = _EventIndexer.parse_event_file(Path(filename))
    binaries: dict[str, dict[str, tuple[int, int]]] = {
        "images": {},
        "audio": {},
        "videos": {},
        "meshes": {},
        "tensors": {},
    }
    for entry in cache.binaries:
        if entry.kind in binaries:
            binaries[entry.kind][entry.tag] = (
                entry.frame.payload_offset,
                entry.frame.payload_size,
            )

    for series in cache.histograms.values():
        del series["_start"]
        del series["_inherits_epoch"]
    for series in cache.scalars.values():
        del series["_inherits_epoch"]
    return {
        "binaries": binaries,
        "binary_entries": cache.binaries,
        "graphs": cache.graphs,
        "histograms": _TableMerge.step_epochs(
            cache.histograms, cache.scalars.get("epoch")
        ),
        "json_entries": cache.json_entries,
        "scalars": _TableMerge.sort_scalars(
            _TableMerge.step_epochs(cache.scalars, cache.scalars.get("epoch"))
        ),
    }


@dataclass(slots=True)
class _ClockDependencies:
    """Record which inherited clock seeds were used or replaced."""

    first_time: float | None
    epoch: object
    reads_first_time: bool = False
    reads_epoch: bool = False
    writes_first_time: bool = False
    writes_epoch: bool = False


@dataclass
class _ParseState:
    """The table rows and the clock of one event file parse.

    :ivar scalar_rows: scalar columns by tag
    :ivar histogram_rows: histogram bucket columns by tag
    :ivar first_time: wall time of the first event with values
    :ivar epoch: the newest value of the epoch scalar
    """

    scalar_rows: dict[str, _ScalarRows] = field(
        default_factory=lambda: defaultdict(_ScalarRows)
    )
    histogram_rows: dict[str, dict[str, list[Any]]] = field(
        default_factory=lambda: defaultdict(_TableRows.empty_histogram_rows)
    )
    first_time: float | None = None
    epoch: object = None
    binary_extensions: dict[tuple[str, bytes], str | None] = field(default_factory=dict)
    large_audio_extension: tuple[bytes, str | None] | None = None
    clock: _ClockDependencies | None = None
    deferred_tensor_content: list[tuple[int, int, int, int]] | None = None
    retain_tensor_extents: bool = False


class _EventIndexer:
    """Build event-file indexes from parsed TensorBoard records."""

    @staticmethod
    def parse_event_file(
        path: Path,
        *,
        ignore_truncated: bool = False,
        strict_framing: bool = False,
        base_cache: EventFileCache | None = None,
        initial_metadata: dict[str, dict[str, Any]] | None = None,
        mountpoint: Path | None = None,
        retain_tensor_extents: bool = False,
    ) -> EventFileCache:
        """Index a validated prefix and appended frames on one descriptor."""
        with EventSource.open(path, mountpoint=mountpoint) as handle:
            status = os.fstat(handle.fileno())
            if base_cache is not None and (
                base_cache.progress.identity != FileIdentity.of(status)
                or status.st_size < base_cache.progress.last_checked_size
                or (
                    status.st_size == base_cache.progress.last_checked_size
                    and FileIdentity.fingerprint(status)
                    != base_cache.progress.checked_status
                )
                or not _EventIndexer.valid_anchor(handle, base_cache)
            ):
                base_cache = None
            return _EventIndexer.parse_open(
                path,
                handle,
                status,
                base_cache,
                initial_metadata,
                ignore_truncated=ignore_truncated,
                strict_framing=strict_framing,
                state=_ParseState(retain_tensor_extents=True)
                if retain_tensor_extents
                else None,
            )

    @staticmethod
    def parse_open(
        path: Path,
        handle: BinaryIO,
        status: os.stat_result,
        base_cache: EventFileCache | None,
        initial_metadata: dict[str, dict[str, Any]] | None,
        *,
        ignore_truncated: bool,
        strict_framing: bool,
        retry: bool = True,
        state: _ParseState | None = None,
    ) -> EventFileCache:
        """Index validated frames from the same descriptor checked for continuity."""
        cache = EventFileCache(source_path=path)
        progress = cache.progress
        provenance = progress.provenance
        provenance.cacheable = retry
        progress.identity = FileIdentity.of(status)
        progress.checked_status = FileIdentity.fingerprint(status)
        progress.initial_metadata = (
            base_cache.progress.initial_metadata
            if base_cache is not None
            else dict(initial_metadata or {})
        )
        progress.effective_metadata = dict(
            base_cache.progress.effective_metadata
            if base_cache is not None
            else progress.initial_metadata
        )
        state = _ParseState() if state is None else state
        if base_cache is not None:
            provenance.resumed = True
            provenance.cacheable = retry and base_cache.progress.provenance.cacheable
            progress.processed_pos = base_cache.progress.processed_pos
            progress.frames = base_cache.progress.frames.copy()
            state.first_time = base_cache.progress.first_time
            state.epoch = base_cache.progress.epoch
            cache.binaries = list(base_cache.binaries)
            cache.graphs = list(base_cache.graphs)
            cache.json_entries = list(base_cache.json_entries)
            cache.scalar_mtimes = dict(base_cache.scalar_mtimes)
            cache.histogram_mtimes = dict(base_cache.histogram_mtimes)

        handle.seek(progress.processed_pos)
        try:
            _EventIndexer._index_records(
                handle, cache, state, strict=not ignore_truncated
            )
        except TFRecordError as error:
            if strict_framing or not ignore_truncated:
                raise TFRecordError(
                    f"{path}: {error}", corrupt=error.corrupt
                ) from error
            provenance.cacheable = False
            if not _EventIndexer.valid_anchor(handle, cache):
                if not retry:
                    logger.warning("discarding unstable prefix of {}: {}", path, error)
                    return EventFileCache(
                        source_path=path,
                        progress=ParseProgress(
                            initial_metadata=progress.initial_metadata,
                            effective_metadata=dict(progress.initial_metadata),
                            provenance=_ParseProvenance(cacheable=False),
                        ),
                    )
                return _EventIndexer.parse_open(
                    path,
                    handle,
                    os.fstat(handle.fileno()),
                    None,
                    initial_metadata,
                    ignore_truncated=ignore_truncated,
                    strict_framing=strict_framing,
                    retry=False,
                    state=_ParseState(
                        retain_tensor_extents=state.retain_tensor_extents
                    ),
                )
            if error.corrupt:
                logger.warning("stopped reading {}: {}", path, error)

        provenance.cacheable = (
            provenance.cacheable
            and _EventIndexer._complete_snapshot(
                handle, cache, status, state.retain_tensor_extents
            )
        )
        progress.last_checked_size = status.st_size
        progress.first_time = state.first_time
        progress.epoch = state.epoch
        for tag, rows in state.scalar_rows.items():
            cache.scalars[tag] = rows.arrays()
            cache.scalar_mtimes[tag] = rows.wall_time[-1]
        cache.histograms = _TableRows.arrays_from_histogram_rows(state.histogram_rows)
        if base_cache is not None:
            cache.scalars = _TableMerge.append_series(base_cache.scalars, cache.scalars)
            cache.histograms = _TableMerge.append_series(
                base_cache.histograms, cache.histograms
            )
            cache.restarts = _EventIndexer.rebase_restarts(cache.restarts, base_cache)
        return cache

    @staticmethod
    def _index_records(
        handle: BinaryIO, cache: EventFileCache, state: _ParseState, *, strict: bool
    ) -> None:
        """Index records and scope conversion diagnostics to this cache."""
        progress = cache.progress
        provenance = progress.provenance
        token = _diagnostic_provenance.set(provenance)
        try:
            for record in iter_tfrecords(handle):
                progress.processed_pos = record.record_offset + record.total_size
                progress.frames.append(progress.processed_pos, record.payload_crc)
                try:
                    _EventIndexer.index_record(cache, record, state, strict=strict)
                except _MALFORMED_RECORD_ERRORS as error:
                    if strict:
                        raise
                    provenance.cacheable = False
                    logger.warning(
                        "skipping malformed record at offset {} in {}: {}",
                        record.record_offset,
                        cache.source_path,
                        error,
                    )
        finally:
            _diagnostic_provenance.reset(token)

    @staticmethod
    def _complete_snapshot(
        handle: BinaryIO,
        cache: EventFileCache,
        status: os.stat_result,
        verify_stability: bool,
    ) -> bool:
        """Require a whole nonempty file and, for disk reuse, unchanged status."""
        progress = cache.progress
        return 0 < progress.processed_pos == status.st_size and (
            not verify_stability
            or FileIdentity.fingerprint(os.fstat(handle.fileno()))
            == progress.checked_status
        )

    @staticmethod
    def rebase_restarts(
        marks: list[RestartMark], base_cache: EventFileCache
    ) -> list[RestartMark]:
        """Count the table rows before each new START from the file start."""
        lengths = {
            (table, tag): len(series["step"])
            for table, series_by_tag in (
                ("scalars", base_cache.scalars),
                ("histograms", base_cache.histograms),
            )
            for tag, series in series_by_tag.items()
        }
        return [*base_cache.restarts, *_offset_restarts(marks, lengths)]

    @staticmethod
    def valid_anchor(handle: BinaryIO, cache: EventFileCache) -> bool:
        """Check the last complete indexed frame before retaining its prefix."""
        anchor = cache.progress.frames.last()
        if anchor is None:
            return True
        try:
            IndexedFrame.verify(handle, FrameRef(cache.progress.identity, *anchor))
        except TFRecordError:
            return False
        return True

    @staticmethod
    def index_record(
        cache: EventFileCache,
        record: Record,
        state: _ParseState,
        *,
        strict: bool,
    ) -> None:
        """Index one framed record, and update the clock when it is indexed.

        A summary value that fails to index is skipped, unless strict.
        """
        first_time, epoch = state.first_time, state.epoch
        path = cache.source_path
        event = _SummaryParser.parse_event(record.payload, indexing=True)
        wall_time = float(event["wall_time"])
        values = event.get("values", [])
        clock = state.clock
        if clock is not None and values and not clock.reads_first_time:
            clock.reads_first_time = True
            clock.writes_first_time = first_time is None
        if first_time is None and values:
            first_time = wall_time
        relative_time = np.nan if first_time is None else wall_time - first_time
        timing = (event["step"], wall_time, relative_time)
        _EventIndexer.add_restart(cache, event, state)
        spans = PayloadSpans.for_payload(record.payload)
        _EventIndexer.add_graph(cache, path, event, record, wall_time, spans)
        binary_start = len(cache.binaries)
        for ordinal, value in enumerate(values):
            tag = value["tag"]
            if "plugin_name" in value:
                cache.progress.effective_metadata[tag] = {
                    "plugin_name": value["plugin_name"],
                    "plugin_content": value.get("plugin_content", b""),
                }
            elif tag in cache.progress.effective_metadata:
                value.update(cache.progress.effective_metadata[tag])
            if len(value) == 2 and "simple_value" in value:
                epoch = _EventIndexer.add_scalar(
                    state, tag, value["simple_value"], _FLOAT32, epoch, timing
                )
                continue
            try:
                indexed = _EventIndexer.add_summary(
                    cache, ordinal, record, value, state, (epoch, *timing), spans
                )
            except Exception as error:
                if not _skip_error(
                    error, f"value {tag!r} at step {timing[0]} in {path}", strict=strict
                ):
                    raise
            else:
                epoch = epoch if indexed is None else indexed
        if len(values) > 1 and len(cache.binaries) - binary_start > 1:
            seen: set[tuple[str, str, str, str]] = set()
            for entry in cache.binaries[binary_start:]:
                key = (entry.kind, entry.tag, entry.suffix, entry.extension)
                if key in seen:
                    entry.suffix += f"_v{entry.value_ordinal}"
                seen.add(key)
        state.first_time, state.epoch = first_time, epoch

    @staticmethod
    def add_restart(
        cache: EventFileCache,
        event: dict[str, Any],
        state: _ParseState,
    ) -> None:
        """Record a file version or SessionLog.START before the event's values.

        The table counts cover the rows of this parse only; ``rebase_restarts``
        adds the rows of an earlier parse. The list counts cover the whole file.
        """
        if "file_version" in event:
            version = _RestartPurge.parse_version(event["file_version"])
            cache.restarts.append(RestartMark("version", version))
        if not event.get("session_start"):
            return
        counts = {
            ("scalars", tag): len(rows.step) for tag, rows in state.scalar_rows.items()
        }
        counts.update(
            (("histograms", tag), len(rows["step"]))
            for tag, rows in state.histogram_rows.items()
        )
        counts[("binaries", "")] = len(cache.binaries)
        counts[("json_entries", "")] = len(cache.json_entries)
        cache.restarts.append(RestartMark("start", event["step"], counts))

    @staticmethod
    def add_graph(
        cache: EventFileCache,
        path: Path,
        event: dict[str, Any],
        record: Record,
        wall_time: float,
        spans: PayloadSpans | None,
    ) -> None:
        """Index graph and meta graph payloads found in an event record."""
        graph = event.get("graph_def")
        meta_graph = event.get("meta_graph_def")
        if graph is None and meta_graph is None:
            return
        frame = record.reference(
            cache.progress.identity, fingerprint=cache.progress.checked_status
        )
        if graph is not None:
            span = spans.graph(4, graph) if spans is not None else None
            cache.graphs.append(
                GraphEntry(
                    source_path=path,
                    wall_time=wall_time,
                    blob=IndexedBlob(
                        frame,
                        len(graph),
                        span,
                        range_proof(graph) if span is not None else b"",
                    ),
                )
            )
        if meta_graph is None:
            return
        span = spans.graph(9, meta_graph) if spans is not None else None
        entry = MetaGraphEntry(
            source_path=path,
            wall_time=wall_time,
            blob=IndexedBlob(
                frame,
                len(meta_graph),
                span,
                range_proof(meta_graph) if span is not None else b"",
            ),
        )
        entry.readable = meta_graph_graph_def(entry, bytes(meta_graph)) is not None
        cache.graphs.append(entry)

    @staticmethod
    def add_summary(
        cache: EventFileCache,
        ordinal: int,
        record: Record,
        value: dict[str, Any],
        state: _ParseState,
        timing: tuple[Any, Any, float, float],
        spans: PayloadSpans | None,
    ) -> object:
        """Index one TensorBoard summary value.

        A value without a tag is kept, as TensorBoard keeps it; its files are
        named "~empty".
        """
        tag = value["tag"]
        epoch, step, wall_time, relative_time = timing
        plain_media = len(value) == 2 and ("image" in value or "audio" in value)
        scalar = None if plain_media else _Classifier.scalar_value(value)
        if scalar is not None:
            epoch = _EventIndexer.add_scalar(
                state,
                tag,
                scalar.value,
                scalar.dtype,
                epoch,
                (step, wall_time, relative_time),
            )
            if _Classifier.plugin_name(value) == "scalars":
                return epoch
        if not plain_media:
            _EventIndexer.add_histogram(
                cache, value, state, (epoch, step, wall_time, relative_time)
            )
        _EventIndexer.add_binary(
            cache,
            ordinal,
            record,
            value,
            (tag, step, wall_time),
            spans,
            state,
        )
        if plain_media:
            return epoch
        entry = _EventIndexer.add_plugin(cache, value, (tag, step, wall_time))
        if entry is not None and spans is not None and state.retain_tensor_extents:
            _EventIndexer._retain_extent(cache, state, entry, record, spans, ordinal)
        return epoch

    @staticmethod
    def add_scalar(
        state: _ParseState,
        tag: str,
        value: int | float,
        dtype: np.dtype,
        epoch: object,
        timing: tuple[int, float, float],
    ) -> object:
        """Append a decoded scalar to its tag's columns."""
        step, wall_time, relative_time = timing
        clock = state.clock
        if tag == "epoch":
            epoch = value
            if clock is not None:
                clock.writes_epoch = True
        elif clock is not None and not clock.writes_epoch:
            clock.reads_epoch = True
        rows = state.scalar_rows[tag]
        rows.epoch.append(epoch)
        rows.step.append(step)
        rows.wall_time.append(wall_time)
        rows.relative_time.append(relative_time)
        rows.value.append(value)
        if rows.dtype is None:
            rows.dtype = dtype
        elif (
            rows.dtype is not dtype
            and rows.dtype is not _FLOAT64
            and rows.dtype != dtype
        ):
            rows.dtype = _FLOAT64
        return epoch

    @staticmethod
    def add_histogram(
        cache: EventFileCache,
        value: dict[str, Any],
        state: _ParseState,
        timing: tuple[Any, Any, float, float],
    ) -> None:
        """Append histogram bucket rows when present."""
        buckets = _TensorSummary.histogram_buckets(value)
        if buckets is None:
            return
        tag = value["tag"]
        epoch, step, wall_time, relative_time = timing
        rows = state.histogram_rows[tag]
        clock = state.clock
        for index, (left, right, count) in enumerate(buckets):
            if index == 0 and clock is not None and not clock.writes_epoch:
                clock.reads_epoch = True
            rows["_start"].append(index == 0)
            rows["step"].append(step)
            rows["epoch"].append(epoch)
            rows["wall_time"].append(wall_time)
            rows["relative_time"].append(relative_time)
            rows["bucket_left"].append(left)
            rows["bucket_right"].append(right)
            rows["count"].append(count)
        cache.histogram_mtimes[tag] = wall_time

    @staticmethod
    def add_binary(
        cache: EventFileCache,
        ordinal: int,
        record: Record,
        value: dict[str, Any],
        identity: tuple[str, Any, float],
        spans: PayloadSpans | None,
        state: _ParseState,
    ) -> None:
        """Index a binary summary value lazily by source record position."""
        tag, step, wall_time = identity
        tensor = value.get("tensor", {})
        strings = tensor.get("string_val", ())
        parts = _EventIndexer.binary_parts(value, tensor, len(strings))
        if parts is None:
            return
        frame = record.reference(
            cache.progress.identity, fingerprint=cache.progress.checked_status
        )
        candidates: list[Span] | None = None
        for part_kind, part, suffix in parts:
            data = _Classifier.binary_blob(value) if part is None else strings[part]
            if data is None:
                continue
            extension = _EventIndexer.binary_extension(value, part_kind, data, state)
            if extension is None:
                continue
            if spans is not None and candidates is None:
                candidates = _EventIndexer.binary_candidates(
                    spans, ordinal, value, part, len(strings)
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
                blob=IndexedBlob(
                    frame,
                    len(data),
                    span,
                    range_proof(data) if span is not None else b"",
                ),
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
    def binary_parts(
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

    @staticmethod
    def binary_extension(
        value: dict[str, Any],
        kind: str,
        data: bytes,
        state: _ParseState,
    ) -> str | None:
        """Reuse exact-byte media classifications within this parse."""
        if kind == "tensors" and value.get("packed_float32"):
            return "bin"
        size = len(data)
        if size <= 4096 and kind != "graphs":
            cached = state.binary_extensions
            key = (kind, data)
            if key in cached:
                return cached[key]
            if len(cached) == 128:
                cached.clear()
            cached[key] = extension = _TensorSummary.part_extension(value, kind, data)
            return extension
        if kind == "audio" and size <= _LARGE_AUDIO_MEMO_LIMIT:
            previous = state.large_audio_extension
            if previous is not None and data == previous[0]:
                return previous[1]
            extension = _TensorSummary.part_extension(value, kind, data)
            state.large_audio_extension = (data, extension)
            return extension
        return _TensorSummary.part_extension(value, kind, data)

    @staticmethod
    def binary_candidates(
        spans: PayloadSpans,
        ordinal: int,
        value: dict[str, Any],
        part: int | None,
        string_count: int,
    ) -> list[Span]:
        """Reject ambiguous field merges and mismatched repeated-field counts."""
        if part is not None:
            found = spans.value_fields(ordinal, 8, 8)
            return found if len(found) == string_count else []
        field = 4 if "image" in value else 6 if "audio" in value else 8
        found = spans.value_fields(ordinal, field, 4)
        return found if len(found) == 1 else []

    @staticmethod
    def add_plugin(
        cache: EventFileCache,
        value: dict[str, Any],
        identity: tuple[str, Any, float],
    ) -> JsonEntry | None:
        """Index plugin metadata as JSON records."""
        payload = _Classifier.plugin_json_payload(value)
        if payload is None or _TensorSummary.unpacked(value):
            return None
        tag, step, wall_time = identity
        plugin_name = str(payload.get("plugin_name") or "unknown").lower()
        tab = PLUGIN_TAB.get(plugin_name, "plugins")
        error = _EventIndexer.plugin_error(tab, payload)
        if error is not None:
            _warn_skipped(
                "skipping {} record {} at step {} in {}: {}",
                plugin_name,
                tag,
                step,
                cache.source_path,
                error,
            )
            return None
        entry = JsonEntry(
            tag=tag, tab=tab, step=step, wall_time=wall_time, payload=payload
        )
        cache.json_entries.append(entry)
        return entry

    @staticmethod
    def _retain_extent(
        cache: EventFileCache,
        state: _ParseState,
        entry: JsonEntry,
        record: Record,
        spans: PayloadSpans,
        ordinal: int,
    ) -> None:
        """Retain verified source evidence without dropping eager serial bytes."""
        tensor = entry.payload.get("tensor")
        if tensor is None:
            return
        content = tensor.get("tensor_content")
        if not isinstance(content, bytes) or len(content) < 64 << 10:
            return
        fields = spans.value_fields(ordinal, 8, 4)
        if len(fields) != 1 or spans.match(fields[0], content) is None:
            return
        offset, size = fields[0]
        extent = (record.payload_offset + offset, size, masked_crc32c(content))
        entry.tensor_extent = extent
        if state.deferred_tensor_content is not None:
            state.deferred_tensor_content.append((len(cache.json_entries) - 1, *extent))
            tensor["tensor_content"] = b""

    @staticmethod
    def plugin_error(tab: str, payload: dict[str, Any]) -> str | None:
        """Return why the files of a plugin record cannot be made, or None.

        The hparams and mesh files decode the plugin content, and the custom
        scalar layout decodes a string of the tensor. A record that gives no
        message is not indexed, so it has no file.
        """
        parsers = {
            "hparams": _ProtoParser.hparams_plugin_data,
            "meshes": _ProtoParser.mesh_plugin_data,
        }
        content = payload.get("plugin_content")
        if tab in parsers and isinstance(content, bytes):
            blobs, parse = [content], parsers[tab]
        elif tab == "custom_scalars":
            blobs = list((payload.get("tensor") or {}).get("string_val") or [])
            parse = _ProtoParser.custom_scalar_layout
        else:
            return None
        error = "the tensor has no layout"
        for blob in blobs:
            try:
                parse(bytes(blob))
            except DecodeError as decode_error:
                error = str(decode_error)
            else:
                return None
        return error
