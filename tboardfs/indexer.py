import os
import struct
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from google.protobuf.message import DecodeError
from loguru import logger

from tboardfs.classify import _Classifier
from tboardfs.constants import PLUGIN_TAB
from tboardfs.crc32c import masked_crc32c
from tboardfs.event_source import SCAN_BUFFER, EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.media_index import _BinaryWriter, _MediaIndex
from tboardfs.meta_graph import meta_graph_graph_def
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
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
from tboardfs.payload_spans import PayloadSpans
from tboardfs.proto_schema import _ProtoParser
from tboardfs.range_proof import range_proof
from tboardfs.restarts import _offset_restarts, _RestartPurge
from tboardfs.skip_failure import _diagnostic_provenance, _skip_error, _warn_skipped
from tboardfs.summary import _Image, _SummaryParser
from tboardfs.tables import (
    _ScalarRows,
    _TableMerge,
    _TableRows,
)
from tboardfs.tensor_summary import _TensorSummary
from tboardfs.tfrecord import _iter_tfrecords

# A framed record whose payload cannot be decoded or indexed; framing is intact.
_MALFORMED_RECORD_ERRORS = (DecodeError, ValueError, struct.error)
_FLOAT32 = np.dtype("float32")
_FLOAT64 = np.dtype("float64")
_SCALAR_FIELDS = frozenset(("tag", "simple_value", "plugin_name", "plugin_content"))


def parse_file(filename: str | Path) -> dict[str, Any]:
    """Return parser data for one TensorBoard event file."""
    cache = _EventIndexer.parse_event_file(Path(filename))
    last: dict[str, dict[str, BinaryEntry]] = {
        "images": {},
        "audio": {},
        "videos": {},
        "meshes": {},
        "tensors": {},
    }
    for entry in cache.binaries:
        if (tags := last.get(entry.kind)) is not None:
            tags[entry.tag] = entry
    binaries = {
        kind: {
            tag: (entry.frame.payload_offset, entry.frame.payload_size)
            for tag, entry in tags.items()
        }
        for kind, tags in last.items()
    }

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
    media: _MediaIndex = field(default_factory=_MediaIndex)
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
        """Index a validated prefix and appended frames on one descriptor.

        Persistent indexes consume every image frame along with the tensor extents.
        """
        buffering = SCAN_BUFFER if base_cache is None else -1
        with EventSource.open(
            path, mountpoint=mountpoint, buffering=buffering
        ) as handle:
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
                state=_ParseState(
                    retain_tensor_extents=retain_tensor_extents,
                    media=_MediaIndex(defer_frames=not retain_tensor_extents),
                ),
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
            cache.binaries = base_cache.binaries.copy()
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
                        retain_tensor_extents=state.retain_tensor_extents,
                        media=_MediaIndex(defer_frames=state.media.defer_frames),
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
            for record in _iter_tfrecords(handle):
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
        message = cast(Any, _ProtoParser.event(record.payload))
        kind = message.WhichOneof("what")
        wall_time = message.wall_time
        spans = PayloadSpans.for_payload(record.payload)
        plain = None
        values: Any
        if kind == "summary":
            source = message.summary.value
            plain = _SummaryParser.plain_values(
                source, cache.progress.effective_metadata
            )
            values = _SummaryParser.index_values(source) if plain is None else plain
        else:
            event = _SummaryParser.convert_event(message, kind, indexing=True)
            _EventIndexer.add_restart(cache, event, state)
            _EventIndexer.add_graph(
                cache, cache.source_path, event, record, wall_time, spans
            )
            values = event.get("values", ())
        first_time = _EventIndexer._first_time(state, wall_time, bool(values))
        relative_time = np.nan if first_time is None else wall_time - first_time
        step = message.step
        timing = (step, wall_time, relative_time)
        if isinstance(plain, tuple):
            scalar_tag, scalar, image_tag, image = plain
            epoch = state.epoch
            if scalar is not None:
                epoch = _EventIndexer.add_scalar(
                    state, scalar_tag, scalar, _FLOAT32, epoch, timing
                )
            if image is not None:
                _EventIndexer._index_plain_image(
                    cache,
                    record,
                    state.media,
                    0 if scalar is None else 1,
                    image,
                    (image_tag, step, wall_time),
                    spans,
                    state.media.writer if spans is None else None,
                    strict,
                )
        else:
            index_values = (
                _EventIndexer._index_plain_values
                if plain is not None
                else _EventIndexer._index_converted_values
            )
            binary_start = len(cache.binaries)
            epoch = index_values(cache, record, state, values, timing, spans, strict)
            if len(cache.binaries) - binary_start > 1 and len(values) > 1:
                _EventIndexer._deduplicate_binaries(cache, binary_start)
        state.first_time, state.epoch = first_time, epoch

    @staticmethod
    def _index_plain_values(
        cache: EventFileCache,
        record: Record,
        state: _ParseState,
        values: list[tuple[str, float | _Image]],
        timing: tuple[int, float, float],
        spans: PayloadSpans | None,
        strict: bool,
    ) -> object:
        """Write prepared values to scalar columns or binary entries."""
        epoch = state.epoch
        step, wall_time, _ = timing
        writer = state.media.writer if spans is None else None
        count = len(values)
        for ordinal, (tag, data) in enumerate(values):
            if isinstance(data, float):
                epoch = _EventIndexer.add_scalar(
                    state, tag, data, _FLOAT32, epoch, timing
                )
                continue
            image_writer = None
            if writer is not None:
                if ordinal + 1 == count or all(
                    isinstance(values[i][1], float) for i in range(ordinal + 1, count)
                ):
                    image_writer = writer
                writer = None
            _EventIndexer._index_plain_image(
                cache,
                record,
                state.media,
                ordinal,
                data,
                (tag, step, wall_time),
                spans,
                image_writer,
                strict,
            )
        return epoch

    @staticmethod
    def _index_plain_image(
        cache: EventFileCache,
        record: Record,
        media: _MediaIndex,
        ordinal: int,
        image: _Image,
        identity: tuple[str, int, float],
        spans: PayloadSpans | None,
        writer: _BinaryWriter | None,
        strict: bool,
    ) -> None:
        """Index an ordinary image with the per-value error policy."""
        try:
            media.add_legacy_image(
                cache,
                ordinal,
                record,
                image.encoded_image_string,
                identity,
                spans,
                writer=writer,
            )
        except Exception as error:
            tag, step, _ = identity
            if not _skip_error(
                error,
                f"value {tag!r} at step {step} in {cache.source_path}",
                strict=strict,
            ):
                raise

    @staticmethod
    def _first_time(
        state: _ParseState, wall_time: float, has_values: bool
    ) -> float | None:
        first_time = state.first_time
        clock = state.clock
        if clock is not None and has_values and not clock.reads_first_time:
            clock.reads_first_time = True
            clock.writes_first_time = first_time is None
        return wall_time if first_time is None and has_values else first_time

    @staticmethod
    def _index_converted_values(
        cache: EventFileCache,
        record: Record,
        state: _ParseState,
        values: Iterable[dict[str, Any]],
        timing: tuple[int, float, float],
        spans: PayloadSpans | None,
        strict: bool,
    ) -> object:
        path = cache.source_path
        metadata = cache.progress.effective_metadata
        epoch = state.epoch
        for ordinal, value in enumerate(values):
            tag = value["tag"]
            if "plugin_name" in value:
                metadata[tag] = {
                    "plugin_name": value["plugin_name"],
                    "plugin_content": value.get("plugin_content", b""),
                }
            elif tag in metadata:
                value.update(metadata[tag])
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
        return epoch

    @staticmethod
    def _deduplicate_binaries(cache: EventFileCache, binary_start: int) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        for entry in cache.binaries[binary_start:]:
            key = (entry.kind, entry.tag, entry.suffix, entry.extension)
            if key in seen:
                entry.suffix += f"_v{entry.value_ordinal}"
            seen.add(key)

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
        if (writer := state.media.writer) is not None:
            counts[("binaries", "")] += len(writer)
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
        if value.keys() <= _SCALAR_FIELDS:
            scalar_value = value.get("simple_value")
            if scalar_value is not None:
                epoch = _EventIndexer.add_scalar(
                    state,
                    tag,
                    scalar_value,
                    _FLOAT32,
                    epoch,
                    (step, wall_time, relative_time),
                )
            if scalar_value is None or _Classifier.plugin_name(value) != "scalars":
                _EventIndexer.add_plugin(cache, value, (tag, step, wall_time))
            return epoch
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
        state.media.add_binary(
            cache,
            ordinal,
            record,
            value,
            (tag, step, wall_time),
            spans,
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
        value: float,
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
