from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import partial
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO

from google.protobuf.message import DecodeError
from loguru import logger
import numpy as np

from tboardfs.classify import _Classifier
from tboardfs.constants import PLUGIN_TAB
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
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
from tboardfs.meta_graph import meta_graph_graph_def
from tboardfs.proto_schema import _ProtoParser
from tboardfs.restarts import _RestartPurge
from tboardfs.run_metadata_text import run_metadata_text
from tboardfs.summary import parse_event
from tboardfs.skip_failure import skip_failure
from tboardfs.summary_graph import summary_graph_def
from tboardfs.tensor_summary import GRAPH_PLUGINS, _TensorSummary
from tboardfs.tables import (
    _TableMerge,
    _TableRows,
)
from tboardfs.tfrecord import iter_tfrecords

# A framed record whose payload cannot be decoded or indexed; framing is intact.
_MALFORMED_RECORD_ERRORS = (DecodeError, ValueError, struct.error)


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


@dataclass
class _ParseState:
    """The table rows and the clock of one event file parse.

    :ivar scalar_rows: scalar columns by tag
    :ivar scalar_dtypes: dtype of each scalar row by tag
    :ivar histogram_rows: histogram bucket columns by tag
    :ivar first_time: wall time of the first event with values
    :ivar epoch: the newest value of the epoch scalar
    """

    scalar_rows: dict[str, dict[str, list[Any]]] = field(
        default_factory=lambda: defaultdict(_TableRows.empty_scalar_rows)
    )
    scalar_dtypes: dict[str, list[np.dtype]] = field(
        default_factory=lambda: defaultdict(list)
    )
    histogram_rows: dict[str, dict[str, list[Any]]] = field(
        default_factory=lambda: defaultdict(_TableRows.empty_histogram_rows)
    )
    first_time: float | None = None
    epoch: object = None


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
    ) -> EventFileCache:
        """Index a validated prefix and appended frames on one descriptor."""
        with path.open("rb") as handle:
            status = os.fstat(handle.fileno())
            if base_cache is not None and (
                base_cache.progress.identity != FileIdentity.of(status)
                or status.st_size < base_cache.progress.processed_pos
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
    ) -> EventFileCache:
        """Index validated frames from the same descriptor checked for continuity."""
        cache = EventFileCache(source_path=path)
        progress = cache.progress
        progress.identity = FileIdentity.of(status)
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
        state = _ParseState()
        processed_pos = 0
        if base_cache is not None:
            processed_pos = base_cache.progress.processed_pos
            progress.frames = base_cache.progress.frames.copy()
            state.first_time = base_cache.progress.first_time
            state.epoch = base_cache.progress.epoch
            cache.binaries = list(base_cache.binaries)
            cache.graphs = list(base_cache.graphs)
            cache.json_entries = list(base_cache.json_entries)
            cache.scalar_mtimes = dict(base_cache.scalar_mtimes)
            cache.histogram_mtimes = dict(base_cache.histogram_mtimes)

        handle.seek(processed_pos)
        try:
            for record in iter_tfrecords(handle):
                processed_pos = record.record_offset + record.total_size
                progress.frames.append(processed_pos, record.payload_crc)
                try:
                    _EventIndexer.index_record(
                        cache, record, state, strict=not ignore_truncated
                    )
                except _MALFORMED_RECORD_ERRORS as error:
                    if not ignore_truncated:
                        raise
                    logger.warning(
                        "skipping malformed record at offset {} in {}: {}",
                        record.record_offset,
                        path,
                        error,
                    )
        except TFRecordError as error:
            if strict_framing or not ignore_truncated:
                raise TFRecordError(
                    f"{path}: {error}", corrupt=error.corrupt
                ) from error
            if not _EventIndexer.valid_anchor(handle, cache):
                if not retry:
                    logger.warning("discarding unstable prefix of {}: {}", path, error)
                    return EventFileCache(
                        source_path=path,
                        progress=ParseProgress(
                            initial_metadata=progress.initial_metadata,
                            effective_metadata=dict(progress.initial_metadata),
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
                )
            if error.corrupt:
                logger.warning("stopped reading {}: {}", path, error)

        progress.processed_pos = processed_pos
        progress.last_checked_size = status.st_size
        progress.first_time = state.first_time
        progress.epoch = state.epoch
        progress.checked_status = FileIdentity.fingerprint(status)
        cache.scalars = _TableRows.arrays_from_scalar_rows(
            state.scalar_rows, state.scalar_dtypes
        )
        cache.histograms = _TableRows.arrays_from_histogram_rows(state.histogram_rows)
        if base_cache is not None:
            cache.scalars = _TableMerge.append_series(base_cache.scalars, cache.scalars)
            cache.histograms = _TableMerge.append_series(
                base_cache.histograms, cache.histograms
            )
            cache.restarts = _EventIndexer.rebase_restarts(cache.restarts, base_cache)
        return cache

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
        return [
            *base_cache.restarts,
            *(
                replace(
                    mark,
                    counts={
                        key: lengths.get(key, 0) + mark.counts.get(key, 0)
                        for key in lengths.keys() | mark.counts.keys()
                    },
                )
                if mark.kind == "start"
                else mark
                for mark in marks
            ),
        ]

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
        event = parse_event(record.payload)
        wall_time = float(event["wall_time"])
        values = event.get("values", [])
        if first_time is None and values:
            first_time = wall_time
        relative_time = np.nan if first_time is None else wall_time - first_time
        timing = (event["step"], wall_time, relative_time)
        _EventIndexer.add_restart(cache, event, state)
        _EventIndexer.add_graph(cache, path, event, record, wall_time)
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
            indexed = skip_failure(
                f"value {value['tag']!r} at step {event['step']} in {path}",
                partial(
                    _EventIndexer.add_summary,
                    cache,
                    ordinal,
                    record,
                    value,
                    state,
                    (epoch, *timing),
                ),
                strict=strict,
            )
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
            (table, tag): len(columns["step"])
            for table, rows_by_tag in (
                ("scalars", state.scalar_rows),
                ("histograms", state.histogram_rows),
            )
            for tag, columns in rows_by_tag.items()
        }
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
    ) -> None:
        """Index graph and meta graph payloads found in an event record."""
        if event.get("graph_def") is not None:
            cache.graphs.append(
                GraphEntry(
                    source_path=path,
                    wall_time=wall_time,
                    frame=record.reference(cache.progress.identity),
                )
            )
        meta_graph = event.get("meta_graph_def")
        if meta_graph is None:
            return
        entry = MetaGraphEntry(
            source_path=path,
            wall_time=wall_time,
            frame=record.reference(cache.progress.identity),
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
    ) -> object:
        """Index one TensorBoard summary value.

        A value without a tag is kept, as TensorBoard keeps it; its files are
        named "~empty".
        """
        tag = value["tag"]
        epoch, step, wall_time, relative_time = timing
        epoch = _EventIndexer.add_scalar(
            cache,
            value,
            state.scalar_rows,
            state.scalar_dtypes,
            (epoch, step, wall_time, relative_time),
        )
        # A TF1 or PyTorch scalar without metadata holds nothing else.
        if value.keys() == {"tag", "simple_value"}:
            return epoch
        _EventIndexer.add_histogram(
            cache, value, state.histogram_rows, (epoch, step, wall_time, relative_time)
        )
        _EventIndexer.add_binary(cache, ordinal, record, value, (tag, step, wall_time))
        _EventIndexer.add_plugin(cache, value, (tag, step, wall_time))
        return epoch

    @staticmethod
    def add_scalar(
        cache: EventFileCache,
        value: dict[str, Any],
        scalar_rows: dict[str, dict[str, list[Any]]],
        scalar_dtypes: dict[str, list[np.dtype]],
        timing: tuple[Any, Any, float, float],
    ) -> object:
        """Append one scalar row when a summary value is scalar-shaped."""
        tag = value["tag"]
        scalar = _Classifier.scalar_value(value)
        if scalar is None:
            return timing[0]
        epoch, step, wall_time, relative_time = timing
        if tag == "epoch":
            epoch = scalar.value
        rows = scalar_rows[tag]
        rows["epoch"].append(epoch)
        rows["step"].append(step)
        rows["wall_time"].append(wall_time)
        rows["relative_time"].append(relative_time)
        rows["value"].append(scalar.value)
        scalar_dtypes[tag].append(scalar.dtype)
        cache.scalar_mtimes[tag] = wall_time
        return epoch

    @staticmethod
    def add_histogram(
        cache: EventFileCache,
        value: dict[str, Any],
        histogram_rows: dict[str, dict[str, list[Any]]],
        timing: tuple[Any, Any, float, float],
    ) -> None:
        """Append histogram bucket rows when present."""
        buckets = _TensorSummary.histogram_buckets(value)
        if buckets is None:
            return
        tag = value["tag"]
        epoch, step, wall_time, relative_time = timing
        rows = histogram_rows[tag]
        for index, (left, right, count) in enumerate(buckets):
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
    ) -> None:
        """Index a binary summary value lazily by source record position."""
        tag, step, wall_time = identity
        parts: Iterable[tuple[str, int | None, str]] | None = (
            _TensorSummary.media_parts(value)
        )
        tensor = value.get("tensor", {})
        strings = tensor.get("string_val", ())
        if parts is None:
            plugin = _Classifier.plugin_name(value)
            if tensor and (plugin in {"images", "audio"} or plugin in GRAPH_PLUGINS):
                return
            kind = _Classifier.binary_kind(value)
            if kind is None:
                return
            if strings and not tensor.get("tensor_content"):
                width = len(str(len(strings) - 1))
                parts = (
                    (kind, i, f"_{i:0{width}d}" if len(strings) > 1 else "")
                    for i in range(len(strings))
                )
            else:
                parts = [(kind, None, "")]
        frame = record.reference(cache.progress.identity)
        for part_kind, part, suffix in parts:
            data = _Classifier.binary_blob(value) if part is None else strings[part]
            if data is None:
                continue
            extension = _TensorSummary.part_extension(value, part_kind, data)
            if extension is None:
                continue
            entry = BinaryEntry(
                source_path=cache.source_path,
                tag=tag,
                kind=part_kind,
                step=step,
                wall_time=wall_time,
                frame=frame,
                value_ordinal=ordinal,
                extension=extension,
                part=part,
                suffix=suffix,
            )
            if part_kind == "graphs":
                graph, profile = _TensorSummary.graph_views(value)
                entry = GraphSummaryEntry(
                    **vars(entry),
                    converts=graph and summary_graph_def(entry, data) is not None,
                    profiles=profile and run_metadata_text(entry, data) is not None,
                )
            cache.binaries.append(entry)

    @staticmethod
    def add_plugin(
        cache: EventFileCache,
        value: dict[str, Any],
        identity: tuple[str, Any, float],
    ) -> None:
        """Index plugin metadata as JSON records."""
        payload = _Classifier.plugin_json_payload(value)
        if payload is None or _TensorSummary.unpacked(value):
            return
        tag, step, wall_time = identity
        plugin_name = str(payload.get("plugin_name") or "unknown").lower()
        tab = PLUGIN_TAB.get(plugin_name, "plugins")
        error = _EventIndexer.plugin_error(tab, payload)
        if error is not None:
            logger.warning(
                "skipping {} record {} at step {} in {}: {}",
                plugin_name,
                tag,
                step,
                cache.source_path,
                error,
            )
            return
        cache.json_entries.append(
            JsonEntry(tag=tag, tab=tab, step=step, wall_time=wall_time, payload=payload)
        )

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
