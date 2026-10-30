from collections import defaultdict
from dataclasses import replace
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO

from google.protobuf.message import DecodeError
from loguru import logger
import numpy as np

from tboardfs.classify import _Classifier, detect_extension
from tboardfs.constants import PLUGIN_TAB
from tboardfs.model import (
    BinaryEntry,
    EventFileCache,
    GraphEntry,
    JsonEntry,
    ParseProgress,
    Record,
    RestartMark,
    TFRecordError,
)
from tboardfs.restarts import _RestartPurge
from tboardfs.summary import parse_event
from tboardfs.tensor_summary import _TensorSummary
from tboardfs.tables import (
    _TableMerge,
    _TableRows,
)
from tboardfs.tfrecord import iter_tfrecords

# A framed record whose payload cannot be decoded or indexed; framing is intact.
_MALFORMED_RECORD_ERRORS = (DecodeError, ValueError, struct.error)
# The first record holds a wall time, so these bytes tell files apart.
IDENTITY_HEAD_BYTES = 64


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
            binaries[entry.kind][entry.tag] = (entry.payload_offset, entry.payload_size)

    return {
        "binaries": binaries,
        "binary_entries": cache.binaries,
        "graphs": cache.graphs,
        "histograms": cache.histograms,
        "json_entries": cache.json_entries,
        "scalars": cache.scalars,
    }


class _EventIndexer:
    """Build event-file indexes from parsed TensorBoard records."""

    @staticmethod
    def parse_event_file(
        path: Path,
        *,
        ignore_truncated: bool = False,
        start_pos: int = 0,
        base_cache: EventFileCache | None = None,
    ) -> EventFileCache:
        """Build an in-memory index for one TensorBoard event file.

        With ``base_cache``, only records from ``start_pos`` are parsed and
        their rows are appended, so the result equals a parse of the whole file.
        """
        cache = EventFileCache(source_path=path)
        scalar_rows: dict[str, dict[str, list[Any]]] = defaultdict(
            _TableRows.empty_scalar_rows
        )
        scalar_dtypes: dict[str, list[np.dtype]] = defaultdict(list)
        histogram_rows: dict[str, dict[str, list[Any]]] = defaultdict(
            _TableRows.empty_histogram_rows
        )
        first_time: float | None = None
        epoch: object = np.nan
        processed_pos = start_pos
        if base_cache is not None:
            first_time, epoch = (
                base_cache.progress.first_time,
                base_cache.progress.epoch,
            )
            cache.binaries = list(base_cache.binaries)
            cache.graphs = list(base_cache.graphs)
            cache.json_entries = list(base_cache.json_entries)
            cache.scalar_mtimes = dict(base_cache.scalar_mtimes)
            cache.histogram_mtimes = dict(base_cache.histogram_mtimes)

        rows = (scalar_rows, scalar_dtypes, histogram_rows)
        with path.open("rb") as handle:
            # Size before reading: records appended during the parse must
            # look unread to the next refresh.
            checked_size, identity = _EventIndexer.file_identity(handle)
            handle.seek(start_pos)
            try:
                for record in iter_tfrecords(handle):
                    processed_pos = record.record_offset + record.total_size
                    try:
                        first_time, epoch = _EventIndexer.index_record(
                            cache, record, rows, (first_time, epoch)
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
                if not ignore_truncated:
                    raise
                if error.corrupt:
                    logger.warning("stopped reading {}: {}", path, error)

        cache.progress = ParseProgress(
            processed_pos, checked_size, identity, first_time, epoch
        )
        cache.scalars = _TableRows.arrays_from_scalar_rows(scalar_rows, scalar_dtypes)
        cache.histograms = _TableRows.arrays_from_histogram_rows(histogram_rows)
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
    def file_identity(handle: BinaryIO) -> tuple[int, tuple[int, int, bytes]]:
        """Return the size and the identity of an open event file."""
        status = os.fstat(handle.fileno())
        head = os.pread(handle.fileno(), IDENTITY_HEAD_BYTES, 0)
        return status.st_size, (status.st_dev, status.st_ino, head)

    @staticmethod
    def index_record(
        cache: EventFileCache,
        record: Record,
        rows: tuple[
            dict[str, dict[str, list[Any]]],
            dict[str, list[np.dtype]],
            dict[str, dict[str, list[Any]]],
        ],
        clock: tuple[float | None, object],
    ) -> tuple[float | None, object]:
        """Index one framed record and return the updated (first_time, epoch)."""
        first_time, epoch = clock
        path = cache.source_path
        event = parse_event(record.payload)
        wall_time = float(event["wall_time"]) if "wall_time" in event else np.nan
        values = event.get("values", [])
        if first_time is None and values and not np.isnan(wall_time):
            first_time = wall_time
        relative_time = (
            np.nan
            if first_time is None or np.isnan(wall_time)
            else wall_time - first_time
        )
        # Proto3 writers leave out step 0, as TensorBoard reads it.
        timing = (event.get("step", 0), wall_time, relative_time)
        _EventIndexer.add_restart(cache, event, rows)
        _EventIndexer.add_graph(cache, path, event, record, wall_time)
        for value in values:
            epoch = _EventIndexer.add_summary(
                cache, path, record, value, rows, (epoch, *timing)
            )
        return first_time, epoch

    @staticmethod
    def add_restart(
        cache: EventFileCache,
        event: dict[str, Any],
        rows: tuple[
            dict[str, dict[str, list[Any]]],
            dict[str, list[np.dtype]],
            dict[str, dict[str, list[Any]]],
        ],
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
        scalar_rows, _, histogram_rows = rows
        counts = {
            (table, tag): len(columns["step"])
            for table, rows_by_tag in (
                ("scalars", scalar_rows),
                ("histograms", histogram_rows),
            )
            for tag, columns in rows_by_tag.items()
        }
        counts[("binaries", "")] = len(cache.binaries)
        counts[("json_entries", "")] = len(cache.json_entries)
        cache.restarts.append(RestartMark("start", event.get("step", 0), counts))

    @staticmethod
    def add_graph(
        cache: EventFileCache,
        path: Path,
        event: dict[str, Any],
        record: Record,
        wall_time: float,
    ) -> None:
        """Index graph payloads found in an event record."""
        if event.get("graph_def") is None:
            return
        cache.graphs.append(
            GraphEntry(
                source_path=path,
                name="graph.pb",
                wall_time=wall_time,
                payload_offset=record.payload_offset,
                payload_size=record.payload_size,
            )
        )

    @staticmethod
    def add_summary(
        cache: EventFileCache,
        path: Path,
        record: Record,
        value: dict[str, Any],
        rows: tuple[
            dict[str, dict[str, list[Any]]],
            dict[str, list[np.dtype]],
            dict[str, dict[str, list[Any]]],
        ],
        timing: tuple[Any, Any, float, float],
    ) -> object:
        """Index one TensorBoard summary value."""
        tag = value.get("tag")
        if not tag:
            return timing[0]
        scalar_rows, scalar_dtypes, histogram_rows = rows
        epoch, step, wall_time, relative_time = timing
        epoch = _EventIndexer.add_scalar(
            cache,
            value,
            scalar_rows,
            scalar_dtypes,
            (epoch, step, wall_time, relative_time),
        )
        _EventIndexer.add_histogram(
            cache, value, histogram_rows, (epoch, step, wall_time, relative_time)
        )
        _EventIndexer.add_binary(cache, path, record, value, (tag, step, wall_time))
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
        for left, right, count in buckets:
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
        path: Path,
        record: Record,
        value: dict[str, Any],
        identity: tuple[str, Any, float],
    ) -> None:
        """Index a binary summary value lazily by source record position."""
        tag, step, wall_time = identity
        parts = _TensorSummary.media_parts(value)
        if parts is not None:
            strings = value["tensor"]["string_val"]
            cache.binaries.extend(
                BinaryEntry(
                    source_path=path,
                    tag=tag,
                    kind=kind,
                    step=step,
                    wall_time=wall_time,
                    payload_offset=record.payload_offset,
                    payload_size=record.payload_size,
                    extension=_TensorSummary.part_extension(
                        value, kind, bytes(strings[part])
                    ),
                    part=part,
                    suffix=suffix,
                )
                for kind, part, suffix in parts
            )
            return
        kind = _Classifier.binary_kind(value)
        if kind is None:
            return
        blob = _Classifier.binary_blob(value)
        if blob is None:
            return
        cache.binaries.append(
            BinaryEntry(
                source_path=path,
                tag=tag,
                kind=kind,
                step=step,
                wall_time=wall_time,
                payload_offset=record.payload_offset,
                payload_size=record.payload_size,
                extension=detect_extension(blob, kind),
            )
        )

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
        cache.json_entries.append(
            JsonEntry(
                tag=tag,
                tab=PLUGIN_TAB.get(plugin_name, "plugins"),
                step=step,
                wall_time=wall_time,
                payload=payload,
            )
        )
