from pathlib import Path
from typing import Any

import numpy as np

from tboardfs.classify import _Classifier, detect_extension
from tboardfs.constants import PLUGIN_TAB
from tboardfs.model import BinaryEntry, EventFileCache, GraphEntry, JsonEntry, Record
from tboardfs.summary import parse_event
from tboardfs.tables import (
    _TableRows,
)
from tboardfs.tfrecord import iter_tfrecords


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
        """Build an in-memory index for one TensorBoard event file."""
        cache = EventFileCache(source_path=path)
        scalar_rows, scalar_dtypes = _TableRows.scalar_rows_from_cache(base_cache)
        histogram_rows = _TableRows.histogram_rows_from_cache(base_cache)
        first_time = _TableRows.first_wall_time(scalar_rows)
        epoch = _TableRows.current_epoch(scalar_rows)
        processed_pos = start_pos
        if base_cache is not None:
            cache.binaries = list(base_cache.binaries)
            cache.graphs = list(base_cache.graphs)
            cache.json_entries = list(base_cache.json_entries)
            cache.scalar_mtimes = dict(base_cache.scalar_mtimes)
            cache.histogram_mtimes = dict(base_cache.histogram_mtimes)

        with path.open("rb") as handle:
            handle.seek(start_pos)
            try:
                for record in iter_tfrecords(handle):
                    processed_pos = record.record_offset + record.total_size
                    event = parse_event(record.payload)
                    wall_time = (
                        float(event["wall_time"]) if "wall_time" in event else np.nan
                    )
                    values = event.get("values", [])
                    if first_time is None and values and not np.isnan(wall_time):
                        first_time = wall_time
                    relative_time = (
                        np.nan
                        if first_time is None or np.isnan(wall_time)
                        else wall_time - first_time
                    )
                    timing = (event.get("step", np.nan), wall_time, relative_time)
                    _EventIndexer.add_graph(cache, path, event, record, wall_time)
                    for value in values:
                        epoch = _EventIndexer.add_summary(
                            cache,
                            path,
                            record,
                            value,
                            (scalar_rows, scalar_dtypes, histogram_rows),
                            (epoch, *timing),
                        )
            except ValueError:
                if not ignore_truncated:
                    raise

        cache.processed_pos = processed_pos
        cache.last_checked_size = path.stat().st_size
        cache.scalars = _TableRows.arrays_from_scalar_rows(scalar_rows, scalar_dtypes)
        cache.histograms = _TableRows.arrays_from_histogram_rows(histogram_rows)
        return cache

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
            cache, value, histogram_rows, (step, wall_time, relative_time)
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
        timing: tuple[Any, float, float],
    ) -> None:
        """Append histogram bucket rows when present."""
        histogram = value.get("histo")
        if histogram is None:
            return
        tag = value["tag"]
        step, wall_time, relative_time = timing
        rows = histogram_rows[tag]
        limits = histogram.get("bucket_limit", [])
        counts = histogram.get("bucket", [])
        left = float(histogram.get("min", float("-inf")))
        for limit, count in zip(limits, counts, strict=False):
            right = float(limit)
            rows["step"].append(step)
            rows["wall_time"].append(wall_time)
            rows["relative_time"].append(relative_time)
            rows["bucket_left"].append(left)
            rows["bucket_right"].append(right)
            rows["count"].append(float(count))
            left = right
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
        kind = _Classifier.binary_kind(value)
        if kind is None:
            return
        tag, step, wall_time = identity
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
                record_size=record.total_size,
                extension=detect_extension(blob, kind),
                blob_size=len(blob),
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
        if payload is None:
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
