from collections import defaultdict
from io import BytesIO
import inspect
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from tboardfs.constants import SCALAR_COLUMNS
from tboardfs.decode import _Decode
from tboardfs.model import EventFileCache


def export_table(series: dict[str, np.ndarray], fmt: str) -> bytes:
    """Render a scalar or histogram table in a requested file format."""
    return _TableExport.render(series, fmt)


class _TableExport:
    """Render column-oriented arrays to supported table formats."""

    @staticmethod
    def render(series: dict[str, np.ndarray], fmt: str) -> bytes:
        """Render a scalar or histogram table in a requested file format."""
        fmt = fmt.lower()
        series = _TableExport.fill_missing_values(series)
        if fmt == "json":
            return _TableExport.json(series)
        if fmt == "npz":
            handle = BytesIO()
            np.savez(handle, **series)
            return handle.getvalue()

        frame = _TableExport.dataframe(series)
        handle = BytesIO()
        if fmt == "tsv":
            frame.to_csv(handle, sep="\t", index=False, na_rep="nan")
        else:
            writer = getattr(frame, f"to_{fmt}", None)
            if writer is None:
                raise ValueError(f"unsupported tabular export format: {fmt}")
            kwargs: dict[str, Any] = {"index": False}
            if "na_rep" in inspect.signature(writer).parameters:
                kwargs["na_rep"] = "nan"
            writer(handle, **kwargs)
        return handle.getvalue()

    @staticmethod
    def json(series: dict[str, np.ndarray]) -> bytes:
        """Render a table as strict JSON rows."""
        rows = [
            {column: _TableExport.json_safe(series[column][index]) for column in series}
            for index in range(_TableExport.series_len(series))
        ]
        return (json.dumps(rows, allow_nan=False) + "\n").encode()

    @staticmethod
    def dataframe(series: dict[str, np.ndarray]) -> pd.DataFrame:
        """Build a pandas dataframe from column arrays."""
        return pd.DataFrame({column: series[column] for column in series})

    @staticmethod
    def json_default(value: object) -> object:
        """Convert NumPy and bytes objects to JSON-compatible values."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def json_safe(value: object) -> object:
        """Return a JSON value, converting missing numeric values to null."""
        converted = _TableExport.json_default(value)
        if isinstance(converted, float) and not np.isfinite(converted):
            return None
        return converted

    @staticmethod
    def fill_missing_values(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Pad shorter columns and missing object values with NaN for export."""
        length = max((len(values) for values in series.values()), default=0)
        return {
            column: _TableExport.fill_missing_column(values, length)
            for column, values in series.items()
        }

    @staticmethod
    def fill_missing_column(values: np.ndarray, length: int) -> np.ndarray:
        """Return one export column with explicit NaN placeholders."""
        if len(values) == length and not _TableExport.contains_none(values):
            return values
        if np.issubdtype(values.dtype, np.number):
            filled = np.full(length, np.nan, dtype=np.float64)
            filled[: len(values)] = values
            return filled
        filled = np.full(length, np.nan, dtype=object)
        filled[: len(values)] = values.astype(object, copy=False)
        return np.asarray(
            [np.nan if value is None else value for value in filled], dtype=object
        )

    @staticmethod
    def contains_none(values: np.ndarray) -> bool:
        """Return true when an array contains Python None values."""
        if values.dtype != object:
            return False
        return any(value is None for value in values)

    @staticmethod
    def series_len(series: dict[str, np.ndarray]) -> int:
        """Return the number of rows in a column-oriented table."""
        if not series:
            return 0
        return len(next(iter(series.values())))


class _TableRows:
    """Convert mutable parse rows to NumPy-backed table columns."""

    @staticmethod
    def scalar_rows_from_cache(
        cache: EventFileCache | None,
    ) -> tuple[dict[str, dict[str, list[Any]]], dict[str, list[np.dtype]]]:
        """Rebuild mutable scalar rows from an existing cache."""
        rows: dict[str, dict[str, list[Any]]] = defaultdict(
            _TableRows.empty_scalar_rows
        )
        dtypes: dict[str, list[np.dtype]] = defaultdict(list)
        if cache is None:
            return rows, dtypes
        for tag, series in cache.scalars.items():
            rows[tag] = {column: list(series[column]) for column in SCALAR_COLUMNS}
            dtypes[tag] = [series["value"].dtype] * len(series["value"])
        return rows, dtypes

    @staticmethod
    def histogram_rows_from_cache(
        cache: EventFileCache | None,
    ) -> dict[str, dict[str, list[Any]]]:
        """Rebuild mutable histogram rows from an existing cache."""
        rows: dict[str, dict[str, list[Any]]] = defaultdict(
            _TableRows.empty_histogram_rows
        )
        if cache is None:
            return rows
        for tag, series in cache.histograms.items():
            rows[tag] = {column: list(series[column]) for column in series}
        return rows

    @staticmethod
    def first_wall_time(scalar_rows: dict[str, dict[str, list[Any]]]) -> float | None:
        """Return the earliest valid scalar wall time in mutable row storage."""
        wall_times = [
            float(value)
            for rows in scalar_rows.values()
            for value in rows["wall_time"]
            if not np.isnan(float(value))
        ]
        return min(wall_times) if wall_times else None

    @staticmethod
    def current_epoch(scalar_rows: dict[str, dict[str, list[Any]]]) -> object:
        """Return the most recent epoch scalar value from mutable row storage."""
        epoch_rows = scalar_rows.get("epoch")
        if epoch_rows and epoch_rows["value"]:
            return epoch_rows["value"][-1]
        return np.nan

    @staticmethod
    def arrays_from_scalar_rows(
        scalar_rows: dict[str, dict[str, list[Any]]],
        scalar_dtypes: dict[str, list[np.dtype]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert mutable scalar rows to NumPy column arrays."""
        return {
            tag: {
                "epoch": np.asarray(rows["epoch"], dtype=np.float64),
                "step": np.asarray(
                    rows["step"],
                    dtype=np.int64 if _Decode.all_ints(rows["step"]) else np.float64,
                ),
                "wall_time": np.asarray(rows["wall_time"], dtype=np.float64),
                "relative_time": np.asarray(rows["relative_time"], dtype=np.float64),
                "value": np.asarray(
                    rows["value"],
                    dtype=_Decode.scalar_array_dtype(scalar_dtypes.get(tag, [])),
                ),
            }
            for tag, rows in scalar_rows.items()
        }

    @staticmethod
    def arrays_from_histogram_rows(
        rows_by_tag: dict[str, dict[str, list[Any]]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert mutable histogram rows to NumPy column arrays."""
        return {
            tag: {
                "step": np.asarray(
                    rows["step"],
                    dtype=np.int64 if _Decode.all_ints(rows["step"]) else np.float64,
                ),
                "wall_time": np.asarray(rows["wall_time"], dtype=np.float64),
                "relative_time": np.asarray(rows["relative_time"], dtype=np.float64),
                "bucket_left": np.asarray(rows["bucket_left"], dtype=np.float64),
                "bucket_right": np.asarray(rows["bucket_right"], dtype=np.float64),
                "count": np.asarray(rows["count"], dtype=np.float64),
            }
            for tag, rows in rows_by_tag.items()
        }

    @staticmethod
    def empty_scalar_rows() -> dict[str, list[Any]]:
        """Create mutable storage for scalar table rows."""
        return {column: [] for column in SCALAR_COLUMNS}

    @staticmethod
    def empty_histogram_rows() -> dict[str, list[Any]]:
        """Create mutable storage for histogram table rows."""
        return {
            "step": [],
            "wall_time": [],
            "relative_time": [],
            "bucket_left": [],
            "bucket_right": [],
            "count": [],
        }


class _TableMerge:
    """Merge per-file table columns and derive distribution rows."""

    @staticmethod
    def merge_series(
        items: Iterable[Mapping[str, Mapping[str, np.ndarray]]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Merge per-file series in caller-provided order."""
        merged: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for item in items:
            for tag, series in item.items():
                for column, values in series.items():
                    merged[tag][column].append(values)
        return {
            tag: {
                column: np.concatenate(values) if values else np.asarray([])
                for column, values in columns.items()
            }
            for tag, columns in merged.items()
        }

    @staticmethod
    def merge_mtimes(items: Iterable[Mapping[str, float]]) -> dict[str, float]:
        """Merge per-file mtimes by keeping the most recent value."""
        merged: dict[str, float] = {}
        for item in items:
            for tag, mtime in item.items():
                merged[tag] = max(merged.get(tag, mtime), mtime)
        return merged

    @staticmethod
    def distribution_from_histogram(
        series: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Derive TensorBoard-style percentile rows from histogram buckets."""
        rows = _TableMerge.rows_from_series(series)
        out = defaultdict(list)
        percentiles = (0.0, 25.0, 50.0, 75.0, 100.0)
        by_step: dict[tuple[Any, float, float], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in rows:
            by_step[(row["step"], row["wall_time"], row["relative_time"])].append(row)
        for (step, wall_time, relative_time), buckets in by_step.items():
            total = sum(float(row["count"]) for row in buckets)
            cumulative = 0.0
            targets = {p: total * p / 100.0 for p in percentiles}
            found: dict[float, float] = {}
            for row in buckets:
                cumulative += float(row["count"])
                for percentile, target in targets.items():
                    if percentile not in found and cumulative >= target:
                        found[percentile] = float(row["bucket_right"])
            for percentile in percentiles:
                out["step"].append(step)
                out["wall_time"].append(wall_time)
                out["relative_time"].append(relative_time)
                out["percentile"].append(percentile)
                out["value"].append(found.get(percentile, float("nan")))
        return {key: np.asarray(value) for key, value in out.items()}

    @staticmethod
    def rows_from_series(series: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        """Return row dictionaries for a column-oriented table."""
        return [
            {column: series[column][index] for column in series}
            for index in range(_TableExport.series_len(series))
        ]
