import base64
from collections import defaultdict
from io import BytesIO
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from tboardfs.constants import SCALAR_COLUMNS
from tboardfs.decode import _Decode
from tboardfs.histogram_compress import NORMAL_HISTOGRAM_BPS, compress_histogram


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
        separators = {"tsv": "\t", "csv": ","}
        if fmt not in separators:
            raise ValueError(f"unsupported tabular export format: {fmt}")
        frame.to_csv(handle, sep=separators[fmt], index=False, na_rep="nan")
        return handle.getvalue()

    @staticmethod
    def json(series: dict[str, np.ndarray]) -> bytes:
        """Render a table as strict JSON rows."""
        rows = [
            {column: _TableExport.json_safe(series[column][index]) for column in series}
            for index in range(_TableExport.series_len(series))
        ]
        return (json.dumps(rows, allow_nan=False, indent=2) + "\n").encode()

    @staticmethod
    def dataframe(series: dict[str, np.ndarray]) -> pd.DataFrame:
        """Build a pandas dataframe from column arrays."""
        return pd.DataFrame({column: series[column] for column in series})

    @staticmethod
    def json_document(data: object) -> bytes:
        """Render any virtual JSON document as strict JSON.

        NaN and infinities use the protobuf JSON spelling ("NaN", "Infinity",
        "-Infinity"). Bytes that are UTF-8 become text; other bytes become
        ``{"base64": ...}``.
        """
        value = _TableExport.json_value(data, "NaN")
        return (json.dumps(value, allow_nan=False, indent=2) + "\n").encode()

    @staticmethod
    def json_value(value: object, nan: str | None) -> object:
        """Convert a value and its items to strict JSON types.

        Args:
            value: NumPy, bytes, container or plain value
            nan: replacement for NaN; tables use null for a missing value

        A complex number becomes its [real, imag] pair.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex):
            value = [value.real, value.imag]
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                return nan
            return "Infinity" if value > 0 else "-Infinity"
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return {"base64": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Mapping):
            return {
                key: _TableExport.json_value(item, nan) for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [_TableExport.json_value(item, nan) for item in value]
        return value

    @staticmethod
    def json_safe(value: object) -> object:
        """Return a table cell as JSON; NaN, a missing value, becomes null."""
        return _TableExport.json_value(value, None)

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
                "epoch": np.asarray(rows["epoch"], dtype=np.float64),
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
            "epoch": [],
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
    def append_series(
        base: Mapping[str, Mapping[str, np.ndarray]],
        new: Mapping[str, Mapping[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Append rows of ``new`` to ``base`` with the dtypes of one parse.

        A parse keeps a column dtype only when all rows agree, else float64.
        """
        merged = {tag: dict(series) for tag, series in base.items()}
        for tag, series in new.items():
            old = merged.setdefault(tag, {})
            for column, values in series.items():
                if column not in old:
                    old[column] = values
                    continue
                dtype = (
                    old[column].dtype
                    if old[column].dtype == values.dtype
                    else np.dtype("float64")
                )
                old[column] = np.concatenate((old[column], values)).astype(
                    dtype, copy=False
                )
        return merged

    @staticmethod
    def step_epochs(
        tables: Mapping[str, Mapping[str, np.ndarray]],
        epochs: Mapping[str, np.ndarray] | None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Give each row the epoch scalar written at its step, when there is one.

        Loggers such as PyTorch Lightning write a metric and then the epoch
        scalar at the same step, so the epoch carried from the events before
        the metric is one epoch late. Rows at a step without an epoch scalar
        keep the carried epoch. The last epoch written at a step wins.
        """
        if not epochs or not len(epochs["step"]):
            return {tag: dict(series) for tag, series in tables.items()}
        backward = np.asarray(epochs["step"], dtype=np.float64)[::-1]
        known, last = np.unique(backward, return_index=True)
        values = np.asarray(epochs["value"], dtype=np.float64)[::-1][last]
        aligned = {}
        for tag, series in tables.items():
            steps = np.asarray(series["step"], dtype=np.float64)
            index = np.minimum(np.searchsorted(known, steps), len(known) - 1)
            aligned[tag] = {
                **series,
                "epoch": np.where(
                    known[index] == steps, values[index], series["epoch"]
                ),
            }
        return aligned

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
        """Derive TensorBoard distribution rows from histogram buckets.

        Each histogram event is one contiguous run of bucket rows with the same
        step and wall time, so repeated steps stay separate groups. The
        percentile column holds the basis points in percent.
        """
        out = defaultdict(list)
        buckets = np.column_stack(
            [
                np.asarray(series.get(column, []), dtype=np.float64)
                for column in ("bucket_left", "bucket_right", "count")
            ]
        )
        epochs = series.get("epoch", np.full(len(buckets), np.nan))
        for start, stop in _TableMerge.event_bounds(series):
            values = compress_histogram(buckets[start:stop])
            for point, value in zip(NORMAL_HISTOGRAM_BPS, values, strict=True):
                out["step"].append(series["step"][start])
                out["epoch"].append(epochs[start])
                out["wall_time"].append(series["wall_time"][start])
                out["relative_time"].append(series["relative_time"][start])
                out["percentile"].append(point / 100)
                out["value"].append(value)
        return {key: np.asarray(value) for key, value in out.items()}

    @staticmethod
    def event_bounds(series: dict[str, np.ndarray]) -> list[tuple[int, int]]:
        """Return [start, stop) row ranges with a constant step and wall time."""
        size = len(series.get("step", []))
        if size == 0:
            return []
        changed = np.zeros(size, dtype=bool)
        changed[0] = True
        for column in ("step", "wall_time"):
            values = np.asarray(series[column], dtype=np.float64)
            changed[1:] |= ~_TableMerge.nan_equal(values[:-1], values[1:])
        starts = np.flatnonzero(changed).tolist()
        return list(zip(starts, [*starts[1:], size], strict=True))

    @staticmethod
    def nan_equal(left: np.ndarray, right: object) -> np.ndarray:
        """Compare elementwise and treat NaN as equal to NaN."""
        values = np.asarray(left, dtype=np.float64)
        other = np.asarray(right, dtype=np.float64)
        return (values == other) | (np.isnan(values) & np.isnan(other))

    @staticmethod
    def unique_steps(series: dict[str, np.ndarray]) -> list[Any]:
        """Return sorted distinct steps; all NaN steps collapse to one entry."""
        steps: list[Any] = np.unique(series["step"]).tolist()
        return steps

    @staticmethod
    def series_for_step(
        series: dict[str, np.ndarray], step: object
    ) -> dict[str, np.ndarray]:
        """Return a column-oriented table for one TensorBoard step."""
        mask = _TableMerge.nan_equal(series["step"], step)
        return {column: values[mask] for column, values in series.items()}
