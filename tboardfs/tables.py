import base64
from collections import defaultdict
from io import BytesIO
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import orjson
import pandas as pd

from tboardfs.constants import SCALAR_COLUMNS
from tboardfs.decode import _Decode
from tboardfs.histogram_compress import NORMAL_HISTOGRAM_BPS, compress_histogram

# Two-space indents, keys such as numbers as text, a final newline, and
# NumPy arrays that json_value lets through.
_JSON_OPTIONS = (
    orjson.OPT_INDENT_2
    | orjson.OPT_NON_STR_KEYS
    | orjson.OPT_APPEND_NEWLINE
    | orjson.OPT_SERIALIZE_NUMPY
)
# Native number types that orjson writes as the same value, much faster than
# NumPy text. orjson writes float16 with float32 digits, and booleans as JSON.
_ORJSON_NUMBERS = frozenset(
    np.dtype(name)
    for name in (
        "float64",
        "float32",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    )
)
# The native types that orjson writes as JSON lists of their values.
_JSON_NUMBERS = _ORJSON_NUMBERS | {np.dtype("float16"), np.dtype("bool")}


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

        separators = {"tsv": "\t", "csv": ","}
        if fmt not in separators:
            raise ValueError(f"unsupported tabular export format: {fmt}")
        text = _TableExport.number_text(series, separators[fmt])
        if text is not None:
            return text
        frame = _TableExport.dataframe(series)
        handle = BytesIO()
        frame.to_csv(handle, sep=separators[fmt], index=False, na_rep="nan")
        return handle.getvalue()

    @staticmethod
    def number_text(series: dict[str, np.ndarray], sep: str) -> bytes | None:
        """Return a table of number columns as text, or None.

        The table has the values that pandas writes, without the cost of a
        dataframe. A table with other columns, or with a column name that
        pandas quotes, gives None.
        """
        if not series or any(
            values.ndim != 1 or values.dtype.kind not in "biuf"
            for values in series.values()
        ):
            return None
        if any(set(name) & {sep, '"', "\n", "\r"} for name in series):
            return None
        columns = [_TableExport.number_cells(values) for values in series.values()]
        row = sep.join(["%s"] * len(columns))
        lines = [sep.join(series), *(row % cells for cells in zip(*columns))]
        return ("\n".join(lines) + "\n").encode()

    @staticmethod
    def json(series: dict[str, np.ndarray]) -> bytes:
        """Render a table as strict JSON rows."""
        text = _TableExport.number_json(series)
        if text is not None:
            return text
        columns = [_TableExport.json_column(values) for values in series.values()]
        rows = [dict(zip(series, cells, strict=True)) for cells in zip(*columns)]
        return orjson.dumps(rows, option=_JSON_OPTIONS)

    @staticmethod
    def number_json(series: dict[str, np.ndarray]) -> bytes | None:
        """Return a table of number columns as JSON rows, or None.

        The bytes are those that orjson writes for a list of row objects, but
        the text is built from the text of each column: an object for each
        row costs more than the rest of the work. A table with other columns
        gives None.
        """
        if any(
            values.ndim != 1 or values.dtype not in _JSON_NUMBERS
            for values in series.values()
        ):
            return None
        columns = [_TableExport.json_cells(values) for values in series.values()]
        rows = min(map(len, columns), default=0)
        if not rows:
            return b"[]\n"
        # Each row is "  {", then '    "name": cell' for each column, then
        # "  }"; slice assignment puts the constant text beside the cells.
        width = 2 * len(columns)
        parts = [""] * (rows * width)
        for index, (name, cells) in enumerate(zip(series, columns, strict=True)):
            start = "  {\n    " if index == 0 else ",\n    "
            parts[2 * index :: width] = [
                start + orjson.dumps(name).decode() + ": "
            ] * rows
            parts[2 * index + 1 :: width] = cells[:rows]
        parts[width - 1 :: width] = [cell + "\n  },\n" for cell in columns[-1][:rows]]
        parts[0] = "[\n" + parts[0]
        parts[-1] = columns[-1][rows - 1] + "\n  }\n]\n"
        return "".join(parts).encode()

    @staticmethod
    def json_cells(values: np.ndarray) -> list[str]:
        """Return the JSON text of each number in a column, as json_column gives it.

        A float is widened to float64, so its digits are those of the list.
        NaN, a missing value, stays null; an infinity becomes its text.
        """
        if values.dtype.kind != "f":
            return _TableExport.orjson_cells(values)
        cells = _TableExport.orjson_cells(values.astype(np.float64, copy=False))
        for index in np.flatnonzero(np.isinf(values)).tolist():
            cells[index] = '"Infinity"' if values[index] > 0 else '"-Infinity"'
        return cells

    @staticmethod
    def orjson_cells(values: np.ndarray) -> list[str]:
        """Return the orjson text of each value of a 1-D number array.

        orjson writes NaN and the infinities as null.
        """
        if not len(values):
            return []
        text = orjson.dumps(
            np.ascontiguousarray(values), option=orjson.OPT_SERIALIZE_NUMPY
        )
        return text[1:-1].decode().split(",")

    @staticmethod
    def json_column(values: np.ndarray) -> list[object]:
        """Return the JSON cells of one column; real numbers convert at once."""
        if values.ndim != 1 or values.dtype.kind not in "biuf":
            return [_TableExport.json_safe(value) for value in values]
        cells: list[object] = values.tolist()
        if values.dtype.kind == "f":
            for index in np.flatnonzero(~np.isfinite(values)):
                cells[index] = _TableExport.json_safe(cells[index])
        return cells

    @staticmethod
    def number_cells(values: np.ndarray) -> list[str]:
        """Return the text of each number in a column.

        A float can be spelled in another way than pandas spells it, such as
        1e-5 for 1e-05, but it reads back as the same value. NaN and the
        infinities are spelled as pandas spells them.
        """
        if values.dtype not in _ORJSON_NUMBERS:
            return list(values.astype(str))
        cells = _TableExport.orjson_cells(values)
        for index in np.flatnonzero(~np.isfinite(values)).tolist():
            cells[index] = str(values[index])
        return cells

    @staticmethod
    def dataframe(series: dict[str, np.ndarray]) -> pd.DataFrame:
        """Build a pandas dataframe from column arrays."""
        return pd.DataFrame({column: series[column] for column in series})

    @staticmethod
    def json_document(data: object) -> bytes:
        """Render any virtual JSON document as strict JSON.

        NaN and infinities use the protobuf JSON spelling ("NaN", "Infinity",
        "-Infinity"). Bytes that are UTF-8 become text; other bytes become
        ``{"base64": ...}``. Text is UTF-8.
        """
        value = _TableExport.json_value(data, "NaN")
        return orjson.dumps(value, option=_JSON_OPTIONS)

    @staticmethod
    def json_array(values: np.ndarray) -> np.ndarray | None:
        """Return an array that orjson writes as values.tolist(), or None.

        orjson writes an array much faster than the list of its values. A
        float is widened to float64, so its digits are those of the list. An
        array with NaN or an infinity gives None, as it needs their spelling.
        """
        if values.ndim == 0 or values.dtype not in _JSON_NUMBERS:
            return None
        if values.dtype.kind == "f":
            if not np.isfinite(values).all():
                return None
            values = values.astype(np.float64, copy=False)
        return np.ascontiguousarray(values)

    @staticmethod
    def json_value(value: object, nan: str | None) -> object:
        """Convert a value and its items to strict JSON types.

        Args:
            value: NumPy, bytes, container or plain value
            nan: replacement for NaN; tables use null for a missing value

        A complex number becomes its [real, imag] pair.
        """
        if isinstance(value, np.ndarray):
            array = _TableExport.json_array(value)
            if array is not None:
                return array
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex):
            value = [value.real, value.imag]
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
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
    def split_steps(
        series: dict[str, np.ndarray],
    ) -> list[tuple[Any, dict[str, np.ndarray]]]:
        """Return each distinct step in order, with its rows in their order.

        All NaN steps are one step.
        """
        steps, inverse = np.unique(series["step"], return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=len(steps))
        ends = np.cumsum(counts)
        return [
            (
                step,
                {
                    column: values[order[end - count : end]]
                    for column, values in series.items()
                },
            )
            for step, count, end in zip(steps.tolist(), counts, ends, strict=True)
        ]
