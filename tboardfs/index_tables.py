"""Fixed-schema scalar and histogram columns in the persisted index."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from tboardfs.index_wire import _IndexReader, _IndexWriter

_COMMON = MappingProxyType(
    {
        "_inherits_epoch": "b1",
        "epoch": "f8",
        "wall_time": "f8",
        "relative_time": "f8",
    }
)
_SCALAR = MappingProxyType({**_COMMON, "step": "i8", "value": None})
_HISTOGRAM = MappingProxyType(
    {
        **_COMMON,
        "step": None,
        "_start": "b1",
        "bucket_left": "f8",
        "bucket_right": "f8",
        "count": "f8",
    }
)
_DTYPES = MappingProxyType(
    {
        dtype.str: dtype
        for dtype in map(
            np.dtype,
            (
                "?",
                "i1",
                "u1",
                "<i2",
                "<u2",
                "<i4",
                "<u4",
                "<i8",
                "<u8",
                "<f2",
                "<f4",
                "<f8",
            ),
        )
    }
)

type Tables = dict[str, dict[str, np.ndarray]]


class TableSection:
    """Encode and validate table columns."""

    @staticmethod
    def _dtype(name: object, column: str, required: str | None) -> np.dtype:
        """Validate and resolve one wire column type."""
        if not isinstance(name, str) or name not in _DTYPES:
            raise ValueError("unsupported index column dtype")
        dtype = _DTYPES[name]
        kind_size = name[1:]
        if required is not None and kind_size != required:
            raise ValueError("wrong index column dtype")
        if column == "step" and kind_size not in ("i8", "f8"):
            raise ValueError("wrong index step dtype")
        return dtype

    @staticmethod
    def _write_column(
        writer: _IndexWriter, values: np.ndarray, dtype: np.dtype
    ) -> None:
        """Write contiguous bytes in the requested wire type."""
        writer.value(dtype.str)
        writer.ensure(values.size * dtype.itemsize)
        contiguous = np.ascontiguousarray(values, dtype=dtype)
        writer.write(memoryview(contiguous).cast("B"))

    @staticmethod
    def _write_columns(
        writer: _IndexWriter,
        columns: dict[str, np.ndarray],
        schema: Mapping[str, str | None],
    ) -> None:
        """Preserve column insertion order and enforce the schema."""
        if columns.keys() != schema.keys():
            raise ValueError("invalid index table columns")
        rows = columns["step"].size
        writer.count(rows)
        for column, values in columns.items():
            if values.shape != (rows,):
                raise ValueError("invalid index column shape")
            dtype = TableSection._dtype(
                values.dtype.newbyteorder("<").str, column, schema[column]
            )
            writer.value(column)
            TableSection._write_column(writer, values, dtype)

    @staticmethod
    def write(
        writer: _IndexWriter,
        tables: Tables,
        mtimes: dict[str, float],
        *,
        histogram: bool,
    ) -> None:
        """Write exact column bytes without a NumPy archive or object payload."""
        if tables.keys() != mtimes.keys():
            raise ValueError("index table modification times differ from tags")
        schema = _HISTOGRAM if histogram else _SCALAR
        writer.count(len(tables))
        for tag, columns in tables.items():
            writer.value(tag)
            writer.value(mtimes[tag])
            TableSection._write_columns(writer, columns, schema)

    @staticmethod
    def _read_columns(
        reader: _IndexReader, schema: Mapping[str, str | None], max_rows: int
    ) -> dict[str, np.ndarray]:
        """Check column evidence against the expected table schema."""
        rows = reader.count()
        if rows > max_rows or (not rows and schema is _SCALAR):
            raise ValueError("invalid index table row count")
        columns: dict[str, np.ndarray] = {}
        for _ in schema:
            column = reader.value()
            if not isinstance(column, str) or column not in schema or column in columns:
                raise ValueError("invalid index table column name")
            dtype = TableSection._dtype(reader.value(), column, schema[column])
            data = reader.take(rows * dtype.itemsize)
            columns[column] = np.frombuffer(data, dtype=dtype).astype(
                dtype.newbyteorder("="), copy=True
            )
        return columns

    @staticmethod
    def read(
        reader: _IndexReader, source_size: int, *, histogram: bool
    ) -> tuple[Tables, dict[str, float]]:
        """Bound expanded histogram rows by actual bytes and scalar rows by source size."""
        schema = _HISTOGRAM if histogram else _SCALAR
        max_rows = len(reader.remaining) if histogram else source_size
        tables: Tables = {}
        mtimes: dict[str, float] = {}
        count = reader.items(width=3)
        if count > source_size:
            raise ValueError("too many index table tags for source size")
        for _ in range(count):
            tag = reader.value()
            mtime = reader.value()
            if not isinstance(tag, str) or tag in tables or type(mtime) is not float:
                raise ValueError("invalid index table tag or time")
            tables[tag] = TableSection._read_columns(reader, schema, max_rows)
            mtimes[tag] = mtime
        return tables, mtimes
