"""JSON tables have the bytes that orjson writes for a list of row objects."""

import numpy as np
import orjson
import pytest

from tboardfs.tables import _TableExport, export_table

EXTREMES = [np.nan, np.inf, -np.inf, -0.0, 5e-324, 1.7976931348623157e308, 1e16]


def _rows(series: dict[str, np.ndarray]) -> bytes:
    columns = [_TableExport.json_column(values) for values in series.values()]
    rows = [dict(zip(series, cells, strict=True)) for cells in zip(*columns)]
    return orjson.dumps(
        rows,
        option=orjson.OPT_INDENT_2
        | orjson.OPT_NON_STR_KEYS
        | orjson.OPT_APPEND_NEWLINE
        | orjson.OPT_SERIALIZE_NUMPY,
    )


def _table(rows: int, dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rows)
    values = rng.normal(size=rows) * 10.0 ** rng.integers(-30, 30, size=rows)
    values[: len(EXTREMES)] = EXTREMES[: min(rows, len(EXTREMES))]
    with np.errstate(invalid="ignore", over="ignore"):
        cast = values > 0 if dtype == "bool" else values.astype(dtype)
    return {
        "step": np.arange(rows, dtype=np.int64),
        "wall_time": 1.7e9 + np.arange(rows) * 0.37,
        "value": cast,
    }


@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "float16", "int64", "int8", "uint64", "bool"]
)
@pytest.mark.parametrize("rows", [0, 1, 50])
def test_number_tables_are_row_objects(dtype: str, rows: int) -> None:
    series = _table(rows, dtype)

    assert export_table(series, "json") == _rows(series)


@pytest.mark.parametrize(
    "series",
    [
        pytest.param({}, id="no column"),
        pytest.param({'say "x"\\%s é': np.arange(3)}, id="odd column name"),
        pytest.param({"name": np.asarray(["a", 'q"\\']), "n": np.arange(2)}, id="text"),
        pytest.param({"v": np.asarray([1.5, None], dtype=object)}, id="missing"),
        pytest.param({"pair": np.arange(6.0).reshape(3, 2)}, id="2-D column"),
        pytest.param({"big": np.arange(3, dtype=">f8")}, id="big-endian"),
    ],
)
def test_other_tables_are_row_objects(series: dict[str, np.ndarray]) -> None:
    filled = _TableExport.fill_missing_values(series)

    assert export_table(series, "json") == _rows(filled)
