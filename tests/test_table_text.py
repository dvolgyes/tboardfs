"""TSV and CSV tables have the bytes that pandas writes for the same columns."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from tboardfs.tables import export_table

SEPARATORS = {"tsv": "\t", "csv": ","}
EXTREMES = [np.nan, np.inf, -np.inf, -0.0, 5e-324, 1.7976931348623157e308, 1e16, 1e-5]


def _pandas(series: dict[str, np.ndarray], fmt: str) -> bytes:
    handle = BytesIO()
    pd.DataFrame(series).to_csv(handle, sep=SEPARATORS[fmt], index=False, na_rep="nan")
    return handle.getvalue()


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


@pytest.mark.parametrize("fmt", ["tsv", "csv"])
@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "float16", "int64", "int8", "uint64", "bool"]
)
@pytest.mark.parametrize("rows", [0, 1, 50])
def test_number_tables_match_pandas(fmt: str, dtype: str, rows: int) -> None:
    series = _table(rows, dtype)

    assert export_table(series, fmt) == _pandas(series, fmt)


@pytest.mark.parametrize("fmt", ["tsv", "csv"])
@pytest.mark.parametrize(
    "series",
    [
        pytest.param({"name": np.asarray(["a\tb", 'say "x"', "c,d"])}, id="text"),
        pytest.param({"a\tb,c": np.arange(3)}, id="quoted column name"),
        pytest.param({"v": np.asarray([1.5, None], dtype=object)}, id="missing value"),
        pytest.param({"short": np.arange(2), "long": np.arange(4.0)}, id="padded"),
    ],
)
def test_other_tables_match_pandas(fmt: str, series: dict[str, np.ndarray]) -> None:
    filled = {
        "v": np.asarray([1.5, np.nan], dtype=object),
        "short": np.asarray([0.0, 1.0, np.nan, np.nan]),
    }
    expected = {key: filled.get(key, values) for key, values in series.items()}

    assert export_table(series, fmt) == _pandas(expected, fmt)
