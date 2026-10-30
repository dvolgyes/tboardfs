"""TSV and CSV tables hold the values that pandas writes for the same columns.

A float cell can be spelled in another way, such as 1e-5 for 1e-05, but it
reads back as the same value. NaN, infinities and other cells are as pandas
writes them.
"""

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


def _cells(text: bytes, fmt: str) -> list[list[str]]:
    return [line.split(SEPARATORS[fmt]) for line in text.decode().splitlines()]


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

    ours = _cells(export_table(series, fmt), fmt)
    theirs = _cells(_pandas(series, fmt), fmt)

    assert ours[0] == theirs[0]
    assert len(ours) == len(theirs)
    for index, values in enumerate(series.values()):
        column = [row[index] for row in ours[1:]]
        expected = [row[index] for row in theirs[1:]]
        if values.dtype.kind != "f":
            assert column == expected
            continue
        read = np.asarray(column).astype(values.dtype)
        assert np.array_equal(read, values, equal_nan=True)
        assert np.array_equal(np.signbit(read), np.signbit(values))
        finite = np.isfinite(values)
        assert [c for c, f in zip(column, finite) if not f] == [
            c for c, f in zip(expected, finite) if not f
        ]


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
