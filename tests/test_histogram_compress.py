"""Distribution values match TensorBoard's compress_histogram."""

import numpy as np
import pytest

from tboardfs.histogram_compress import compress_histogram
from tboardfs.tables import _TableMerge

# Expected values are from tensorboard 2.21.0 compressor.compress_histogram.
CASES = {
    "leading_empty": (
        [[-2.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 3.0], [1.0, 4.0, 1.0]],
        [-2.0, -1.7328, -1.3652, -0.766, 0.0, 0.766, 2.0956, 3.1984, 4.0],
    ),
    "zero_counts": ([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]], [2.0] * 9),
    "zero_width": (
        [[1.0, 1.0, 2.0], [1.0, 3.0, 2.0]],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.766, 2.3652, 2.7328, 3.0],
    ),
    "empty": (np.zeros((0, 3)), [0.0] * 9),
    "empty_inner": (
        [[0.0, 1.0, 1.0], [1.0, 2.0, 0.0], [2.0, 3.0, 1.0]],
        [0.0, 0.1336, 0.3174, 0.617, 2.0, 2.383, 2.6826, 2.8664, 3.0],
    ),
    "inverted_first": (
        [[1.5, 1.0, 1.0], [1.0, 2.0, 1.0]],
        [1.5, 1.4332, 1.3413, 1.1915, 1.5, 1.6915, 1.8413, 1.9332, 2.0],
    ),
    "wide_inner": (
        [[0.0, 3.0, 1.0], [1.0, 2.0, 1.0]],
        [0.0, 0.2672, 0.6348, 1.234, 3.0, 2.617, 2.3174, 2.1336, 2.0],
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_values_match_tensorboard(name: str) -> None:
    """Edge cases give the values that TensorBoard shows."""
    buckets, expected = CASES[name]

    assert compress_histogram(np.asarray(buckets, dtype=np.float64)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (
            np.float32,
            [
                0,
                0.9574667231578267,
                1,
                1,
                1,
                1.6734500842048288,
                2.317590043317038,
                2.7127600182330065,
                3,
            ],
        ),
        (
            np.float64,
            [
                0,
                0.9574666666666666,
                1,
                1,
                1,
                1.6734499999999999,
                2.31759,
                2.7127600000000003,
                3,
            ],
        ),
    ],
)
def test_duplicate_limits_keep_count_precision(
    dtype: type[np.float32] | type[np.float64], expected: list[float]
) -> None:
    """Weights accumulate at the input floating-point precision."""
    buckets = np.array([[0, 1, 0.3], [1, 1, 2], [1, 3, 2]], dtype=dtype)
    assert compress_histogram(buckets) == expected


@pytest.mark.parametrize(
    ("buckets", "first_nonfinite", "nonfinite"),
    [
        ([[0, 1, np.nan], [1, 2, 1]], 0, np.nan),
        ([[0, 1, np.inf], [1, 2, 1]], 0, np.nan),
        ([[0, 1, 1], [1, np.nan, 2]], 4, np.nan),
        ([[0, 1, 1], [1, np.inf, 2]], 4, np.inf),
    ],
)
def test_nonfinite_compression(
    buckets: list[list[float]], first_nonfinite: int, nonfinite: float
) -> None:
    """Nonfinite counts and limits retain their established interpolation."""
    with np.errstate(invalid="ignore"):
        values = compress_histogram(np.array(buckets))
    expected = [0.0, 0.2004, 0.47609999999999997, 0.9255][:first_nonfinite]
    np.testing.assert_array_equal(values, expected + [nonfinite] * (9 - len(expected)))


def test_distribution_repeats_each_observations_metadata() -> None:
    """A repeated step keeps both observations and exact metadata dtypes."""
    step = 2**63 + 5
    series = {
        "step": np.array([step, step, step], dtype=">u8"),
        "wall_time": np.array([20.5, 20.5, 45.5], dtype=np.float32),
        "relative_time": np.array([0, 0, 25], dtype=np.int16),
        "bucket_left": np.array([0.0, 1.0, 4.0]),
        "bucket_right": np.array([1.0, 2.0, 7.0]),
        "count": np.array([0.0, 0.0, 0.0]),
        "_start": np.array([True, False, True]),
    }

    result = _TableMerge.distribution_from_histogram(series)
    percentiles = [0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100]
    expected = {
        "step": np.array([step] * 18, dtype=np.uint64),
        "epoch": np.full(18, np.nan),
        "wall_time": np.array([20.5] * 9 + [45.5] * 9, dtype=np.float32),
        "relative_time": np.array([0] * 9 + [25] * 9, dtype=np.int16),
        "percentile": np.array(percentiles * 2),
        "value": np.array([2.0] * 9 + [7.0] * 9),
    }
    assert list(result) == list(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(result[name], values, strict=True)
