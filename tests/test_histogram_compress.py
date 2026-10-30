"""Distribution values match TensorBoard's compress_histogram."""

import numpy as np
import pytest

from tboardfs.histogram_compress import compress_histogram

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
