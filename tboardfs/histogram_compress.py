"""Compress a histogram to the distribution values that TensorBoard shows."""

import numpy as np

# Basis points (1/100 of a percent): the mean and +-1, 2 and 3 sigma of a normal.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)


def compress_histogram(buckets: np.ndarray) -> list[float]:
    """Return the value at each of NORMAL_HISTOGRAM_BPS.

    A port of compress_histogram in tensorboard/plugins/distribution/compressor.py
    (TensorBoard 2.21). The value interpolates linearly inside the bucket that
    holds the basis point, clamped to the first left and last right edge.

    Args:
        buckets: rows of (left, right, count).
    """
    if not len(buckets):
        return [0.0] * len(NORMAL_HISTOGRAM_BPS)
    edges = (float(buckets[0, 0]), float(buckets[-1, 1]))
    counts = buckets[:, 2]
    weights = np.cumsum(counts * NORMAL_HISTOGRAM_BPS[-1] / (counts.sum() or 1.0))
    values: list[float] = []
    for point in NORMAL_HISTOGRAM_BPS:
        value = _value_at(point, weights, buckets[:, 1], edges)
        if value is None:
            break
        values.append(value)
    return values + [edges[1]] * (len(NORMAL_HISTOGRAM_BPS) - len(values))


def _value_at(
    point: int, weights: np.ndarray, rights: np.ndarray, edges: tuple[float, float]
) -> float | None:
    """Interpolate one basis point, or None past the last non-empty bucket."""
    lowest, highest = edges
    # The first bucket with more weight than the point is never empty, so
    # TensorBoard's loop that skips empty buckets has no effect here.
    index = int(np.searchsorted(weights, point, side="right"))
    if index == len(weights):
        return None
    before = float(weights[index - 1]) if index else 0.0
    left = max(float(rights[index - 1]), lowest) if index and before else lowest
    right = min(float(rights[index]), highest)
    return left + (point - before) / (float(weights[index]) - before) * (right - left)
