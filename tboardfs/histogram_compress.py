"""Compress a histogram to the distribution values that TensorBoard shows."""

import numpy as np

# Basis points (1/100 of a percent): the mean and +-1, 2 and 3 sigma of a normal.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)


def compress_histogram(buckets: np.ndarray) -> list[float]:
    """Return the value at each of NORMAL_HISTOGRAM_BPS.

    A port of compress_histogram in tensorboard/plugins/distribution/compressor.py
    (TensorBoard 2.21). The value interpolates linearly inside the bucket that
    holds the basis point, clamped to the first left and last right edge.
    Each basis point uses an independent scalar lookup, even with nonfinite weights.

    Args:
        buckets: rows of (left, right, count).
    """
    if not len(buckets):
        return [0.0] * len(NORMAL_HISTOGRAM_BPS)
    edges = (float(buckets[0, 0]), float(buckets[-1, 1]))
    counts = buckets[:, 2]
    weights = np.cumsum(counts * NORMAL_HISTOGRAM_BPS[-1] / (counts.sum() or 1.0))
    rights = buckets[:, 1]
    lowest, highest = edges
    values: list[float] = []
    for point in NORMAL_HISTOGRAM_BPS:
        index = int(weights.searchsorted(point, side="right"))
        if index == len(weights):
            break
        before = float(weights[index - 1]) if index else 0.0
        left = max(float(rights[index - 1]), lowest) if index and before else lowest
        right = min(float(rights[index]), highest)
        values.append(
            left + (point - before) / (float(weights[index]) - before) * (right - left)
        )
    return values + [highest] * (len(NORMAL_HISTOGRAM_BPS) - len(values))
