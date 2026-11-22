"""Recognize fixed-width step tables suitable for bounded serialization."""

import numpy as np

METADATA = ("step", "epoch", "wall_time", "relative_time")
HISTOGRAM = (*METADATA, "bucket_left", "bucket_right", "count")
DISTRIBUTION = (*METADATA, "percentile", "value")
_MAX_ROWS = 256
_FLOAT = np.dtype(np.float64)
_INT = np.dtype(np.int64)


def _column(values: np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    return (
        type(values) is np.ndarray
        and values.shape == shape
        and values.dtype == dtype
        and values.dtype.metadata is None
        and values.flags.c_contiguous
    )


def schema(series: dict[str, np.ndarray]) -> tuple[str, ...] | None:
    """Accept canonical columns only; other tables use the general writer."""
    names = tuple(series)
    if names not in (HISTOGRAM, DISTRIBUTION):
        return None
    step = series["step"]
    if (
        type(step) is not np.ndarray
        or step.ndim != 1
        or step.dtype not in (_INT, _FLOAT)
    ):
        return None
    if not 0 < len(step) <= _MAX_ROWS:
        return None
    if all(
        _column(values, step.shape, step.dtype if name == "step" else _FLOAT)
        for name, values in series.items()
    ):
        return names
    return None
