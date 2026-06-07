import numpy as np


class _Decode:
    """Small decoding helpers for TensorBoard scalar and text fields."""

    @staticmethod
    def text(data: bytes) -> str:
        """Decode TensorBoard text bytes without failing on partial UTF-8."""
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def all_ints(values: list[object]) -> bool:
        """Return true when all values are plain Python integers."""
        return all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        )

    @staticmethod
    def scalar_array_dtype(dtypes: list[np.dtype]) -> np.dtype:
        """Return the narrow scalar dtype when all rows agree."""
        if not dtypes:
            return np.dtype("float64")
        first = dtypes[0]
        if all(dtype == first for dtype in dtypes):
            return first
        return np.dtype("float64")
