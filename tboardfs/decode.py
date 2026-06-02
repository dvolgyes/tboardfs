import numpy as np


class _Decode:
    """Small decoding helpers for TensorBoard scalar and text fields."""

    @staticmethod
    def text(data: bytes) -> str:
        """Decode TensorBoard text bytes without failing on partial UTF-8."""
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def signed_int32(value: int) -> int:
        """Decode a protobuf int32 stored in unsigned two's-complement form."""
        value &= 0xFFFFFFFF
        if value >= 0x80000000:
            return value - 0x100000000
        return value

    @staticmethod
    def signed_int64(value: int) -> int:
        """Decode a protobuf int64 stored in unsigned two's-complement form."""
        value &= 0xFFFFFFFFFFFFFFFF
        if value >= 0x8000000000000000:
            return value - 0x10000000000000000
        return value

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
