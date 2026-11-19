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
