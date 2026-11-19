"""Encode UTF-8 without allocating a string buffer over the index budget."""


def encode_utf8(value: str, limit: int) -> bytes:
    """Reject text that cannot fit before allocating its encoded bytes."""
    length = len(value)
    if length > limit:
        raise ValueError("UTF-8 text exceeds index budget")
    if length * 4 > limit and not value.isascii():
        size = sum(
            1 + (code > 0x7F) + (code > 0x7FF) + (code > 0xFFFF)
            for code in map(ord, value)
        )
        if size > limit:
            raise ValueError("UTF-8 text exceeds index budget")
    return value.encode()
