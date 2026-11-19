"""Masked CRC32C (Castagnoli) checksums as used by TFRecord framing."""

import google_crc32c

_CRC_MASK_DELTA = 0xA282EAD8


def masked_crc32c(data: bytes) -> int:
    """Return the TFRecord-masked CRC32C checksum of data."""
    crc: int = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF
