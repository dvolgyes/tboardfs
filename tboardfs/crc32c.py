"""Masked CRC32C (Castagnoli) checksums as used by TFRecord framing."""

_CRC32C_POLYNOMIAL = 0x82F63B78
_CRC_MASK_DELTA = 0xA282EAD8


def _crc32c_table() -> tuple[int, ...]:
    table = []
    for index in range(256):
        crc = index
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC32C_POLYNOMIAL if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_CRC32C_TABLE = _crc32c_table()


def masked_crc32c(data: bytes) -> int:
    """Return the TFRecord-masked CRC32C checksum of data."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF
