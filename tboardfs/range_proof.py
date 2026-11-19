"""Block fingerprints for bounded reads of original indexed binary bytes."""

import struct

from tboardfs.crc32c import masked_crc32c

BLOCK_SIZE = 4096
_CRC = struct.Struct("<I")


def range_proof(data: bytes) -> bytes:
    """Compute a packed CRC for each block in the payload."""
    return b"".join(
        _CRC.pack(masked_crc32c(data[begin : begin + BLOCK_SIZE]))
        for begin in range(0, len(data), BLOCK_SIZE)
    )


def _matches_range(proof: bytes, data: bytes, start: int) -> bool:
    """Check fetched blocks against the proof at a block-aligned offset."""
    index = start // BLOCK_SIZE * _CRC.size
    for position in range(0, len(data), BLOCK_SIZE):
        expected = _CRC.unpack_from(proof, index)[0]
        if masked_crc32c(data[position : position + BLOCK_SIZE]) != expected:
            return False
        index += _CRC.size
    return True
