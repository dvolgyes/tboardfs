#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-crc32c>=1.7,<2"]
# ///
"""Generate deterministic valid and damaged TFRecord framing cases."""

import argparse
from pathlib import Path
import struct

import google_crc32c


_MASK_DELTA = 0xA282EAD8
_MAX_RECORD_SIZE = 1 << 30


def _masked_crc(data: bytes) -> int:
    crc = int(google_crc32c.value(data))
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def _scalar_frame(step: int, value: float) -> bytes:
    payload = (
        b"\x09"
        + struct.pack("<d", float(step + 1))
        + b"\x10"
        + bytes([step])
        + b"\x2a\x0d\x0a\x0b\x0a\x04loss\x15"
        + struct.pack("<f", value)
    )
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", _masked_crc(length))
        + payload
        + struct.pack("<I", _masked_crc(payload))
    )


def main() -> None:
    """Write two scalar events and their damaged framing variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    first = _scalar_frame(1, 1.25)
    second = _scalar_frame(2, 2.5)
    (output / "events.out.tfevents.adjacent-valid").write_bytes(first + second)

    bad_length = bytearray(first)
    bad_length[8] ^= 0x01
    (output / "events.out.tfevents.bad-length-crc").write_bytes(bad_length)

    bad_payload = bytearray(first)
    bad_payload[-1] ^= 0x01
    (output / "events.out.tfevents.bad-payload-crc").write_bytes(bad_payload)

    (output / "events.out.tfevents.valid-prefix-incomplete-tail").write_bytes(
        first + second[:-2]
    )

    oversized_length = struct.pack("<Q", _MAX_RECORD_SIZE + 1)
    (output / "events.out.tfevents.header-only-over-ceiling").write_bytes(
        oversized_length + struct.pack("<I", _masked_crc(oversized_length))
    )


if __name__ == "__main__":
    main()
