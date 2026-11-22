#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-crc32c>=1.7,<2"]
# ///
"""Copy a byte-exact, CRC-validated TFRecord prefix into a new event file."""

import argparse
import os
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, BinaryIO

from google_crc32c import value

_HEADER = struct.Struct("<QI")
_CRC = struct.Struct("<I")


def _masked_crc(data: bytes) -> int:
    crc = int(value(data))
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class _PrefixReader:
    """Validate consecutive TFRecords within a fixed source size and byte cap."""

    _reader: BinaryIO
    _limit: int
    _end: int

    def __init__(self, reader: BinaryIO, limit: int) -> None:
        self._reader = reader
        self._limit = limit
        self._end = os.fstat(reader.fileno()).st_size

    def _read_frame(self, offset: int) -> tuple[bytes, bytes, bytes] | None:
        """Validate a frame or return None if it will not fit the byte budget."""
        header = self._reader.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError(f"incomplete TFRecord header at offset {offset}")
        size, header_crc = _HEADER.unpack(header)
        if _masked_crc(header[:8]) != header_crc:
            raise ValueError(f"invalid TFRecord length checksum at offset {offset}")
        frame_end = offset + _HEADER.size + size + _CRC.size
        if self._limit < self._end and frame_end > self._limit:
            return None
        if frame_end > self._end:
            raise ValueError(f"incomplete TFRecord payload at offset {offset}")
        payload = self._reader.read(size)
        footer = self._reader.read(_CRC.size)
        if len(payload) != size or len(footer) != _CRC.size:
            raise ValueError(f"incomplete TFRecord payload at offset {offset}")
        if _masked_crc(payload) != _CRC.unpack(footer)[0]:
            raise ValueError(f"invalid TFRecord payload checksum at offset {offset}")
        return header, payload, footer

    def write_to(self, writer: IO[bytes]) -> None:
        """Write whole frames, checking for truncated tails when the cap includes EOF."""
        min_frame_size = _HEADER.size + _CRC.size
        last_frame_start = self._limit - min_frame_size
        stop = self._end if self._end <= self._limit else last_frame_start + 1
        offset = 0
        while offset < stop:
            frame = self._read_frame(offset)
            if frame is None:
                break
            writer.writelines(frame)
            offset += min_frame_size + len(frame[1])
        if not offset:
            raise ValueError("no complete TFRecord fits the byte limit")


def _copy_prefix(source: Path, output: Path, limit: int) -> None:
    """Publish a closed, validated prefix without replacing an existing path."""
    with NamedTemporaryFile(dir=output.parent, delete_on_close=False) as writer:
        with source.open("rb") as reader:
            _PrefixReader(reader, limit).write_to(writer.file)
        writer.close()
        output.hardlink_to(writer.name)


def main() -> None:
    """Extract a prefix without reading payloads beyond the byte limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--max-bytes", type=int, default=40 * 2**20)
    args = parser.parse_args()
    if args.max_bytes < 1:
        parser.error("--max-bytes must be positive")
    try:
        _copy_prefix(args.source, args.output, args.max_bytes)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
