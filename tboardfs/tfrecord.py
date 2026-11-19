"""TFRecord framing: length header, masked CRC32C checks, and payload reads."""

import os
import struct
from typing import BinaryIO
from collections.abc import Iterator

from tboardfs.crc32c import masked_crc32c
from tboardfs.model import Record, TFRecordError

_HEADER_SIZE = 12
_FOOTER_SIZE = 4
DEFAULT_MAX_RECORD_SIZE = 1 << 30


def _read_header(
    handle: BinaryIO, record_offset: int, end: int, max_record_size: int
) -> tuple[bytes, int]:
    """Validate the length header and check that the whole frame fits."""
    header = handle.read(_HEADER_SIZE) if end - record_offset >= _HEADER_SIZE else b""
    if len(header) != _HEADER_SIZE:
        raise TFRecordError(
            f"truncated TFRecord length at offset {record_offset}",
            corrupt=False,
        )
    length_bytes = header[:8]
    (length_crc,) = struct.unpack("<I", header[8:])
    if masked_crc32c(length_bytes) != length_crc:
        raise TFRecordError(
            f"TFRecord length checksum mismatch at offset {record_offset}",
            corrupt=True,
        )
    (payload_size,) = struct.unpack("<Q", length_bytes)
    if payload_size > max_record_size:
        raise TFRecordError(
            f"TFRecord length {payload_size} exceeds maximum {max_record_size} at offset {record_offset}",
            corrupt=True,
        )
    payload_offset = record_offset + _HEADER_SIZE
    if end - payload_offset < payload_size + _FOOTER_SIZE:
        raise TFRecordError(
            f"truncated TFRecord payload at offset {payload_offset}",
            corrupt=False,
        )
    return header, payload_size


def iter_tfrecords(
    handle: BinaryIO,
    *,
    max_record_size: int = DEFAULT_MAX_RECORD_SIZE,
) -> Iterator[Record]:
    """Yield complete CRC-validated records from the current file position.

    The file size is fixed when iteration starts. A short final frame is
    reported as an incomplete tail; invalid checksums and oversized frames
    are corruption.
    """
    start = handle.tell()
    end = handle.seek(0, os.SEEK_END)
    handle.seek(start)
    yield from _iter_tfrecords(handle, end, max_record_size=max_record_size)


def _iter_tfrecords(
    handle: BinaryIO, end: int, *, max_record_size: int = DEFAULT_MAX_RECORD_SIZE
) -> Iterator[Record]:
    """Read validated frames up to an absolute exclusive source offset."""
    while (record_offset := handle.tell()) < end:
        header, payload_size = _read_header(handle, record_offset, end, max_record_size)
        payload_offset = record_offset + _HEADER_SIZE
        payload = handle.read(payload_size)
        footer = handle.read(_FOOTER_SIZE)
        if len(payload) != payload_size or len(footer) != _FOOTER_SIZE:
            raise TFRecordError(
                f"truncated TFRecord payload at offset {payload_offset}",
                corrupt=False,
            )
        (payload_crc,) = struct.unpack("<I", footer)
        if masked_crc32c(payload) != payload_crc:
            raise TFRecordError(
                f"TFRecord payload checksum mismatch at offset {record_offset}",
                corrupt=True,
            )
        yield Record(
            record_offset=record_offset,
            payload_offset=payload_offset,
            payload_size=payload_size,
            total_size=payload_size + _HEADER_SIZE + _FOOTER_SIZE,
            payload=payload,
            header=header,
            footer=footer,
            payload_crc=payload_crc,
        )
