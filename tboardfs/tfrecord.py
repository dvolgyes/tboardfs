"""TFRecord framing: length header, masked CRC32C checks, and payload reads."""

import os
import struct
from collections.abc import Iterator
from typing import BinaryIO

from tboardfs.crc32c import masked_crc32c
from tboardfs.model import Record, TFRecordError

_HEADER_SIZE = 12
_FOOTER_SIZE = 4
DEFAULT_MAX_RECORD_SIZE = 1 << 30


def _read_header(
    handle: BinaryIO,
    record_offset: int,
    end: int,
    max_record_size: int,
    previous: tuple[bytes, int] | None,
) -> tuple[bytes, int]:
    """Validate the length header and check that the whole frame fits."""
    header = handle.read(_HEADER_SIZE) if end - record_offset >= _HEADER_SIZE else b""
    if previous is None or header != previous[0]:
        if len(header) != _HEADER_SIZE:
            raise TFRecordError(
                f"truncated TFRecord length at offset {record_offset}",
                corrupt=False,
            )
        payload_size, length_crc = struct.unpack("<QI", header)
        if masked_crc32c(header[:8]) != length_crc:
            raise TFRecordError(
                f"TFRecord length checksum mismatch at offset {record_offset}",
                corrupt=True,
            )
        previous = header, payload_size
    else:
        payload_size = previous[1]
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
    return previous


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
    yield from _iter_tfrecords(
        handle, max_record_size=max_record_size, shared_cursor=True
    )


def _iter_tfrecords(
    handle: BinaryIO,
    *,
    end: int | None = None,
    stop: int | None = None,
    max_record_size: int = DEFAULT_MAX_RECORD_SIZE,
    shared_cursor: bool = False,
) -> Iterator[Record]:
    """Yield frames starting before stop and bounded by end."""
    record_offset = handle.tell()
    if end is None:
        end = handle.seek(0, os.SEEK_END)
        handle.seek(record_offset)
    stop = end if stop is None else min(stop, end)
    previous = None
    while record_offset < stop:
        previous = _read_header(handle, record_offset, end, max_record_size, previous)
        header, payload_size = previous
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
        total_size = payload_size + _HEADER_SIZE + _FOOTER_SIZE
        yield Record(
            record_offset=record_offset,
            payload_offset=payload_offset,
            payload_size=payload_size,
            total_size=total_size,
            payload=payload,
            header=header,
            footer=footer,
            payload_crc=payload_crc,
        )
        record_offset = handle.tell() if shared_cursor else record_offset + total_size
