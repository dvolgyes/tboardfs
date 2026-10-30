"""TFRecord framing: length header, masked CRC32C checks, and payload reads."""

import os
import struct
from typing import BinaryIO
from collections.abc import Iterator

from tboardfs.crc32c import masked_crc32c
from tboardfs.model import Record, TFRecordError

_HEADER_SIZE = 12
_FOOTER_SIZE = 4


def iter_tfrecords(handle: BinaryIO) -> Iterator[Record]:
    """Yield complete TFRecord payloads from the current file position.

    The end of the data is fixed when iteration starts, so records appended
    later wait for the next pass. The length header CRC is verified; payload
    CRCs are not, because a pure-Python CRC over every payload is too slow.

    Raises:
        TFRecordError: The last record is incomplete, or a length header fails
            its checksum.
    """
    start = handle.tell()
    end = handle.seek(0, os.SEEK_END)
    handle.seek(start)
    while (record_offset := handle.tell()) < end:
        if end - record_offset < _HEADER_SIZE:
            raise TFRecordError(
                f"truncated TFRecord length at offset {record_offset}",
                corrupt=False,
            )
        header = handle.read(_HEADER_SIZE)
        length_bytes = header[:8]
        (length_crc,) = struct.unpack("<I", header[8:])
        if masked_crc32c(length_bytes) != length_crc:
            raise TFRecordError(
                f"TFRecord length checksum mismatch at offset {record_offset}",
                corrupt=True,
            )
        (payload_size,) = struct.unpack("<Q", length_bytes)
        payload_offset = record_offset + _HEADER_SIZE
        if end - payload_offset < payload_size + _FOOTER_SIZE:
            raise TFRecordError(
                f"truncated TFRecord payload at offset {payload_offset}",
                corrupt=False,
            )
        payload = handle.read(payload_size)
        footer = handle.read(_FOOTER_SIZE)
        if len(payload) != payload_size or len(footer) != _FOOTER_SIZE:
            raise TFRecordError(
                f"truncated TFRecord payload at offset {payload_offset}",
                corrupt=False,
            )
        yield Record(
            record_offset=record_offset,
            payload_offset=payload_offset,
            payload_size=payload_size,
            total_size=payload_size + _HEADER_SIZE + _FOOTER_SIZE,
            payload=payload,
        )
