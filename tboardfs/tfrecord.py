import struct
from typing import BinaryIO
from collections.abc import Iterator

from tboardfs.model import Record


def iter_tfrecords(handle: BinaryIO) -> Iterator[Record]:
    """Yield complete TFRecord payloads from the current file position."""
    while True:
        record_offset = handle.tell()
        length_bytes = handle.read(8)
        if not length_bytes:
            return
        if len(length_bytes) != 8:
            raise ValueError(f"truncated TFRecord length at offset {record_offset}")

        (payload_size,) = struct.unpack("<Q", length_bytes)
        length_crc = handle.read(4)
        if len(length_crc) != 4:
            raise ValueError(
                f"truncated TFRecord length CRC at offset {record_offset + 8}"
            )

        payload_offset = handle.tell()
        payload = handle.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError(f"truncated TFRecord payload at offset {payload_offset}")

        payload_crc = handle.read(4)
        if len(payload_crc) != 4:
            raise ValueError(
                f"truncated TFRecord payload CRC at offset {payload_offset + payload_size}"
            )

        yield Record(
            record_offset=record_offset,
            payload_offset=payload_offset,
            payload_size=payload_size,
            total_size=payload_size + 16,
            payload=payload,
        )
