"""Validate lazy reads against an indexed source frame and generation."""

import os
import struct
from errno import EIO
from pathlib import Path
from typing import BinaryIO

from tboardfs.crc32c import masked_crc32c
from tboardfs.errors import fuse_error
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.model import Record, TFRecordError
from tboardfs.tfrecord import _FOOTER_SIZE, _HEADER_SIZE, _iter_tfrecords

_SMALL_FRAME = 64 << 10
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def _seek_read(fd: int, size: int, offset: int) -> bytes:
    """Read from a private descriptor on platforms without positional reads."""
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, size)


def _pread_extent(path: Path, ref: FrameRef) -> bytes:
    fd = os.open(path, _READ_FLAGS)
    try:
        if FileIdentity.of(os.fstat(fd)) != ref.identity:
            raise TFRecordError(f"{path} changed generation", corrupt=True)
        read_at = getattr(os, "pread", _seek_read)
        data = read_at(fd, ref.size, ref.offset)
        if len(data) == ref.size or not data:
            return data
        result = bytearray(data)
        while (received := len(result)) < ref.size:
            part = read_at(fd, ref.size - received, ref.offset + received)
            if not part:
                break
            result.extend(part)
        return bytes(result)
    finally:
        os.close(fd)


class IndexedFrame:
    """Read previously indexed frames without mixing source generations."""

    @staticmethod
    def _verified_bytes(data: bytes, ref: FrameRef, raw: bool) -> bytes:
        if len(data) != ref.size:
            raise TFRecordError(
                f"truncated indexed TFRecord at offset {ref.offset}", corrupt=False
            )
        payload_size, length_crc = struct.unpack_from("<QI", data)
        footer_offset = ref.size - _FOOTER_SIZE
        (payload_crc,) = struct.unpack_from("<I", data, footer_offset)
        if (
            payload_size != ref.payload_size
            or length_crc != masked_crc32c(data[:8])
            or payload_crc != ref.checksum
        ):
            raise TFRecordError(
                f"indexed TFRecord changed at offset {ref.offset}", corrupt=True
            )
        payload = data[_HEADER_SIZE:footer_offset]
        if masked_crc32c(payload) != payload_crc:
            raise TFRecordError(
                f"TFRecord payload checksum mismatch at offset {ref.offset}",
                corrupt=True,
            )
        return data if raw else payload

    @staticmethod
    def verify(handle: BinaryIO, ref: FrameRef) -> Record:
        """Validate the frame size, both checksums, and indexed payload checksum."""
        handle.seek(ref.offset)
        return IndexedFrame._record(handle, ref, None)

    @staticmethod
    def _record(handle: BinaryIO, ref: FrameRef, end: int | None) -> Record:
        record = next(
            _iter_tfrecords(
                handle, start=ref.offset, end=end, max_record_size=ref.payload_size
            ),
            None,
        )
        if (
            record is None
            or record.total_size != ref.size
            or record.payload_crc != ref.checksum
        ):
            raise TFRecordError(
                f"indexed TFRecord changed at offset {ref.offset}", corrupt=True
            )
        return record

    @staticmethod
    def read(path: Path, ref: FrameRef, *, raw: bool = False) -> bytes:
        """Read the verified payload or full frame from its original generation."""
        small = _HEADER_SIZE + _FOOTER_SIZE <= ref.size <= _SMALL_FRAME
        try:
            if small:
                return IndexedFrame._verified_bytes(_pread_extent(path, ref), ref, raw)
            with path.open("rb") as handle:
                status = os.fstat(handle.fileno())
                if FileIdentity.of(status) != ref.identity:
                    raise TFRecordError(f"{path} changed generation", corrupt=True)
                handle.seek(ref.offset)
                record = IndexedFrame._record(handle, ref, status.st_size)
                return (
                    b"".join((record.header, record.payload, record.footer))
                    if raw
                    else record.payload
                )
        except (TFRecordError, OSError) as error:
            raise fuse_error(EIO) from error
