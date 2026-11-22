"""Validate lazy reads against an indexed source frame and generation."""

import os
from errno import EIO
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from tboardfs.errors import fuse_error
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.model import Record, TFRecordError
from tboardfs.tfrecord import _iter_tfrecords

_SMALL_FRAME = 64 << 10


def _read_extent(handle: BinaryIO, size: int) -> bytes:
    """Read a small extent without reading ahead; a positive short read is not EOF."""
    data = handle.read(size)
    if len(data) == size or not data:
        return data
    result = bytearray(data)
    while len(result) < size and (part := handle.read(size - len(result))):
        result.extend(part)
    return bytes(result)


class IndexedFrame:
    """Read previously indexed frames without mixing source generations."""

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
    def read(path: Path, ref: FrameRef) -> Record:
        """Read the indexed record from its original generation, or report EIO."""
        small = 16 <= ref.size <= _SMALL_FRAME
        try:
            with path.open("rb", buffering=0 if small else -1) as handle:
                status = os.fstat(handle.fileno())
                if FileIdentity.of(status) != ref.identity:
                    raise TFRecordError(f"{path} changed generation", corrupt=True)
                handle.seek(ref.offset)
                if small:
                    data = _read_extent(handle, ref.size)
                    return IndexedFrame._record(
                        BytesIO(data), ref, ref.offset + len(data)
                    )
                return IndexedFrame._record(handle, ref, status.st_size)
        except (TFRecordError, OSError) as error:
            raise fuse_error(EIO) from error
