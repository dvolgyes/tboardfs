"""Validate lazy reads against an indexed source frame and generation."""

from errno import EIO
import os
from pathlib import Path
from typing import BinaryIO

from tboardfs.errors import fuse_error
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.model import Record, TFRecordError
from tboardfs.tfrecord import iter_tfrecords


class IndexedFrame:
    """Read previously indexed frames without mixing source generations."""

    @staticmethod
    def verify(handle: BinaryIO, ref: FrameRef) -> Record:
        """Validate the frame size, both checksums, and indexed payload checksum."""
        handle.seek(ref.offset)
        record = next(iter_tfrecords(handle, max_record_size=ref.payload_size), None)
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
        try:
            with path.open("rb") as handle:
                if FileIdentity.of(os.fstat(handle.fileno())) != ref.identity:
                    raise TFRecordError(f"{path} changed generation", corrupt=True)
                return IndexedFrame.verify(handle, ref)
        except (TFRecordError, OSError) as error:
            raise fuse_error(EIO) from error
