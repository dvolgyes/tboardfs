"""Read deferred tensor content from its verified source frame."""

import os
from dataclasses import dataclass
from errno import EIO
from pathlib import Path
from typing import BinaryIO

from tboardfs.crc32c import masked_crc32c
from tboardfs.errors import fuse_error
from tboardfs.event_source import EventSource
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import TFRecordError


@dataclass(frozen=True, slots=True)
class TensorSource:
    """An immutable tensor extent and the snapshot that supplied its metadata."""

    source_path: Path
    mountpoint: Path
    frame: FrameRef
    extent: tuple[int, int, int]

    def read(self) -> bytes:
        """Return the indexed content or report EIO if its evidence no longer holds."""
        try:
            with EventSource.open(
                self.source_path, mountpoint=self.mountpoint
            ) as handle:
                fd = handle.fileno()
                fingerprint = self._checked_fingerprint(os.fstat(fd))
                data = self._read_content(handle, fingerprint)
                if (
                    FileIdentity.fingerprint(os.fstat(fd)) != fingerprint
                    or self._path_fingerprint() != fingerprint
                ):
                    raise fuse_error(EIO)
                return data
        except OSError as error:
            if error.errno == EIO:
                raise
            raise fuse_error(EIO) from error
        except TFRecordError as error:
            raise fuse_error(EIO) from error

    def _path_fingerprint(self) -> tuple[int | None, ...]:
        return FileIdentity.fingerprint(
            EventSource.stat(self.source_path, mountpoint=self.mountpoint)
        )

    def _checked_fingerprint(self, status: os.stat_result) -> tuple[int | None, ...]:
        frame = self.frame
        fingerprint = FileIdentity.fingerprint(status)
        if (
            FileIdentity.of(status) != frame.identity
            or status.st_size < frame.offset + frame.size
            or self._path_fingerprint() != fingerprint
        ):
            raise fuse_error(EIO)
        return fingerprint

    def _read_content(
        self, handle: BinaryIO, fingerprint: tuple[int | None, ...]
    ) -> bytes:
        frame = self.frame
        offset, size, checksum = self.extent
        start = offset - frame.payload_offset
        if start < 0 or size <= 0 or start + size > frame.payload_size:
            raise fuse_error(EIO)
        if fingerprint == frame.fingerprint:
            handle.seek(offset)
            data = handle.read(size)
        else:
            record = IndexedFrame.verify(handle, frame)
            data = record.payload[start : start + size]
        if len(data) != size or masked_crc32c(data) != checksum:
            raise fuse_error(EIO)
        return data
