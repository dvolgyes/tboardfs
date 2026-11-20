"""Read an indexed byte span through its original source descriptor."""

import os
from errno import EIO
from threading import Lock
from typing import BinaryIO

from tboardfs.errors import fuse_error
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import TFRecordError
from tboardfs.range_proof import BLOCK_SIZE, _matches_range


class _FrameValidations:
    """Keep bounded full-frame checks for shared descriptor fingerprints."""

    _status: dict[FrameRef, tuple[int | None, ...]]
    _lock: Lock

    def __init__(self) -> None:
        self._status = {}
        self._lock = Lock()

    def matches(self, frame: FrameRef, fingerprint: tuple[int | None, ...]) -> bool:
        """Return whether this exact frame was checked at this fingerprint."""
        with self._lock:
            return self._status.get(frame) == fingerprint

    def remember(self, frame: FrameRef, fingerprint: tuple[int | None, ...]) -> None:
        """Record a successful check, evicting the oldest entry if full."""
        with self._lock:
            if len(self._status) >= 256 and frame not in self._status:
                del self._status[next(iter(self._status))]
            self._status[frame] = fingerprint

    def discard(self, frame: FrameRef, fingerprint: tuple[int | None, ...]) -> None:
        """Remove a reported fingerprint without removing a different one."""
        with self._lock:
            if self._status.get(frame) == fingerprint:
                del self._status[frame]


class _VerifiedSpan:
    """Read one indexed payload version through a pinned descriptor."""

    size: int
    _file: BinaryIO
    _frame: FrameRef
    _checksums: bytes
    _start: int
    _lock: Lock
    _fingerprint: tuple[int | None, ...]
    _validations: _FrameValidations

    def __init__(
        self,
        file: BinaryIO,
        frame: FrameRef,
        span: tuple[int, int],
        checksums: bytes,
        validations: _FrameValidations,
    ) -> None:
        start, size = span
        self._checksums = checksums
        self._file = file
        self._frame = frame
        self._validations = validations
        self._start = start
        self.size = size
        self._lock = Lock()
        try:
            if start < 0 or size < 0 or start + size > frame.payload_size:
                raise fuse_error(EIO)
            status = os.fstat(file.fileno())
            if FileIdentity.of(status) != frame.identity:
                raise fuse_error(EIO)
            self._fingerprint = FileIdentity.fingerprint(status)
            if status.st_size < frame.offset + frame.size or (
                self._fingerprint != frame.fingerprint
                and not validations.matches(frame, self._fingerprint)
            ):
                self._verify(self._fingerprint)
        except (OSError, TFRecordError) as error:
            raise fuse_error(EIO) from error

    def _verify(self, fingerprint: tuple[int | None, ...]) -> bytes:
        record = IndexedFrame.verify(self._file, self._frame)
        if self._status() != fingerprint:
            raise fuse_error(EIO)
        self._validations.remember(self._frame, fingerprint)
        return record.payload

    def _status(self) -> tuple[int | None, ...]:
        return FileIdentity.fingerprint(os.fstat(self._file.fileno()))

    def read(self, size: int, offset: int) -> bytes:
        """Return verified bytes from a range of the indexed payload."""
        begin, end, _ = slice(offset, offset + size).indices(self.size)
        if end <= begin:
            return b""
        with self._lock:
            if self._file.closed:
                raise fuse_error(EIO)
            try:
                fingerprint = self._status()
                aligned = begin // BLOCK_SIZE * BLOCK_SIZE
                stop = min(self.size, (end + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE)
                start = self._start + aligned
                count = stop - aligned
                if (
                    fingerprint != self._fingerprint
                    and fingerprint != self._frame.fingerprint
                    and not self._validations.matches(self._frame, fingerprint)
                ):
                    result = self._verify(fingerprint)[start : start + count]
                else:
                    result = os.pread(
                        self._file.fileno(), count, self._frame.payload_offset + start
                    )
                if (
                    len(result) != count
                    or self._status() != fingerprint
                    or not _matches_range(self._checksums, result, aligned)
                ):
                    self._validations.discard(self._frame, fingerprint)
                    raise fuse_error(EIO)
                self._fingerprint = fingerprint
                return result[begin - aligned : end - aligned]
            except (OSError, TFRecordError) as error:
                raise fuse_error(EIO) from error

    def close(self) -> None:
        """Close the pinned descriptor after any in-flight read."""
        with self._lock:
            self._file.close()
