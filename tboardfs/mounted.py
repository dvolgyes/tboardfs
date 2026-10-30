"""mfusepy operations that give each open file its own uncached version."""

from typing import Any, Protocol

from mfusepy import Operations

from tboardfs.filesystem import TensorBoardFS


class _FileInfo(Protocol):
    """The ``fuse_file_info`` fields that the operations use."""

    flags: int
    fh: int
    direct_io: int


class MountedTensorBoardFS(TensorBoardFS, Operations):  # type: ignore[misc]
    """TensorBoardFS operations for a FUSE mount made with ``raw_fi=True``.

    The kernel page cache and its cached file size are shared by every
    handle of a file, so with a growing run they mix bytes of two versions.
    Each open file therefore uses direct I/O: the kernel sends every read
    to the bytes kept for that handle and does not stop at a cached size.
    File info arguments are ``fuse_file_info`` structures, or ``None`` when
    the kernel has no open handle.
    """

    def open(self, path: str, info: _FileInfo) -> int:  # type: ignore[override]
        """Open a virtual file with direct I/O."""
        info.fh = super().open(path, info.flags)
        info.direct_io = 1
        return 0

    def release(self, path: str, info: _FileInfo) -> int:  # type: ignore[override]
        """Free the bytes kept for the handle in the file info."""
        return super().release(path, info.fh)

    def getattr(  # type: ignore[override]
        self, path: str, *file_handles: _FileInfo | None
    ) -> dict[str, Any]:
        """Return stat data for a path or for the handle in the file info."""
        return super().getattr(path, *self._handles(file_handles))

    def read(  # type: ignore[override]
        self, path: str, size: int, offset: int, *file_handles: _FileInfo | None
    ) -> bytes:
        """Read a byte range through the handle in the file info."""
        return super().read(path, size, offset, *self._handles(file_handles))

    @staticmethod
    def _handles(file_infos: tuple[_FileInfo | None, ...]) -> tuple[object, ...]:
        return tuple(None if info is None else info.fh for info in file_infos)
