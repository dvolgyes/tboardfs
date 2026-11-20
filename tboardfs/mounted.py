"""mfusepy operations for direct file I/O and typed directory listings."""

from collections.abc import Iterator
from stat import S_IFDIR, S_IFREG
from typing import Any, Protocol

from mfusepy import Operations

from tboardfs.filesystem import TensorBoardFS
from tboardfs.leaf_spec import LeafSpec
from tboardfs.raw_frames import _RawFrameChildren
from tboardfs.run_listing import RunListing


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

    def readdir(  # type: ignore[override]
        self, path: str, *file_handles: object
    ) -> Iterator[tuple[str, int, int]]:
        """Return type hints with offset zero so libfuse paginates the listing."""
        del file_handles
        directory, names = self._listed_directory(path)
        children = directory["children"]

        yield ".", S_IFDIR, 0
        yield "..", S_IFDIR, 0
        if isinstance(children, _RawFrameChildren):
            for name in names:
                yield name, S_IFREG, 0
            return
        entries = children.entries if isinstance(children, RunListing) else children
        for name in names:
            mode = S_IFDIR if LeafSpec.kind_of(entries[name]) == "dir" else S_IFREG
            yield name, mode, 0

    def read(  # type: ignore[override]
        self, path: str, size: int, offset: int, *file_handles: _FileInfo | None
    ) -> bytes:
        """Read a byte range through the handle in the file info."""
        return super().read(path, size, offset, *self._handles(file_handles))

    @staticmethod
    def _handles(file_infos: tuple[_FileInfo | None, ...]) -> tuple[object, ...]:
        return tuple(None if info is None else info.fh for info in file_infos)
