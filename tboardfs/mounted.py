"""mfusepy operations for direct file I/O and typed directory listings."""

from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from stat import S_IFDIR, S_IFREG
from threading import Thread
from typing import Any, Protocol

from mfusepy import Operations

from tboardfs.constants import TENSORBOARD_EVENT_GLOB
from tboardfs.event_source import EventSource
from tboardfs.filesystem import TensorBoardFS
from tboardfs.leaf_spec import LeafSpec
from tboardfs.mount_index import _RunData
from tboardfs.paths import _Paths
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

    def _start_index_warmup(self) -> None:
        """Start the newest-event warmup without delaying FUSE initialization."""
        Thread(
            target=self._warm_index,
            name="tboardfs-index-warmup",
            daemon=True,
        ).start()

    def _warm_index(self) -> None:
        """Select the newest event source without holding the mount lock."""
        latest: tuple[int, Path] | None = None
        for directory, entries in _Paths.mount_walk(self.source, self._allowed):
            for entry in entries:
                if fnmatchcase(entry.name, TENSORBOARD_EVENT_GLOB):
                    latest = self._newest_event(directory / entry.name, latest)
        if latest is not None:
            _RunData.warm(latest[1], self._state.index, log=self.log_to_screen)

    def _newest_event(
        self, path: Path, latest: tuple[int, Path] | None
    ) -> tuple[int, Path] | None:
        options = self._state.index
        try:
            status = EventSource.stat(
                path, mountpoint=options.mountpoint, storage=options.storage
            )
        except OSError:
            return latest
        candidate = status.st_mtime_ns, path
        return candidate if latest is None or candidate > latest else latest

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
