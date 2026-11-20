"""Open the direct sidecar files written by the Open3D TensorBoard plugin."""

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tboardfs.model import SidecarEntry

OPEN3D_PLUGIN = "Open3D"
OPEN3D_DIRECTORY = ("plugins", OPEN3D_PLUGIN)
OPEN3D_ASSETS = (*OPEN3D_DIRECTORY, "~assets")


class Open3DAsset:
    """Find or read a sidecar without following links below its run root."""

    @staticmethod
    def matches(parts: tuple[str, ...]) -> bool:
        """Match the exact sidecar location and suffix used by the writer."""
        return (
            len(parts) == 3
            and parts[:2] == OPEN3D_DIRECTORY
            and parts[-1].endswith(".msgpack")
        )

    @staticmethod
    def stat(path: Path) -> SidecarEntry | None:
        """Index a real regular file without requiring permission to read it."""
        try:
            with Open3DAsset._directory(path.parents[2]) as directory:
                status = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except OSError:
            return None
        if not stat.S_ISREG(status.st_mode):
            return None
        return SidecarEntry(
            path, OPEN3D_PLUGIN, (path.name,), status.st_size, status.st_mtime_ns
        )

    @staticmethod
    def read(entry: SidecarEntry) -> bytes:
        """Read a regular sidecar without following links below the run root."""
        path = entry.source_path
        with Open3DAsset._directory(path.parents[2]) as directory:
            fd = os.open(path.name, os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, "rb") as file:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), path
                    )
                return file.read()

    @staticmethod
    @contextmanager
    def _directory(run_dir: Path) -> Iterator[int]:
        with ExitStack() as stack:
            fd = os.open(run_dir, os.O_DIRECTORY)
            stack.callback(os.close, fd)
            for part in OPEN3D_DIRECTORY:
                fd = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                stack.callback(os.close, fd)
            yield fd
