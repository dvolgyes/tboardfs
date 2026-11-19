"""Access regular event files without traversing the local FUSE mount."""

from collections.abc import Iterator
from contextlib import contextmanager
from errno import EINVAL, ELOOP, ENOENT, ENOTDIR, ESTALE
import os
from pathlib import Path
from stat import S_ISLNK, S_ISREG
from typing import BinaryIO

from tboardfs.file_status import FileIdentity


class EventSource:
    """Read event sources from outside the mount, including through symlinks."""

    @classmethod
    def stat(cls, path: Path, *, mountpoint: Path | None = None) -> os.stat_result:
        """Read the current regular-file status without opening it for data."""
        if mountpoint is None:
            return cls._regular(path, path.stat())
        with cls._target(path, mountpoint) as (_, _, status):
            return cls._regular(path, status)

    @classmethod
    def open(cls, path: Path, *, mountpoint: Path | None = None) -> BinaryIO:
        """Open a nonblocking descriptor and check that same descriptor's type."""
        flags = os.O_RDONLY | os.O_NONBLOCK
        expected = None
        if mountpoint is None:
            fd = os.open(path, flags)
        else:
            with cls._target(path, mountpoint) as (parent, name, status):
                expected = FileIdentity.of(cls._regular(path, status))
                fd = os.open(name, flags | os.O_NOFOLLOW, dir_fd=parent)
        try:
            actual = cls._regular(path, os.fstat(fd))
            if expected is not None and expected != FileIdentity.of(actual):
                raise OSError(ESTALE, "source changed while opening", path)
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    @contextmanager
    def _target(
        path: Path, mountpoint: Path
    ) -> Iterator[tuple[int, str, os.stat_result]]:
        root = Path("/")
        pending = list(reversed(path.absolute().parts[1:]))
        flags = (
            os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY))
        )
        links = 0
        parents = [(root, os.open(root, flags))]
        try:
            while pending:
                part = pending.pop()
                if part == "..":
                    if len(parents) > 1:
                        os.close(parents.pop()[1])
                    continue
                directory, parent = parents[-1]
                child = directory / part
                if child == mountpoint:
                    raise FileNotFoundError(ENOENT, "source enters own mount", path)
                if pending:
                    try:
                        fd = os.open(part, flags, dir_fd=parent)
                    except OSError as error:
                        if error.errno not in {ENOTDIR, ELOOP}:
                            raise
                    else:
                        parents.append((child, fd))
                        continue
                status = os.stat(part, dir_fd=parent, follow_symlinks=False)
                if not S_ISLNK(status.st_mode):
                    if pending:
                        raise NotADirectoryError(
                            ENOTDIR, "not a source directory", child
                        )
                    yield parent, part, status
                    return
                links += 1
                if links > 40:
                    raise OSError(ELOOP, "too many source symlinks", path)
                target = Path(os.readlink(part, dir_fd=parent))
                if target.is_absolute():
                    while len(parents) > 1:
                        os.close(parents.pop()[1])
                pending.extend(reversed(target.parts[int(target.is_absolute()) :]))
        finally:
            for _, fd in reversed(parents):
                os.close(fd)
        raise OSError(EINVAL, "source is not a regular file", path)

    @staticmethod
    def _regular(path: Path, status: os.stat_result) -> os.stat_result:
        if not S_ISREG(status.st_mode):
            raise OSError(EINVAL, "source is not a regular file", path)
        return status
