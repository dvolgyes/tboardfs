"""Access regular event files without traversing the local FUSE mount."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from errno import EINVAL, ELOOP, ENOENT, ENOTDIR, ESTALE
from functools import partial
import os
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import BinaryIO

from tboardfs.file_status import FileIdentity


class EventSource:
    """Read event sources from outside the mount, including through symlinks."""

    @classmethod
    def stat(cls, path: Path, *, mountpoint: Path | None = None) -> os.stat_result:
        """Read the current regular-file status without opening it for data."""
        if mountpoint is None:
            return cls._regular(path, path.stat())
        with cls._target(path, (str(mountpoint),)) as (_, _, _, status):
            return cls._regular(path, status)

    @classmethod
    def directory_stat(
        cls,
        path: Path,
        *,
        mountpoint: Path | None = None,
        storage: Path | None = None,
    ) -> os.stat_result:
        """Read directory status without entering the mount or private index."""
        blocked = tuple(str(root) for root in (mountpoint, storage) if root is not None)
        if blocked:
            with cls._target(path, blocked, directory=True) as (_, _, _, status):
                return status
        status = path.stat()
        if not S_ISDIR(status.st_mode):
            raise NotADirectoryError(ENOTDIR, "not a source directory", path)
        return status

    @classmethod
    @contextmanager
    def _sibling_stat(
        cls, parent: Path | None, *, mountpoint: Path | None
    ) -> Iterator[Callable[[Path], os.stat_result]]:
        """Check regular siblings through one guarded parent for this operation."""
        if mountpoint is None:
            yield cls.stat
            return
        fallback = partial(cls.stat, mountpoint=mountpoint)
        if parent is None:
            yield fallback
            return
        with ExitStack() as stack:
            try:
                resolved, fd, _, _ = stack.enter_context(
                    cls._target(parent, (str(mountpoint),), directory=True)
                )
            except OSError:
                yield fallback
                return
            blocked = (
                mountpoint.name if resolved == os.fspath(mountpoint.parent) else None
            )
            yield partial(cls._stat_at, parent, fd, blocked, fallback)

    @classmethod
    def _stat_at(
        cls,
        parent: Path,
        fd: int,
        blocked: str | None,
        fallback: Callable[[Path], os.stat_result],
        path: Path,
    ) -> os.stat_result:
        """Read a sibling's status without following a link or entering the mount."""
        name = path.name
        if name not in ("", "..", blocked) and path.parent == parent:
            status = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if not S_ISLNK(status.st_mode):
                return cls._regular(path, status)
        return fallback(path)

    @classmethod
    def open(cls, path: Path, *, mountpoint: Path | None = None) -> BinaryIO:
        """Open a nonblocking descriptor and check that same descriptor's type."""
        flags = os.O_RDONLY | os.O_NONBLOCK
        expected = None
        if mountpoint is None:
            fd = os.open(path, flags)
        else:
            with cls._target(path, (str(mountpoint),)) as (_, parent, name, status):
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
        path: Path, blocked: tuple[str, ...], *, directory: bool = False
    ) -> Iterator[tuple[str, int, str, os.stat_result]]:
        root = "/"
        pending = list(reversed(path.absolute().parts[1:]))
        flags = (
            os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY))
        )
        links = 0
        if root in blocked:
            raise FileNotFoundError(ENOENT, "source enters a protected path", path)
        parents = [(root, os.open(root, flags))]
        try:
            while pending:
                part = pending.pop()
                if part == "..":
                    if len(parents) > 1:
                        os.close(parents.pop()[1])
                    continue
                resolved, parent = parents[-1]
                child = f"{resolved.rstrip('/')}/{part}"
                if child in blocked:
                    raise FileNotFoundError(
                        ENOENT, "source enters a protected path", path
                    )
                if pending or directory:
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
                    if pending or directory:
                        raise NotADirectoryError(
                            ENOTDIR, "not a source directory", Path(child)
                        )
                    yield resolved, parent, part, status
                    return
                links += 1
                if links > 40:
                    raise OSError(ELOOP, "too many source symlinks", path)
                target = Path(os.readlink(part, dir_fd=parent))
                if target.is_absolute():
                    while len(parents) > 1:
                        os.close(parents.pop()[1])
                pending.extend(reversed(target.parts[int(target.is_absolute()) :]))
            if directory:
                resolved, fd = parents[-1]
                yield resolved, fd, "", os.fstat(fd)
                return
        finally:
            for _, fd in reversed(parents):
                os.close(fd)
        raise OSError(EINVAL, "source is not a regular file", path)

    @staticmethod
    def _regular(path: Path, status: os.stat_result) -> os.stat_result:
        if not S_ISREG(status.st_mode):
            raise OSError(EINVAL, "source is not a regular file", path)
        return status
