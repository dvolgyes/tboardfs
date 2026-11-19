"""Access private index files through pinned directory and file descriptors."""

import os
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

_MANAGED = re.compile(r"index-[0-9a-f]{32}\.idx|pending-[0-9a-f]{32}\.tmp")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


class IndexCacheFiles:
    """Read and publish regular files without following links."""

    @staticmethod
    def _open_subdirectory(parent: int, name: str, create: bool) -> int:
        if create:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=parent)
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)

    @staticmethod
    def _private(status: os.stat_result) -> bool:
        return status.st_uid == os.geteuid() and not (status.st_mode & 0o077)

    @classmethod
    def _private_file(cls, status: os.stat_result) -> bool:
        return (
            stat.S_ISREG(status.st_mode)
            and cls._private(status)
            and status.st_nlink == 1
        )

    @classmethod
    @contextmanager
    def directory(cls, path: Path, *, create: bool) -> Iterator[int]:
        """Pin each component so operations cannot be redirected through symlinks."""
        if ".." in path.parts:
            raise PermissionError("parent traversal is not allowed in the cache path")
        fd = os.open(path.anchor, _DIRECTORY_FLAGS)
        try:
            for component in path.parts[1:]:
                child = cls._open_subdirectory(fd, component, create)
                os.close(fd)
                fd = child
            if not cls._private(os.fstat(fd)):
                raise PermissionError("index directory is not owner-only")
            yield fd
        finally:
            os.close(fd)

    @classmethod
    @contextmanager
    def open_file(
        cls, directory: int, name: str
    ) -> Iterator[tuple[int, os.stat_result]]:
        """Open only a regular, private, singly linked file."""
        if not cls._private_file(
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        ):
            raise PermissionError("index file is not private and regular")
        fd = os.open(name, _READ_FLAGS, dir_fd=directory)
        try:
            status = os.fstat(fd)
            if not cls._private_file(status):
                raise PermissionError("index file is not private and regular")
            yield fd, status
        finally:
            os.close(fd)

    @staticmethod
    def read(fd: int, length: int) -> bytes:
        """Read up to the validated byte ceiling without closing the descriptor."""
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read(length)

    @staticmethod
    def publish(directory: int, name: str, body: bytes) -> None:
        """Durably publish a complete file next to the catalog."""
        pending = f"pending-{uuid.uuid4().hex}.tmp"
        fd = os.open(pending, _CREATE_FLAGS, mode=0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending, name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(pending, dir_fd=directory)

    @classmethod
    def sweep(cls, directory: int, retained: set[str]) -> None:
        """Remove unreferenced regular files in the reserved cache namespace."""
        for name in os.listdir(directory):
            if name not in retained and _MANAGED.fullmatch(name):
                cls._remove_orphan(directory, name)

    @classmethod
    def _remove_orphan(cls, directory: int, name: str) -> None:
        with suppress(OSError):
            if cls._private_file(
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            ):
                os.unlink(name, dir_fd=directory)
