"""Publish complete virtual files without following destination links."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
import os
from pathlib import Path
import secrets
import stat


class _OutputConflictError(FileExistsError):
    """The destination exists and cannot be replaced under the selected policy."""


_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@contextmanager
def _descend(fd: int, parts: tuple[str, ...]) -> Iterator[int]:
    current = fd
    try:
        for name in parts:
            try:
                child = os.open(name, _DIRECTORY, dir_fd=current)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(name, dir_fd=current)
                child = os.open(name, _DIRECTORY, dir_fd=current)
            except NotADirectoryError as error:
                mode = os.stat(name, dir_fd=current, follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise OSError(
                        f"refusing output link (possibly outside the output directory): {name}"
                    ) from error
                raise
            if current != fd:
                os.close(current)
            current = child
        yield current
    finally:
        if current != fd:
            os.close(current)


def _existing(directory: int, name: str, policy: str) -> bool:
    try:
        mode = os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise OSError(f"output is not a regular file: {name}")
    if policy == "fail":
        raise _OutputConflictError(name)
    return policy == "skip"


class OutputPublisher:
    """Publish output relative to a pinned host directory."""

    @staticmethod
    @contextmanager
    def directory(path: Path) -> Iterator[int]:
        """Pin the selected directory without following links."""
        root = path.anchor or "."
        parts = path.parts[1:] if path.anchor else path.parts
        fd = os.open(root, _DIRECTORY)
        try:
            with _descend(fd, parts) as directory:
                yield directory
        finally:
            os.close(fd)

    @staticmethod
    def publish(
        root: int, parts: tuple[str, ...], render: Callable[[], bytes], policy: str
    ) -> bool:
        """Publish a complete file below a pinned directory; return False if skipped."""
        if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
            raise ValueError("unsafe output path component")
        if policy not in {"fail", "skip", "overwrite"}:
            raise ValueError(f"unknown output policy: {policy}")
        name = parts[-1]
        with _descend(root, parts[:-1]) as directory:
            if _existing(directory, name, policy):
                return False
            temp = f".tboardfs-{secrets.token_hex(12)}.tmp"
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=directory
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(render())
                if policy == "overwrite":
                    os.replace(temp, name, src_dir_fd=directory, dst_dir_fd=directory)
                else:
                    try:
                        os.link(
                            temp,
                            name,
                            src_dir_fd=directory,
                            dst_dir_fd=directory,
                            follow_symlinks=False,
                        )
                    except FileExistsError as error:
                        if _existing(directory, name, policy):
                            return False
                        raise _OutputConflictError(name) from error
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temp, dir_fd=directory)
        return True

    @classmethod
    def publish_get(cls, output: str, basename: str, data: bytes, policy: str) -> Path:
        """Publish to a file, or into an existing or explicitly requested directory."""
        path = Path(output)
        if output.endswith("/") or not path.name or path.name == "..":
            with cls.directory(path) as directory:
                cls.publish(directory, (basename,), lambda: data, policy)
            return path / basename
        with cls.directory(path.parent) as parent:
            try:
                mode = os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_mode
            except FileNotFoundError:
                mode = 0
            parts = (path.name, basename) if stat.S_ISDIR(mode) else (path.name,)
            cls.publish(parent, parts, lambda: data, policy)
            return path.parent.joinpath(*parts)
