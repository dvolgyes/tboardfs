from fnmatch import fnmatchcase
import os
from pathlib import Path
import posixpath
from typing import Any

import numpy as np

from tboardfs.constants import SCALAR_FORMATS, TENSORBOARD_EVENT_GLOB

_UNSAFE_PATH_CHARS = frozenset(
    [chr(code) for code in range(0x20)] + ["\\", "/", "\x7f"]
)


def find_tensorboard_files(directory: str | Path) -> list[Path]:
    """Return TensorBoard event files below a directory, recursively.

    Symlinked directories are followed, so runs linked in from elsewhere are
    found. Each directory is walked once, by the first path in sorted order,
    so link loops end. A directory is known by its device and inode, which
    one stat gives, also on a network filesystem.
    """
    walked: set[tuple[int, int]] = set()
    found: list[Path] = []
    pending = [Path(directory)]
    while pending:
        current = pending.pop()
        try:
            status = current.stat()
        except OSError:
            continue  # removed since its parent was listed
        if (status.st_dev, status.st_ino) in walked:
            continue
        walked.add((status.st_dev, status.st_ino))
        entries = _Paths.entries(current)
        found += _Paths.event_files(current, entries)
        pending += reversed(
            [current / entry.name for entry in entries if _Paths.is_dir(entry)]
        )
    return sorted(found)


class _Paths:
    """Virtual path helpers for TensorBoard run and tag names."""

    @staticmethod
    def entries(directory: Path) -> list[os.DirEntry[str]]:
        """Return the entries of a directory by name, or none when it cannot be read.

        The entries keep the file types of the listing, so most need no stat.
        """
        try:
            with os.scandir(directory) as listing:
                return sorted(listing, key=lambda entry: entry.name)
        except OSError:
            return []

    @staticmethod
    def event_files(directory: Path, entries: list[os.DirEntry[str]]) -> list[Path]:
        """Return the paths of the event files among the entries of a directory."""
        return [
            directory / entry.name
            for entry in entries
            if fnmatchcase(entry.name, TENSORBOARD_EVENT_GLOB) and _Paths.is_file(entry)
        ]

    @staticmethod
    def is_file(entry: os.DirEntry[str]) -> bool:
        """Return whether an entry is a regular file or a link to one.

        A file in a folder that cannot be searched has no status, so it is
        not shown, as a removed file is not.
        """
        try:
            return entry.is_file()
        except OSError:
            return False

    @staticmethod
    def is_dir(entry: os.DirEntry[str], *, follow_symlinks: bool = True) -> bool:
        """Return whether an entry is a directory, or a link to one when followed."""
        try:
            return entry.is_dir(follow_symlinks=follow_symlinks)
        except OSError:
            return False

    @staticmethod
    def run_parts_for(root: Path, path: Path) -> tuple[str, ...]:
        """Return the virtual run path for an event file."""
        parent = path.parent.relative_to(root)
        if str(parent) == ".":
            return ()
        return tuple(parent.parts)

    @staticmethod
    def path_parts(path: str) -> tuple[str, ...]:
        """Normalize a virtual POSIX path into parts."""
        normalized = posixpath.normpath("/" + path.lstrip("/"))
        if normalized == "/":
            return ()
        return tuple(part for part in normalized.split("/") if part)

    @staticmethod
    def norm_virtual_path(path: str, *, base: str = "/") -> str:
        """Resolve a virtual FUSE path relative to a directory path."""
        if not path.startswith("/"):
            path = posixpath.join(base, path)
        normalized = posixpath.normpath(path)
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized

    @staticmethod
    def join_path(parts: tuple[str, ...]) -> str:
        """Join virtual path parts into an absolute POSIX path."""
        return "/" + "/".join(parts)

    @staticmethod
    def mkdirs(root: dict[str, Any], parts: tuple[str, ...]) -> dict[str, Any]:
        """Create virtual directory nodes and return the final node."""
        node = root
        for part in parts:
            node = node["children"].setdefault(part, {"type": "dir", "children": {}})
        return node

    @staticmethod
    def tag_dir_parts(tag: str) -> tuple[str, ...]:
        """Preserve TensorBoard slash hierarchy in a virtual path."""
        parts = tuple(_Paths.safe_component(part) for part in tag.split("/") if part)
        return parts or ("untagged",)

    @staticmethod
    def safe_component(name: str) -> str:
        """Return one virtual path component that cannot traverse or split.

        Dot-only names, backslashes, and control characters are percent-escaped,
        so untrusted tags and plugin names stay inside their parent directory.
        """
        if name in {".", ".."}:
            return "%2E" * len(name)
        return "".join(
            f"%{ord(ch):02X}" if ch in _UNSAFE_PATH_CHARS else ch for ch in name
        )

    @staticmethod
    def step_name(step: object, digits: int) -> str:
        """Format a TensorBoard step for a virtual filename."""
        if isinstance(step, np.generic):
            step = step.item()
        if isinstance(step, int):
            return f"{step:0{digits}d}"
        if isinstance(step, float) and step.is_integer():
            return f"{int(step):0{digits}d}"
        return str(step)

    @staticmethod
    def normalize_formats(
        formats: str | list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize comma-separated scalar export format names.

        Raises:
            ValueError: A name is not one of SCALAR_FORMATS.
        """
        items = formats.split(",") if isinstance(formats, str) else list(formats)
        normalized = []
        for item in items:
            fmt = item.strip().lower().lstrip(".")
            if fmt and fmt not in SCALAR_FORMATS:
                raise ValueError(
                    f"unsupported scalar format {fmt!r}; "
                    f"use one or more of {', '.join(SCALAR_FORMATS)}"
                )
            if fmt and fmt not in normalized:
                normalized.append(fmt)
        return tuple(normalized)
