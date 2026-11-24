import os
import posixpath
from collections.abc import Callable, Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import xxhash

from tboardfs.constants import SCALAR_FORMATS, TENSORBOARD_EVENT_GLOB
from tboardfs.event_source import EventSource

_UNSAFE_PATH_CHARS = frozenset(
    [chr(code) for code in range(0x20)] + ["\\", "/", "\x7f", "%", "~"]
)
NAME_MAP = ".tboardfs-names.json"


class _DirectoryNode(dict[str, Any]):
    """A virtual directory with its listing cached after publication."""

    __slots__: tuple[str, ...] = ("_sorted_names",)

    _sorted_names: tuple[str, ...] | None

    def __init__(self) -> None:
        super().__init__(type="dir", children={})
        self._sorted_names = None

    def sorted_names(self) -> tuple[str, ...]:
        """Return the sorted child names once the directory is published."""
        if self._sorted_names is None:
            self._sorted_names = tuple(sorted(self["children"]))
        return self._sorted_names


def find_tensorboard_files(directory: str | Path) -> list[Path]:
    """Return TensorBoard event files below a directory, recursively.

    Symlinked directories are followed, so runs linked in from elsewhere are
    found. Each directory is walked once, by the first path in sorted order,
    so link loops end.
    """
    return sorted(
        path
        for folder, entries in _Paths.event_dirs(Path(directory)).items()
        for path in _Paths.event_files(folder, entries)
    )


class _Paths:
    """Virtual path helpers for TensorBoard run and tag names."""

    @staticmethod
    def event_dirs(directory: Path) -> dict[Path, list[os.DirEntry[str]]]:
        """Return the listing of each directory below ``directory`` with event files.

        A directory is known by its device and inode, which one stat gives,
        also on a network filesystem.
        """
        walked: set[tuple[int, int]] = set()
        listings: dict[Path, list[os.DirEntry[str]]] = {}
        pending = [directory]
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
            if _Paths.event_files(current, entries):
                listings[current] = entries
            pending += reversed(
                [current / entry.name for entry in entries if _Paths.is_dir(entry)]
            )
        return listings

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
    def mount_entries(
        directory: Path, allowed: Callable[[Path], bool]
    ) -> list[os.DirEntry[str]]:
        """List the source without following links into blocked directories."""
        if not allowed(directory):
            return []
        return [
            entry
            for entry in _Paths.entries(directory)
            if not entry.is_symlink() or allowed(Path(entry.path))
        ]

    @staticmethod
    def mount_walk(
        directory: Path, allowed: Callable[[Path], bool]
    ) -> Iterator[tuple[Path, list[os.DirEntry[str]]]]:
        """Walk source branches using the same link rules as mount discovery."""
        pending = [directory]
        while pending:
            current = pending.pop()
            entries = _Paths.mount_entries(current, allowed)
            yield current, entries
            pending.extend(
                current / name for name in _Paths.child_names(current, entries, ())
            )

    @staticmethod
    def check_branch(
        directory: Path,
        allowed: Callable[[Path], bool],
        mountpoint: Path | None,
        *,
        remember: bool,
    ) -> list[list[os.DirEntry[str]]] | None:
        """Find a log or return the scanned listings for sidecar discovery."""
        listings = []
        for current, entries in _Paths.mount_walk(directory, allowed):
            if _Paths.event_files(current, entries, mountpoint=mountpoint):
                return None
            if remember and entries:
                listings.append(entries)
        return listings

    @staticmethod
    def child_names(
        directory: Path, entries: list[os.DirEntry[str]], owned: set[Path] | tuple[()]
    ) -> list[str]:
        """List physical child branches except sidecar roots and link cycles."""
        children = [
            entry
            for entry in entries
            if _Paths.is_dir(entry) and Path(entry.path) not in owned
        ]
        if not any(entry.is_symlink() for entry in children):
            return [entry.name for entry in children]
        parent = directory.resolve()
        seen = {parent, *(p.resolve() for p in directory.parents)}
        names = []
        for entry in children:
            target = (
                Path(entry.path).resolve()
                if entry.is_symlink()
                else parent / entry.name
            )
            if target in seen or parent.is_relative_to(target):
                continue
            seen.add(target)
            names.append(entry.name)
        return names

    @staticmethod
    def event_files(
        directory: Path,
        entries: list[os.DirEntry[str]],
        *,
        mountpoint: Path | None = None,
    ) -> list[Path]:
        """Return the paths of the event files among the entries of a directory."""
        return [
            directory / entry.name
            for entry in entries
            if fnmatchcase(entry.name, TENSORBOARD_EVENT_GLOB)
            and (
                _Paths.is_file(entry)
                if mountpoint is None
                else _Paths._event_source_exists(directory / entry.name, mountpoint)
            )
        ]

    @staticmethod
    def _event_source_exists(path: Path, mountpoint: Path) -> bool:
        try:
            EventSource.stat(path, mountpoint=mountpoint)
        except OSError:
            return False
        return True

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
        return tuple(normalized[1:].split("/"))

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
            children = node["children"]
            child = children.get(part)
            if child is None:
                child = children[part] = _DirectoryNode()
            node = child
        return node

    @staticmethod
    def tag_dir_parts(
        tag: str, *, name_map: dict[str, str] | None = None
    ) -> tuple[str, ...]:
        """Preserve every slash-separated tag component."""
        return tuple(
            _Paths.safe_component(part, name_map=name_map) for part in tag.split("/")
        )

    @staticmethod
    def safe_component(name: str, *, name_map: dict[str, str] | None = None) -> str:
        """Encode one component and record its original name when shortened."""
        if not name:
            return "~empty"
        if name in {".", "..", NAME_MAP}:
            return "%2E" + name[1:].replace(".", "%2E")
        if (
            len(name) <= 255
            and name.isascii()
            and name.isprintable()
            and not ("\\" in name or "/" in name or "%" in name or "~" in name)
        ):
            return name
        encoded = "".join(
            f"%{ord(ch):02X}"
            if ch in _UNSAFE_PATH_CHARS
            else f"%{ord(ch) - 0xDC00:02X}"
            if 0xDC80 <= ord(ch) <= 0xDCFF
            else ch
            for ch in name
        )
        return _Paths.fit(encoded, name, name_map)

    @staticmethod
    def fit(encoded: str, original: str, name_map: dict[str, str] | None) -> str:
        """Shorten a component to 255 bytes while retaining its extension."""
        if len(encoded) <= 63:
            return encoded
        data = encoded.encode("utf-8")
        if len(data) <= 255:
            return encoded
        stem, dot, extension = encoded.rpartition(".")
        if not dot or len(extension) > 8:
            stem, dot, extension = encoded, "", ""
        tail = f"~{xxhash.xxh3_128_hexdigest(data)}{dot}{extension}"
        budget = 255 - len(tail.encode())
        shortened = stem.encode()[:budget].decode("utf-8", "ignore") + tail
        if (
            name_map is not None
            and name_map.setdefault(shortened, original) != original
        ):
            raise ValueError(f"Conflicting originals for shortened name {shortened!r}")
        return shortened

    @staticmethod
    def decode_component(
        component: str, *, name_map: dict[str, str] | None = None
    ) -> str:
        """Recover an original component from its virtual name and optional map."""
        if name_map is not None and component in name_map:
            return name_map[component]
        base, marker, suffix = component.rpartition("~")
        if marker and suffix.isascii() and suffix.isdecimal():
            component = base
            if name_map is not None and component in name_map:
                return name_map[component]
        if component == "~empty":
            return ""
        return unquote(component, errors="surrogateescape")

    @staticmethod
    def step_name(step: object, digits: int) -> str:
        """Format a TensorBoard step for a virtual filename."""
        if type(step) is int:
            return f"{step:0{digits}d}"
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
