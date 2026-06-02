from pathlib import Path
import posixpath
from typing import Any

import numpy as np

from tboardfs.constants import TENSORBOARD_EVENT_GLOB


def find_tensorboard_files(directory: str | Path) -> list[Path]:
    """Return TensorBoard event files below a directory, recursively."""
    root = Path(directory)
    return sorted(path for path in root.rglob(TENSORBOARD_EVENT_GLOB) if path.is_file())


class _Paths:
    """Virtual path helpers for TensorBoard run and tag names."""

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
    def tag_parts(tag: str, fmt: str) -> tuple[str, ...]:
        """Return virtual path parts for a tagged file with an extension."""
        stem_parts = _Paths.tag_dir_parts(tag)
        return (*stem_parts[:-1], f"{stem_parts[-1]}.{fmt}")

    @staticmethod
    def tag_dir_parts(tag: str) -> tuple[str, ...]:
        """Preserve TensorBoard slash hierarchy in a virtual path."""
        return tuple(part for part in tag.split("/") if part) or ("untagged",)

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
    def split_ext(rel: str) -> tuple[str, str]:
        """Split a virtual relative path into stem and extension."""
        parent = posixpath.dirname(rel)
        name = posixpath.basename(rel)
        stem, dot, ext = name.rpartition(".")
        if not dot:
            return rel, ""
        return (posixpath.join(parent, stem) if parent else stem), ext.lower()

    @staticmethod
    def normalize_formats(
        formats: str | list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize comma-separated scalar export format names."""
        items = formats.split(",") if isinstance(formats, str) else list(formats)
        normalized = []
        for item in items:
            fmt = item.strip().lower().lstrip(".")
            if fmt and fmt not in normalized:
                normalized.append(fmt)
        return tuple(normalized)
