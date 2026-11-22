"""Identify source-backed nodes and enforce the mounted source path guard."""

import os
from collections.abc import Callable
from errno import EIO, ENOENT
from pathlib import Path
from stat import S_ISREG
from typing import Any

from tboardfs._directory import _GuardPlans, _OpenHow
from tboardfs.errors import fuse_error
from tboardfs.event_source import EventSource


class _NodeSource:
    """Apply the source checks appropriate to a virtual node operation."""

    @staticmethod
    def path(node: dict[str, Any]) -> Path | None:
        """Return the source read by a raw, lazy tensor, or entry-backed node."""
        entry = node.get("entry")
        source: Path | None = node.get("source") if entry is None else entry.source_path
        return source

    @classmethod
    def require_allowed(
        cls,
        node: dict[str, Any],
        allowed: Callable[[Path], bool],
        *,
        checked_paths: frozenset[Path] | None = None,
    ) -> None:
        """Require a fresh source path check unless this lookup already made one."""
        source = cls.path(node)
        if (
            source is not None
            and (checked_paths is None or source not in checked_paths)
            and not allowed(source)
        ):
            raise fuse_error(ENOENT)

    @classmethod
    def require_readable(
        cls,
        node: dict[str, Any],
        *,
        plans: _GuardPlans | None,
        mountpoint: Path | None,
        storage: Path | None,
    ) -> None:
        """Check current read access without rereading the cached payload."""
        source = cls.path(node)
        if source is None:
            return
        try:
            fd = _OpenHow.open_source(source, plans, mountpoint)
            if fd is not None:
                try:
                    if not S_ISREG(os.fstat(fd).st_mode):
                        raise OSError(EIO, "source is not a regular file", source)
                finally:
                    os.close(fd)
                return
            EventSource.open(
                source, mountpoint=mountpoint, storage=storage, buffering=0
            ).close()
        except OSError as error:
            raise fuse_error(ENOENT if error.errno == ENOENT else EIO) from error
