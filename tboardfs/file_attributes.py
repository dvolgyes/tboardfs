"""Represent the complete stat attributes of a virtual file or directory."""

import stat
from collections.abc import Callable
from errno import ENOENT
from pathlib import Path
from typing import Any, NamedTuple

from tboardfs.errors import fuse_error
from tboardfs.leaf_cache import _CachedLeaf
from tboardfs.model import BinaryEntry, GraphEntry, SidecarEntry
from tboardfs.node_source import _NodeSource
from tboardfs.open3d_asset import OPEN3D_PLUGIN, Open3DAsset


class _FileAttributes(NamedTuple):
    """Values that determine all exported stat fields."""

    directory: bool
    size: int
    mtime: float
    owner: tuple[int, int]

    def as_dict(self) -> dict[str, int | float]:
        """Expand the resolved values into the attributes expected by mfusepy."""
        uid, gid = self.owner
        return {
            "st_mode": stat.S_IFDIR | 0o555 if self.directory else stat.S_IFREG | 0o444,
            "st_nlink": 2 if self.directory else 1,
            "st_size": self.size,
            "st_blocks": (self.size + 511) // 512,
            "st_ctime": self.mtime,
            "st_mtime": self.mtime,
            "st_atime": self.mtime,
            "st_uid": uid,
            "st_gid": gid,
        }


def _unrendered_size(
    node: dict[str, Any],
    allowed: Callable[[Path], bool],
    checked_paths: frozenset[Path] | None,
) -> int | None:
    """Return the size without rendering, reusing source checks only for index entries."""
    if isinstance(node, _CachedLeaf) and (size := node.size) is not None:
        return size
    kind = node["type"]
    if kind == "raw":
        _NodeSource.require_allowed(node, allowed, checked_paths=checked_paths)
        return int(node["frame"].size)
    if kind in {"binary", "graph"}:
        stored: BinaryEntry | GraphEntry = node["entry"]
        _NodeSource.require_allowed(node, allowed, checked_paths=checked_paths)
        return stored.size
    if kind == "sidecar":
        entry: SidecarEntry = node["entry"]
        if not allowed(entry.source_path):
            raise fuse_error(ENOENT)
        if entry.tab != OPEN3D_PLUGIN:
            return entry.source_path.stat().st_size
        asset = Open3DAsset.stat(entry.source_path)
        if asset is None:
            raise fuse_error(ENOENT)
        return asset.size
    return None
