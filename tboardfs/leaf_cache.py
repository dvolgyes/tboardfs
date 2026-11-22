"""Bounded resolution cache for event leaves in one mount publication."""

from collections import OrderedDict
from dataclasses import dataclass, field
from errno import ENOENT
from os import strerror
from typing import Any, ClassVar

from tboardfs.file_times import FileTimes
from tboardfs.render_cache import RenderKey

_LIMIT = 32_768
_MAX_PATH_CHARS = 4096

type _Location = tuple[dict[str, Any], RenderKey]


def _resolve_node(
    tree: dict[str, Any], parts: tuple[str, ...], path: str
) -> dict[str, Any]:
    """Walk the published tree, preserving the original path in lookup errors."""
    node = tree
    for part in parts:
        if node["type"] != "dir" or (child := node["children"].get(part)) is None:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)
        node = child
    return node


class _CachedLeaf(dict[str, Any]):
    """Keep mount-only metadata off the published node."""

    __slots__: ClassVar[tuple[str, ...]] = ("mtime", "size")

    mtime: float
    size: int | None

    def __init__(self, node: dict[str, Any]) -> None:
        super().__init__(node)
        self.mtime = FileTimes.of(node)
        self.size = None

    @staticmethod
    def wrap(node: dict[str, Any]) -> dict[str, Any]:
        """Wrap only generated leaves whose rendered size cannot change on disk."""
        if "source" in node or "entry" in node:
            return node
        kind = node["type"]
        if (kind in {"scalar", "table"} and "series" in node) or (
            kind in {"bytes", "obj", "json", "npy"} and "data" in node
        ):
            return _CachedLeaf(node)
        return node

    @staticmethod
    def time_of(node: dict[str, Any]) -> float:
        """Reuse the generated leaf's timestamp when available."""
        return node.mtime if isinstance(node, _CachedLeaf) else FileTimes.of(node)


class _LeafCache:
    """The mount lock protects cache membership and publication changes."""

    _entries: OrderedDict[tuple[str, ...], tuple[tuple[str, ...], _Location]]

    def __init__(self) -> None:
        self._entries = OrderedDict()

    def get(self, parts: tuple[str, ...]) -> tuple[tuple[str, ...], _Location] | None:
        """Return a recent leaf and the owner that must be refreshed before use."""
        entry = self._entries.get(parts)
        if entry is not None:
            self._entries.move_to_end(parts)
        return entry

    def remember(
        self, path: str, owner: tuple[str, ...], node: dict[str, Any], key: RenderKey
    ) -> _Location:
        """Remember an eligible leaf without retaining its rendered bytes."""
        if len(path) > _MAX_PATH_CHARS or node["type"] in {
            "dir",
            "sidecar",
            "name_map",
        }:
            return node, key
        result = _CachedLeaf.wrap(node), key
        parts = key[1]
        self._entries[parts] = owner, result
        self._entries.move_to_end(parts)
        if len(self._entries) > _LIMIT:
            self._entries.popitem(last=False)
        return result

    def clear(self) -> None:
        """Drop all entries when the mount tree changes."""
        self._entries.clear()


@dataclass
class _MountTree:
    """Last published root, its leaf cache, and the runs changed since publication."""

    root: dict[str, Any] | None = None
    dirty: set[tuple[str, ...]] = field(default_factory=set)
    generation: int = 0
    leaves: _LeafCache = field(default_factory=_LeafCache, init=False)
