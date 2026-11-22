"""Bounded resolution cache for event leaves in one mount publication."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from errno import ENOENT
from os import strerror
from typing import Any, ClassVar

from tboardfs.file_times import FileTimes
from tboardfs.paths import _Paths
from tboardfs.render_cache import RenderKey

_LIMIT = 32_768
_MAX_PATH_CHARS = 4096

type _Parts = tuple[str, ...]
type _Node = dict[str, Any]
type _Children = Mapping[str, _Node]
type _Location = tuple[_Node, RenderKey]
type _LeafEntry = tuple[_Parts, _Location]
type _DiskParts = dict[_Parts, _Parts]


def _resolve_node(
    tree: _Node, parts: _Parts, path: str
) -> tuple[_Node, _Children | None]:
    """Walk the publication, returning the node and its parent's children."""
    node = tree
    children: _Children | None = None
    for part in parts:
        children = node["children"] if node["type"] == "dir" else None
        if children is None or (child := children.get(part)) is None:
            raise FileNotFoundError(ENOENT, strerror(ENOENT), path)
        node = child
    return node, children


def _cacheable(node: _Node) -> bool:
    return node["type"] not in {"dir", "sidecar", "name_map"}


@dataclass(frozen=True, slots=True)
class _VirtualParent:
    """One unmapped directory's immutable children and their owning run."""

    prefix: str
    parts: _Parts
    owner: _Parts
    children: _Children

    def direct_child_parts(self, path: str) -> _Parts | None:
        """Accept only the canonical spelling of one immediate child."""
        if not path.startswith(self.prefix):
            return None
        name = path[len(self.prefix) :]
        if name and name not in {".", ".."} and "/" not in name:
            return (*self.parts, name)
        return None

    def leaf(self, name: str) -> _Node | None:
        """Use the hinted parent only for ordinary generated leaves."""
        node = self.children.get(name)
        return node if node is not None and _cacheable(node) else None


type _LeafProbe = tuple[_Parts, _LeafEntry | None, _VirtualParent | None]


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

    _entries: OrderedDict[_Parts, _LeafEntry]
    _recent: tuple[str, _LeafProbe] | None
    _parent: _VirtualParent | None

    def __init__(self) -> None:
        self._entries = OrderedDict()
        self._recent = None
        self._parent = None

    def probe(self, path: str) -> _LeafProbe:
        """Resolve the exact repeat, tuple LRU, or canonical sibling candidate."""
        if (recent := self._recent) is not None and path == recent[0]:
            return recent[1]
        self._recent = None
        if len(path) > _MAX_PATH_CHARS:
            return _Paths.path_parts(path), None, None
        parent = self._parent
        parts = parent.direct_child_parts(path) if parent is not None else None
        if parts is None:
            parent = None
            parts = _Paths.path_parts(path)
        entry = self._entries.get(parts)
        if entry is None:
            return parts, None, parent
        self._entries.move_to_end(parts)
        return self._set_recent(path, parts, entry)

    def _set_recent(self, path: str, parts: _Parts, entry: _LeafEntry) -> _LeafProbe:
        probe = parts, entry, None
        self._recent = path, probe
        return probe

    def remember(
        self,
        path: str,
        owner: _Parts,
        node: _Node,
        key: RenderKey,
        *,
        children: _Children | None = None,
        disk_parts: _DiskParts | None = None,
    ) -> _Location:
        """Remember an eligible leaf without retaining its rendered bytes."""
        if len(path) > _MAX_PATH_CHARS or not _cacheable(node):
            return node, key
        result = _CachedLeaf.wrap(node), key
        parts = key[1]
        entry = owner, result
        self._entries[parts] = entry
        self._entries.move_to_end(parts)
        if len(self._entries) > _LIMIT:
            self._entries.popitem(last=False)
        self._set_recent(path, parts, entry)
        if children is not None and disk_parts is not None:
            self._remember_parent(parts[:-1], owner, children, disk_parts)
        return result

    def _remember_parent(
        self, parts: _Parts, owner: _Parts, children: _Children, disk_parts: _DiskParts
    ) -> None:
        """A parent outside the prefix-closed run map has no mapped children."""
        if parts and parts not in disk_parts:
            self._parent = _VirtualParent(
                f"/{'/'.join(parts)}/", parts, owner, children
            )

    def clear(self) -> None:
        """Drop all entries when the mount tree changes."""
        self._entries.clear()
        self._recent = None
        self._parent = None


@dataclass
class _MountTree:
    """Last published root, its leaf cache, and the runs changed since publication."""

    root: dict[str, Any] | None = None
    dirty: set[tuple[str, ...]] = field(default_factory=set)
    generation: int = 0
    leaves: _LeafCache = field(default_factory=_LeafCache, init=False)
