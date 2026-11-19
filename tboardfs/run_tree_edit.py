"""Apply a batch of leaf changes without mutating a published run tree."""

from dataclasses import dataclass, field
from typing import Any

from tboardfs.constants import FIXED_TABS
from tboardfs.file_times import FileTimes
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.paths import _DirectoryNode


type _LogicalFile = tuple[tuple[str, ...], str | None, str, bool, str, str]
_IMPLICIT_PRODUCERS = frozenset(("binary", "scalar"))


class _RunListing(dict[str, Any]):
    """Directory children with separate provenance for their direct files."""

    __slots__: tuple[str, ...] = ("logical",)

    logical: dict[str, _LogicalFile]

    def __init__(
        self,
        children: dict[str, Any],
        *,
        logical: dict[str, _LogicalFile] | None = None,
    ) -> None:
        super().__init__(children)
        self.logical = {} if logical is None else logical

    def copy(self) -> "_RunListing":
        """Copy the children and their file provenance."""
        return _RunListing(self, logical=self.logical.copy())


@dataclass(frozen=True, slots=True)
class _RunLeaf:
    """A candidate leaf and the logical identity of its producer."""

    node: dict[str, Any]
    logical: _LogicalFile

    def replaces(self, node: dict[str, Any], logical: _LogicalFile | None) -> bool:
        """Compare recorded provenance, or infer it from an ordinary producer."""
        if self.node["type"] != node["type"]:
            return False
        if logical is not None:
            return self.logical == logical
        root, tag, name, _, _, producer = self.logical
        if producer == "scalar":
            return node.get("format") == name
        if producer == "binary":
            entry = node.get("entry")
            return (
                isinstance(entry, BinaryEntry)
                and entry.tag == tag
                and (entry.kind,) == root
            )
        return False


@dataclass(slots=True)
class _TreeEdits:
    """An edit trie; each touched directory is copied at most once."""

    children: "dict[str, _TreeEdits | _RunLeaf]" = field(default_factory=dict)
    explicit_leaves: bool = False

    def add(self, path: tuple[str, ...], leaf: _RunLeaf) -> bool:
        """Queue a leaf unless another edit conflicts with its identity or path."""
        edits = self
        for part in path[:-1]:
            child = edits.children.get(part)
            if child is None:
                child = edits.children[part] = _TreeEdits()
            elif isinstance(child, _RunLeaf):
                return False
            edits = child
        child = edits.children.get(path[-1])
        if child is not None and (
            isinstance(child, _TreeEdits)
            or child.logical != leaf.logical
            or child.node["type"] != leaf.node["type"]
        ):
            return False
        edits.children[path[-1]] = leaf
        edits.explicit_leaves |= leaf.logical[-1] not in _IMPLICIT_PRODUCERS
        return True

    def apply(self, tree: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any] | None:
        """Copy changed paths, then let empty fixed tabs inherit the new run time."""
        updated = self._apply(tree)
        if updated is None or not self._install_raw(updated, raw):
            return None
        for tab in FIXED_TABS:
            child = updated["children"].get(tab)
            if (
                child is not None
                and child["type"] == "dir"
                and not child["children"]
                and child.get("mtime") != updated["mtime"]
            ):
                empty = _DirectoryNode()
                empty.update(child, mtime=updated["mtime"])
                updated["children"][tab] = empty
        return updated

    @staticmethod
    def _install_raw(tree: dict[str, Any], raw: dict[str, Any]) -> bool:
        children = tree["children"]
        prior = children.get("raw")
        if prior is not None and (
            prior["type"] != "dir" or (prior.get("view") != "raw" and prior["children"])
        ):
            return False
        raw["mtime"] = 0.0 if raw["children"] else tree["mtime"]
        children["raw"] = raw
        return True

    def _apply(self, old: dict[str, Any] | None) -> dict[str, Any] | None:
        if old is not None and (
            old["type"] != "dir" or not isinstance(old["children"], dict)
        ):
            return None
        updated, children = self._copy_directory(old)
        time = self._apply_children(children, _contribution(old))
        if time is None:
            return None
        updated["mtime"] = time
        return updated

    def _copy_directory(
        self, old: dict[str, Any] | None
    ) -> tuple[_DirectoryNode, dict[str, Any]]:
        updated = _DirectoryNode()
        old_children = old["children"] if old is not None else updated["children"]
        if isinstance(old_children, _RunListing):
            children: dict[str, Any] = old_children.copy()
        elif self.explicit_leaves:
            children = _RunListing(old_children)
        else:
            children = old_children.copy() if old is not None else old_children
        if old is not None:
            updated.update(old)
        updated["children"] = children
        return updated, children

    def _apply_children(
        self, children: dict[str, Any], old_time: float
    ) -> float | None:
        identities = children.logical if isinstance(children, _RunListing) else None
        new_max, lost_max = 0.0, False
        for name, edit in self.children.items():
            prior = children.get(name)
            if isinstance(edit, _TreeEdits):
                new = edit._apply(prior)
            else:
                new = _replace_leaf(name, prior, edit, identities)
            if new is None:
                return None
            children[name] = new
            time = FileTimes.of(new)
            new_max = max(new_max, time)
            if time < old_time and _contribution(prior) == old_time:
                lost_max = True
        if lost_max and new_max < old_time:
            return max(_contribution(child) for child in children.values())
        return max(old_time, new_max)


def _replace_leaf(
    name: str,
    prior: dict[str, Any] | None,
    edit: _RunLeaf,
    identities: dict[str, _LogicalFile] | None,
) -> dict[str, Any] | None:
    logical = identities.get(name) if identities is not None else None
    if prior is not None and not edit.replaces(prior, logical):
        return None
    if identities is not None and edit.logical[-1] not in _IMPLICIT_PRODUCERS:
        identities[name] = edit.logical
    return edit.node


def _contribution(node: dict[str, Any] | None) -> float:
    """Only files and nonempty directories contribute their time to their parent."""
    if node is None or (node["type"] == "dir" and not node["children"]):
        return 0.0
    return FileTimes.of(node)
