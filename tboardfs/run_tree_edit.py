"""Apply a batch of leaf changes without mutating a published run tree."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from tboardfs import run_listing
from tboardfs.constants import FIXED_TABS
from tboardfs.file_times import FileTimes
from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.packed_leaf import PackedLeaf
from tboardfs.paths import _DirectoryNode
from tboardfs.run_listing import RunListing

_IMPLICIT_PRODUCERS = frozenset(("binary", "scalar"))


@dataclass(frozen=True, slots=True)
class _RunLeaf:
    """A candidate leaf and the logical identity of its producer."""

    node: LeafValue
    logical: run_listing._LogicalFile

    def replaces(
        self, node: LeafValue, logical: run_listing._LogicalFile | None
    ) -> bool:
        """Compare recorded provenance, or infer it from an ordinary producer."""
        if LeafSpec.kind_of(self.node) != LeafSpec.kind_of(node):
            return False
        if logical is not None:
            return self.logical == logical
        root, tag, name, _, _, producer = self.logical
        if producer == "scalar":
            if isinstance(node, LeafSpec):
                return node.format == name
            return isinstance(node, dict) and node.get("format") == name
        if producer == "binary":
            entry = node.get("entry") if isinstance(node, dict) else node
            if isinstance(entry, PackedLeaf):
                entry = entry.entry()
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
            or LeafSpec.kind_of(child.node) != LeafSpec.kind_of(leaf.node)
        ):
            return False
        edits.children[path[-1]] = leaf
        return True

    def apply(self, tree: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any] | None:
        """Copy changed paths, then let empty fixed tabs inherit the new run time."""
        updated = self._apply(tree)
        if updated is None or not self._install_raw(updated, raw):
            return None
        children: RunListing = updated["children"]
        for tab in FIXED_TABS:
            child = children.stored(tab)
            if (
                isinstance(child, dict)
                and child["type"] == "dir"
                and not child["children"]
                and child.get("mtime") != updated["mtime"]
            ):
                empty = _DirectoryNode()
                empty.update(child, mtime=updated["mtime"])
                children[tab] = empty
        return updated

    @staticmethod
    def _install_raw(tree: dict[str, Any], raw: dict[str, Any]) -> bool:
        children: RunListing = tree["children"]
        prior = children.stored("raw")
        if prior is not None and (
            not isinstance(prior, dict)
            or prior["type"] != "dir"
            or (prior.get("view") != "raw" and prior["children"])
        ):
            return False
        raw["mtime"] = 0.0 if raw["children"] else tree["mtime"]
        children["raw"] = raw
        return True

    def _apply(self, old: LeafValue | None) -> dict[str, Any] | None:
        root = self._start("", old)
        if root is None:
            return None
        stack = [root]
        while True:
            frame = stack[-1]
            item = next(frame.pending, None)
            if item is not None and not self._apply_child(
                frame, item[0], item[1], stack
            ):
                return None
            if item is not None:
                continue
            completed = frame.finish()
            stack.pop()
            if not stack:
                return completed
            stack[-1].store(frame.name, frame.old, completed)

    def _start(self, name: str, old: LeafValue | None) -> "_DirectoryEdit | None":
        if old is not None and (
            not isinstance(old, dict)
            or old["type"] != "dir"
            or not isinstance(old["children"], (RunListing, dict))
        ):
            return None
        updated, children = self._copy_directory(old)
        return _DirectoryEdit(
            name,
            old,
            updated,
            children,
            iter(self.children.items()),
            _contribution(old),
        )

    @staticmethod
    def _apply_child(
        frame: "_DirectoryEdit",
        name: str,
        edit: "_TreeEdits | _RunLeaf",
        stack: "list[_DirectoryEdit]",
    ) -> bool:
        prior = frame.children.stored(name)
        if isinstance(edit, _TreeEdits):
            child = edit._start(name, prior)
            if child is None:
                return False
            stack.append(child)
            return True
        new = _replace_leaf(name, prior, edit, frame.children.logical)
        if new is None:
            return False
        frame.store(name, prior, new)
        return True

    def _copy_directory(
        self, old: dict[str, Any] | None
    ) -> tuple[_DirectoryNode, RunListing]:
        updated = _DirectoryNode()
        old_children: RunListing | dict[str, LeafValue] | None = (
            old["children"] if old is not None else None
        )
        if isinstance(old_children, RunListing):
            children = old_children.copy()
        elif old_children is not None:
            children = RunListing(old_children.copy())
        else:
            children = RunListing({})
        if old is not None:
            updated.update(old)
        updated["children"] = children
        return updated, children


@dataclass(slots=True)
class _DirectoryEdit:
    """Unpublished directory copy with suspended edits and accumulated time."""

    name: str
    old: LeafValue | None
    updated: _DirectoryNode
    children: RunListing
    pending: Iterator[tuple[str, _TreeEdits | _RunLeaf]]
    old_time: float
    new_max: float = 0.0
    lost_max: bool = False

    def store(self, name: str, prior: LeafValue | None, new: LeafValue) -> None:
        """Install a changed child and track whether its old time must be replaced."""
        self.children.set_leaf(name, new)
        time = FileTimes.of(new)
        self.new_max = max(self.new_max, time)
        self.lost_max = self.lost_max or (
            time < self.old_time and _contribution(prior) == self.old_time
        )

    def finish(self) -> _DirectoryNode:
        """Stamp the directory after all its edits are complete."""
        time = max(self.old_time, self.new_max)
        if self.lost_max and self.new_max < self.old_time:
            time = max(_contribution(child) for child in self.children.entries.values())
        self.updated["mtime"] = time
        return self.updated


def _replace_leaf(
    name: str,
    prior: LeafValue | None,
    edit: _RunLeaf,
    identities: dict[str, run_listing._LogicalFile],
) -> LeafValue | None:
    if prior is not None and not edit.replaces(prior, identities.get(name)):
        return None
    if edit.logical[-1] not in _IMPLICIT_PRODUCERS:
        identities[name] = edit.logical
    return edit.node


def _contribution(node: LeafValue | None) -> float:
    """Only files and nonempty directories contribute their time to their parent."""
    if node is None or (
        isinstance(node, dict) and node["type"] == "dir" and not node["children"]
    ):
        return 0.0
    return FileTimes.of(node)
