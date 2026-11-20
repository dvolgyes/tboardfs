"""Give each virtual node a file time that comes from its data."""

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypeGuard

from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.run_listing import RunListing

type _Directory = dict[str, Any]
type _Stamp = tuple[_Directory, _Directory | None, float | None, list[_Directory]]


class FileTimes:
    """Read and set the file times of virtual nodes.

    A file has the wall time of its newest event, and a directory the newest
    time below it, so a time changes only when the data changes. tar, for
    example, warns when a time changes while it reads a file.
    """

    @staticmethod
    def of(node: LeafValue) -> float:
        """Return the file time of a node, or 0 for an event time that is not valid.

        An event wall time can be NaN, infinite, negative, or after the year
        2242, which a file time cannot hold.
        """
        if isinstance(node, BinaryEntry):
            value = float(node.wall_time)
        elif isinstance(node, LeafSpec):
            value = float(node.mtime)
        else:
            value = float(node.get("mtime", 0.0))
        return value if math.isfinite(value) and 0 <= value < 1 << 33 else 0.0

    @staticmethod
    def stamp_dirs(tree: dict[str, Any]) -> None:
        """Give each directory of a new tree that has no time the newest time below it.

        A directory without files below it gets the time of the directory
        above it. A directory that has a time keeps it, so a shared subtree
        that has its times is not changed.
        """
        if FileTimes.newest(tree) is None:
            FileTimes.fill(tree, 0.0)

    @staticmethod
    def newest(node: LeafValue) -> float | None:
        """Stamp directories with data and return their newest time."""
        if not _unstamped_dir(node):
            return FileTimes.of(node)
        stamps = FileTimes._plan_stamps(node)
        for directory, _, direct_time, children in reversed(stamps):
            newest = FileTimes._child_max(direct_time, children)
            if newest is not None:
                directory["mtime"] = newest
        if "mtime" not in node:
            return None
        for directory, parent, _, _ in stamps:
            if parent is not None and "mtime" not in directory:
                directory["mtime"] = parent["mtime"]
        return FileTimes.of(node)

    @staticmethod
    def fill(node: dict[str, Any], above: float) -> None:
        """Give the directories without a time the time of the directory above."""
        pending = [node]
        while pending:
            directory = pending.pop()
            if "mtime" in directory:
                continue
            directory["mtime"] = above
            pending.extend(
                child for child in _child_values(directory) if _unstamped_dir(child)
            )

    @staticmethod
    def _plan_stamps(root: _Directory) -> list[_Stamp]:
        """Scan children once and keep only directory links and known maxima."""
        pending: list[tuple[_Directory, _Directory | None]] = [(root, None)]
        stamps: list[_Stamp] = []
        while pending:
            directory, parent = pending.pop()
            newest, children = FileTimes._scan_children(directory)
            stamps.append((directory, parent, newest, children))
            pending.extend((child, directory) for child in children)
        return stamps

    @staticmethod
    def _scan_children(directory: _Directory) -> tuple[float | None, list[_Directory]]:
        """Combine known direct times and defer unstamped child directories."""
        newest: float | None = None
        children: list[_Directory] = []
        for child in _child_values(directory):
            if (
                isinstance(child, dict)
                and child["type"] == "dir"
                and "mtime" not in child
            ):
                children.append(child)
                continue
            time = FileTimes.of(child)
            if newest is None or time > newest:
                newest = time
        return newest, children

    @staticmethod
    def _child_max(newest: float | None, children: list[_Directory]) -> float | None:
        """Include the now-known times of deferred child directories."""
        for child in children:
            time: float | None = child.get("mtime")
            if time is not None and (newest is None or time > newest):
                newest = time
        return newest


def _child_values(node: dict[str, Any]) -> Iterable[LeafValue]:
    children: Mapping[str, LeafValue] = node["children"]
    if isinstance(children, RunListing):
        return children.entries.values()
    return children.values()


def _unstamped_dir(value: LeafValue) -> TypeGuard[_Directory]:
    """Identify directories whose children still need a time scan."""
    return isinstance(value, dict) and value["type"] == "dir" and "mtime" not in value
