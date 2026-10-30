"""Give each virtual node a file time that comes from its data."""

import math
from typing import Any


class FileTimes:
    """Read and set the file times of virtual nodes.

    A file has the wall time of its newest event, and a directory the newest
    time below it, so a time changes only when the data changes. tar, for
    example, warns when a time changes while it reads a file.
    """

    @staticmethod
    def of(node: dict[str, Any]) -> float:
        """Return the file time of a node, or 0 for an event time that is not valid.

        An event wall time can be NaN, infinite, negative, or after the year
        2242, which a file time cannot hold.
        """
        value = float(node.get("mtime", 0.0))
        return value if math.isfinite(value) and 0 <= value < 1 << 33 else 0.0

    @staticmethod
    def stamp_dirs(tree: dict[str, Any]) -> None:
        """Give each directory of a new tree that has no time the newest time below it.

        A directory without files below it gets the time of the directory
        above it. A directory that has a time keeps it, so a shared subtree
        that has its times is not changed.
        """
        FileTimes.newest(tree)
        FileTimes.fill(tree, FileTimes.of(tree))

    @staticmethod
    def newest(node: dict[str, Any]) -> float | None:
        """Set and return the newest time below a directory, or None without files."""
        if node["type"] != "dir" or "mtime" in node:
            return FileTimes.of(node)
        times = [FileTimes.newest(child) for child in node["children"].values()]
        newest = max((time for time in times if time is not None), default=None)
        if newest is not None:
            node["mtime"] = newest
        return newest

    @staticmethod
    def fill(node: dict[str, Any], above: float) -> None:
        """Give the directories without a time the time of the directory above."""
        node.setdefault("mtime", above)
        for child in node["children"].values():
            if child["type"] == "dir":
                FileTimes.fill(child, node["mtime"])
