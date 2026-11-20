"""Place the directory trees of all runs in one virtual tree."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from tboardfs.paths import NAME_MAP, _DirectoryNode, _Paths
from tboardfs.run_listing import RunListing


class RunPlaces:
    """Place the directory trees of all runs in one virtual tree.

    A directory holds the entries of one run and the folders of the runs
    below it. A run folder can have the name of an entry of the run above it,
    such as the run exp/scalars below the run exp, which has the tab scalars.
    Such a folder gets "~2" (then "~3", and so on, but not the name of another
    folder on disk) on that part, and the runs below it follow it.

    Attributes:
        renamed: by run path prefix, its virtual path with a suffix, at the
            last build, so a rename is logged once.
        disk_parts: by virtual folder path, its run path prefix on disk, at
            the last build.
        flat: the last build has only direct, identity-mapped run folders
            and no entries of its own.
    """

    renamed: dict[tuple[str, ...], tuple[str, ...]]
    disk_parts: dict[tuple[str, ...], tuple[str, ...]]
    flat: bool

    def __init__(self) -> None:
        self.renamed = {}
        self.disk_parts = {(): ()}
        self.flat = False

    def build(
        self, run_trees: Mapping[tuple[str, ...], dict[str, Any]]
    ) -> dict[str, Any]:
        """Return the virtual tree of all runs.

        Args:
            run_trees: the directory tree of each run, by run path; they are
                not changed.
        """
        root = _DirectoryNode()
        maps: dict[tuple[str, ...], dict[str, str]] = {}
        dirs: dict[tuple[str, ...], dict[str, Any]] = {(): root}
        paths: dict[tuple[str, ...], tuple[str, ...]] = {(): ()}
        renamed: dict[tuple[str, ...], tuple[str, ...]] = {}
        # A run sorts before the runs below it, so its entries are in its
        # directory before a folder below it gets a name.
        runs = sorted(run_trees)
        for index, run_parts in enumerate(runs):
            run = run_trees[run_parts]
            children = run["children"]
            share = self._share_run(run, runs, index)
            for end in range(1, len(run_parts) + 1):
                prefix = run_parts[:end]
                if prefix in dirs:
                    continue
                parent = dirs[prefix[:-1]]
                names = maps.setdefault(prefix[:-1], {})
                component = _Paths.safe_component(prefix[-1], name_map=names)
                name = RunPlaces.free_name(component, parent)
                node = run if share and end == len(run_parts) else _DirectoryNode()
                dirs[prefix] = parent["children"][name] = node
                paths[prefix] = (*paths[prefix[:-1]], name)
                if name != component:
                    renamed[prefix] = paths[prefix]
            if not share:
                dirs[run_parts]["children"] = (
                    children.copy()
                    if isinstance(children, RunListing)
                    else dict(children)
                )
        for parts, names in maps.items():
            if names:
                children = dirs[parts]["children"]
                existing = children.get(NAME_MAP, {}).get("data", {})
                children[NAME_MAP] = {"type": "name_map", "data": existing | names}
        self.log(renamed)
        self.disk_parts = {path: prefix for prefix, path in paths.items()}
        self.flat = len(root["children"]) + 1 == len(self.disk_parts) and all(
            not path or (len(path) == 1 and path == parts)
            for path, parts in self.disk_parts.items()
        )
        return root

    @staticmethod
    def _share_run(
        run: dict[str, Any], runs: list[tuple[str, ...]], index: int
    ) -> bool:
        """Share stamped, populated folders only where placement adds nothing."""
        parts = runs[index]
        return (
            bool(parts)
            and bool(run["children"])
            and isinstance(run, _DirectoryNode)
            and "mtime" in run
            and (index + 1 == len(runs) or runs[index + 1][: len(parts)] != parts)
        )

    def update(
        self,
        root: dict[str, Any],
        changes: Mapping[tuple[str, ...], dict[str, Any] | None],
    ) -> _DirectoryNode:
        """Replace branches in a published flat tree with the same folder names.

        Changes are keyed by existing top-level run paths. A replacement
        is a stamped run tree or None if the folder has no run files.
        Empty folders inherit the new root time instead of contributing to it.
        """
        children: dict[str, dict[str, Any] | None] = root["children"].copy()
        for (name,), run in changes.items():
            if run is not None and run["children"]:
                children[name] = run
            elif (previous := children[name]) is not None and previous["children"]:
                children[name] = None

        newest = max(
            (
                child["mtime"]
                for child in children.values()
                if child is not None and child["children"]
            ),
            default=0.0,
        )
        empty = None
        for name, child in children.items():
            if child is None or (not child["children"] and child["mtime"] != newest):
                if empty is None:
                    empty = _DirectoryNode()
                    empty["mtime"] = newest
                children[name] = empty

        updated = _DirectoryNode()
        updated["children"] = children
        updated["mtime"] = newest
        if isinstance(root, _DirectoryNode):
            updated._sorted_names = root._sorted_names
        return updated

    @staticmethod
    def free_name(
        component: str,
        parent: dict[str, Any],
    ) -> str:
        """Return the name of a run folder that no entry of its parent has."""
        name, number = component, 1
        while name in parent["children"]:
            number += 1
            name = f"{component}~{number}"
        return name

    def log(self, renamed: dict[tuple[str, ...], tuple[str, ...]]) -> None:
        """Log each run folder that gets a new suffix, and keep the suffixes."""
        for prefix, path in renamed.items():
            if self.renamed.get(prefix) != path:
                logger.warning(
                    "run folder {!r} is under {!r}: the run {!r} has an entry {!r}",
                    "/".join(prefix),
                    "/".join(path),
                    "/".join(prefix[:-1]) or ".",
                    prefix[-1],
                )
        self.renamed = renamed
