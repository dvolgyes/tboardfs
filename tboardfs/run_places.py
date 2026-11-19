"""Place the directory trees of all runs in one virtual tree."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from tboardfs.paths import NAME_MAP, _DirectoryNode, _Paths


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
    """

    renamed: dict[tuple[str, ...], tuple[str, ...]]
    disk_parts: dict[tuple[str, ...], tuple[str, ...]]

    def __init__(self) -> None:
        self.renamed = {}
        self.disk_parts = {(): ()}

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
        for run_parts in sorted(run_trees):
            for end in range(1, len(run_parts) + 1):
                prefix = run_parts[:end]
                if prefix in dirs:
                    continue
                parent = dirs[prefix[:-1]]
                names = maps.setdefault(prefix[:-1], {})
                component = _Paths.safe_component(prefix[-1], name_map=names)
                name = RunPlaces.free_name(component, parent)
                dirs[prefix] = _DirectoryNode()
                parent["children"][name] = dirs[prefix]
                paths[prefix] = (*paths[prefix[:-1]], name)
                if name != component:
                    renamed[prefix] = paths[prefix]
            dirs[run_parts]["children"].update(run_trees[run_parts]["children"])
        for parts, names in maps.items():
            if names:
                children = dirs[parts]["children"]
                existing = children.get(NAME_MAP, {}).get("data", {})
                children[NAME_MAP] = {"type": "name_map", "data": existing | names}
        self.log(renamed)
        self.disk_parts = {path: prefix for prefix, path in paths.items()}
        return root

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
