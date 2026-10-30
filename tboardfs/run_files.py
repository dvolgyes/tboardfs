"""Collect the virtual files of one run, then give each tag a directory that no file has."""

from dataclasses import dataclass
import time
from typing import Any

from loguru import logger

from tboardfs.model import RunCache
from tboardfs.paths import _Paths
from tboardfs.tag_paths import TagPaths

# Each round can only move tag directories that a file is in the way of; more
# rounds than this mean that the moved names keep meeting files.
_ROUNDS = 8


@dataclass
class _RunFile:
    """One virtual file: a root, a tag directory and a file name.

    A file without a tag is at root + name. A file named after its tag
    (``named``) is the last part of the tag with ``name`` as its extension.
    """

    root: tuple[str, ...]
    tag: str | None
    name: str
    node: dict[str, Any]
    named: bool

    def path(self, tags: TagPaths) -> tuple[str, ...]:
        """Return the virtual path of the file."""
        if self.tag is None:
            return (*self.root, self.name)
        parts = tags.dir_parts(self.tag)
        if self.named:
            return (*self.root, *parts[:-1], f"{parts[-1]}.{self.name}")
        return (*self.root, *parts, self.name)

    def tag_dir_ends(self, tags: TagPaths) -> range:
        """Return the lengths of the path prefixes that are tag directories."""
        if self.tag is None:
            return range(0)
        count = len(tags.dir_parts(self.tag)) - self.named
        return range(len(self.root) + 1, len(self.root) + count + 1)


class RunFiles:
    """Collect the virtual files of one run and build its directory tree.

    The tree is built only when all files are known. Then a tag directory
    that has the name of a file, such as the tag "a/000001.npy" beside the
    step file 000001.npy of the tag "a", gets "~2" in every tab, and no file
    hides another. A later file at the path of an earlier one replaces it.

    Attributes:
        files: the files in the order that they were added.
    """

    files: list[_RunFile]

    def __init__(self) -> None:
        self.files = []

    def add(
        self, root: tuple[str, ...], tag: str, name: str, node: dict[str, Any]
    ) -> None:
        """Add a file to the directory of a tag below a root, such as a tab."""
        self.files.append(_RunFile(root, tag, name, node, named=False))

    def add_named(
        self, root: tuple[str, ...], tag: str, extension: str, node: dict[str, Any]
    ) -> None:
        """Add a file named after its tag, such as /scalars/<tag>.json."""
        self.files.append(_RunFile(root, tag, extension, node, named=True))

    def add_fixed(self, parts: tuple[str, ...], node: dict[str, Any]) -> None:
        """Add a file at a path without a tag, such as /graphs/graph.pb."""
        self.files.append(_RunFile(parts[:-1], None, parts[-1], node, named=False))

    def fill(self, run_node: dict[str, Any], run: RunCache) -> None:
        """Add every file to the directory of a run.

        Args:
            run_node: the run directory to fill.
            run: the run, which keeps its moved tags so a run that grows
                logs each move once.
        """
        tag_names = {file.tag for file in self.files if file.tag is not None}
        blocked: dict[tuple[str, ...], str] = {}
        for _ in range(_ROUNDS):
            tags = TagPaths(tag_names, blocked)
            paths = [file.path(tags) for file in self.files]
            found = self.blocked_dirs(paths, tags)
            if found.keys() <= blocked.keys():
                break
            blocked |= found
        tags.log(run)
        for path, file in zip(paths, self.files, strict=True):
            self.insert(run_node, path, file.node, run)

    def blocked_dirs(
        self, paths: list[tuple[str, ...]], tags: TagPaths
    ) -> dict[tuple[str, ...], str]:
        """Return the tag directories that are also files, below their roots.

        Each value is the path of the file, for the warning.
        """
        file_paths = set(paths)
        found: dict[tuple[str, ...], str] = {}
        for path, file in zip(paths, self.files, strict=True):
            for end in file.tag_dir_ends(tags):
                if path[:end] in file_paths:
                    found.setdefault(path[len(file.root) : end], "/".join(path[:end]))
        return found

    @staticmethod
    def insert(
        run_node: dict[str, Any],
        path: tuple[str, ...],
        node: dict[str, Any],
        run: RunCache,
    ) -> None:
        """Add one file to a run directory, unless a file and a directory meet.

        That can happen only between paths without tags, such as sidecars,
        which no suffix can separate; the later file is left out.
        """
        parent = run_node
        for end, part in enumerate(path, start=1):
            child = parent["children"].get(part)
            if child is None:
                break
            if (child["type"] == "dir") == (end == len(path)):
                logger.warning(
                    "{!r} in run {!r} is both a file and a directory; "
                    "a file at {!r} is left out",
                    "/".join(path[:end]),
                    "/".join(run.run_parts),
                    "/".join(path),
                )
                return
            parent = child
        node.setdefault("mtime", time.time())
        _Paths.mkdirs(run_node, path[:-1])["children"][path[-1]] = node
