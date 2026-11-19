"""Collect the virtual files of one run, then give each tag a directory that no file has."""

from dataclasses import dataclass
from typing import Any

from tboardfs.model import RunCache
from tboardfs.paths import NAME_MAP, _Paths
from tboardfs.tag_paths import TagPaths


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
    identity: str = ""

    def path(self, tags: TagPaths, name_map: dict[str, str]) -> tuple[str, ...]:
        """Return the file's virtual path, including any generated extension."""
        if self.tag is None:
            return (*self.root, _Paths.fit(self.name, self.name, name_map))
        parts = tags.dir_parts(self.tag)
        if not self.named:
            return (*self.root, *parts, _Paths.fit(self.name, self.name, name_map))
        name = _Paths.fit(
            f"{parts[-1]}.{self.name}",
            f"{self.tag.rsplit('/', 1)[-1]}.{self.name}",
            name_map,
        )
        return (*self.root, *parts[:-1], name)

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
    hides another. A later observation of the same logical file replaces it.

    Attributes:
        files: the files in the order that they were added.
    """

    files: list[_RunFile]
    name_map: dict[str, str]

    def __init__(self) -> None:
        self.files = []
        self.name_map = {}

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

    def add_fixed(
        self, parts: tuple[str, ...], node: dict[str, Any], *, identity: str = ""
    ) -> None:
        """Add a generated file or a sidecar with a distinct source identity."""
        self.files.append(_RunFile(parts[:-1], None, parts[-1], node, False, identity))

    def fill(self, run_node: dict[str, Any], run: RunCache) -> None:
        """Add every file to the directory of a run.

        Args:
            run_node: the run directory to fill.
            run: the run, which keeps its moved tags so a run that grows
                logs each move once.
        """
        tag_names = dict.fromkeys(
            file.tag for file in self.files if file.tag is not None
        )
        blocked: dict[tuple[str, ...], str] = {}
        while True:
            tags = TagPaths(tag_names, blocked, self.name_map)
            paths = [file.path(tags, self.name_map) for file in self.files]
            found = self.blocked_dirs(paths, tags)
            if found.keys() <= blocked.keys():
                break
            blocked |= found
        tags.log(run)
        self._insert_files(run_node, paths)
        if self.name_map:
            run_node["children"][NAME_MAP] = {
                "type": "name_map",
                "data": self.name_map,
            }

    def _insert_files(
        self, run_node: dict[str, Any], paths: list[tuple[str, ...]]
    ) -> None:
        leaves = set(paths)
        directories = {path[:end] for path in leaves for end in range(1, len(path))}
        collisions: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {
            path: {} for path in leaves & directories
        }
        for path, file in zip(paths, self.files, strict=True):
            if file.identity:
                collisions.setdefault(path, {})
        for path, file in zip(paths, self.files, strict=True):
            sources = collisions.get(path)
            if sources is None:
                _Paths.mkdirs(run_node, path[:-1])["children"][path[-1]] = file.node
            else:
                sources[file.identity] = file.node
        for path, sources in collisions.items():
            children = _Paths.mkdirs(run_node, path[:-1])["children"]
            number = 1
            for index, source in enumerate(sorted(sources)):
                resolved = path
                if index or path in directories:
                    while True:
                        number += 1
                        name = _Paths.fit(
                            f"{path[-1]}~{number}",
                            _Paths.decode_component(path[-1], name_map=self.name_map),
                            self.name_map,
                        )
                        resolved = (*path[:-1], name)
                        if resolved not in leaves and resolved not in directories:
                            break
                    leaves.add(resolved)
                children[resolved[-1]] = sources[source]

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
