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
        self._separate_tensor_payloads()
        tag_names = dict.fromkeys(
            file.tag for file in self.files if file.tag is not None
        )
        blocked: dict[tuple[str, ...], str] = {}
        while True:
            tags = TagPaths(tag_names, blocked, self.name_map)
            paths = [file.path(tags, self.name_map) for file in self.files]
            leaves = set(paths)
            found = self.blocked_dirs(paths, leaves)
            if found.keys() <= blocked.keys():
                break
            blocked |= found
        tags.log(run)
        self._insert_files(run_node, paths, leaves)
        if self.name_map:
            run_node["children"][NAME_MAP] = {
                "type": "name_map",
                "data": self.name_map,
            }

    def _separate_tensor_payloads(self) -> None:
        """Keep embedded bytes next to a generated tensor view with the same filename."""
        payloads = [
            f
            for f in self.files
            if f.root == ("tensors",) and f.node["type"] == "binary"
        ]
        if not payloads:
            return
        views = {
            (f.tag, f.name)
            for f in self.files
            if f.root == ("tensors",) and f.node["type"] != "binary"
        }
        for payload in payloads:
            if (payload.tag, payload.name) in views:
                stem, ext = payload.name.rsplit(".", 1)
                payload.name = f"{stem}.embedded.{ext}"

    def _insert_files(
        self,
        run_node: dict[str, Any],
        paths: list[tuple[str, ...]],
        leaves: set[tuple[str, ...]],
    ) -> None:
        directories: dict[tuple[str, ...], dict[str, Any] | None] = {}
        for path in leaves:
            parent = path[:-1]
            while parent and parent not in directories:
                directories[parent] = None
                parent = parent[:-1]
        collisions: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {
            path: {} for path in leaves & directories.keys()
        }
        for path, file in zip(paths, self.files, strict=True):
            if file.identity:
                collisions.setdefault(path, {})
        for path, file in zip(paths, self.files, strict=True):
            sources = collisions.get(path)
            if sources is None:
                parent = path[:-1]
                children = directories.get(parent)
                if children is None:
                    node = _Paths.mkdirs(run_node, parent)
                    children = directories[parent] = node["children"]
                children[path[-1]] = file.node
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
        self, paths: list[tuple[str, ...]], leaves: set[tuple[str, ...]]
    ) -> dict[tuple[str, ...], str]:
        """Return the tag directories that are also files, below their roots.

        Each value is the path of the file, for the warning.
        """
        found: dict[tuple[str, ...], str] = {}
        checked: set[tuple[tuple[str, ...], str, bool]] = set()
        for path, file in zip(paths, self.files, strict=True):
            if file.tag is None:
                continue
            scope = (file.root, file.tag, file.named)
            if scope in checked:
                continue
            checked.add(scope)
            for end in range(len(file.root) + 1, len(path)):
                prefix = path[:end]
                if prefix in leaves:
                    found.setdefault(prefix[len(file.root) :], "/".join(prefix))
        return found
