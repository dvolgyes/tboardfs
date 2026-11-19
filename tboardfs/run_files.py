"""Collect the virtual files of one run, then give each tag a directory that no file has."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from tboardfs.model import RunCache
from tboardfs.paths import NAME_MAP, _Paths
from tboardfs.run_tree_edit import (
    _IMPLICIT_PRODUCERS,
    _LogicalFile,
    _RunLeaf,
    _RunListing,
    _TreeEdits,
)
from tboardfs.tag_paths import TagPaths


type _Path = tuple[str, ...]
type _TagScope = tuple[_Path, str, bool]


@dataclass(frozen=True, slots=True)
class _PathHistory:
    """Record cold-only resolutions that a local patch cannot freely change."""

    directories: frozenset[_Path]
    scopes: frozenset[_TagScope]
    renamed: frozenset[str]
    unstable: bool

    def permits(self, file: "_RunFile", path: _Path) -> bool:
        """Reject edits that could change a prior collision resolution."""
        if path in self.directories:
            return False
        if file.tag is None:
            return True
        return file.tag not in self.renamed and file.scope() in self.scopes


class _RunChildren(_RunListing):
    """A run listing with exceptional path history kept out of virtual nodes."""

    __slots__: tuple[str, ...] = ("history",)

    history: _PathHistory

    def __init__(self, children: dict[str, Any], history: _PathHistory) -> None:
        super().__init__(
            children,
            logical=children.logical if isinstance(children, _RunListing) else None,
        )
        self.history = history

    def copy(self) -> "_RunChildren":
        """Copy the listing and retain its immutable path history."""
        copied = _RunChildren(self, self.history)
        copied.logical = self.logical.copy()
        return copied


@dataclass(slots=True)
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
    producer: str = ""

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

    def logical(self) -> _LogicalFile:
        """Give the cold and append publishers the same source identity."""
        return (
            self.root,
            self.tag,
            self.name,
            self.named,
            self.identity,
            self.producer,
        )

    def scope(self) -> _TagScope | None:
        """Return the tag-directory scope, or None for a fixed file."""
        return None if self.tag is None else (self.root, self.tag, self.named)

    def nonstandard_tensor(self) -> bool:
        """Identify tensor paths outside ordinary tagged step files."""
        return self.root == ("tensors",) and (self.tag is None or self.named)

    def embedded_tensor(self) -> bool:
        """Identify an embedded filename that a cold build can generate."""
        return (
            self.root == ("tensors",)
            and self.node["type"] == "binary"
            and ".embedded." in self.name
        )

    def patchable_name(self, path: _Path) -> bool:
        """Return whether this path avoids namespaces that need a cold build."""
        return (
            path[0] not in ("raw", NAME_MAP)
            and not self.nonstandard_tensor()
            and not self.embedded_tensor()
        )

    def store(self, directory: dict[str, Any], name: str) -> None:
        """Insert the original leaf, recording only non-inferable provenance."""
        children = directory["children"]
        if self.producer in _IMPLICIT_PRODUCERS:
            if isinstance(children, _RunListing):
                children.logical.pop(name, None)
        else:
            if not isinstance(children, _RunListing):
                children = directory["children"] = _RunListing(children)
            children.logical[name] = self.logical()
        children[name] = self.node


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
        self,
        root: tuple[str, ...],
        tag: str,
        name: str,
        node: dict[str, Any],
        *,
        producer: str = "",
    ) -> None:
        """Add a file to the directory of a tag below a root, such as a tab."""
        self.files.append(
            _RunFile(root, tag, name, node, named=False, producer=producer)
        )

    def add_named(
        self,
        root: tuple[str, ...],
        tag: str,
        extension: str,
        node: dict[str, Any],
        *,
        producer: str = "",
    ) -> None:
        """Add a file named after its tag, such as /scalars/<tag>.json."""
        self.files.append(
            _RunFile(root, tag, extension, node, named=True, producer=producer)
        )

    def add_fixed(
        self, parts: tuple[str, ...], node: dict[str, Any], *, identity: str = ""
    ) -> None:
        """Add a generated file or a sidecar with a distinct source identity."""
        self.files.append(_RunFile(parts[:-1], None, parts[-1], node, False, identity))

    def patch(
        self, tree: dict[str, Any], run: RunCache, *, raw: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Publish changed logical files, or decline if cold resolution can differ.

        The caller proves append provenance and projects the candidate cold
        producers in order. The fresh raw tab replaces its old snapshot.
        """
        children = tree["children"]
        if "mtime" not in tree or self.name_map or NAME_MAP in children:
            return None
        history = children.history if isinstance(children, _RunChildren) else None
        if history is None and run.renamed_tags:
            return None
        if history is not None and history.unstable:
            return None
        try:
            edits = self._patch_edits(history)
        except ValueError:
            return None
        return None if edits is None else edits.apply(tree, raw)

    def _patch_edits(self, history: _PathHistory | None) -> _TreeEdits | None:
        tags = TagPaths(self._tag_names(), {}, self.name_map)
        edits = _TreeEdits()
        for file in self.files:
            path = file.path(tags, self.name_map)
            if self.name_map or not file.patchable_name(path):
                return None
            if history is not None and not history.permits(file, path):
                return None
            if not edits.add(path, _RunLeaf(file.node, file.logical())):
                return None
        return edits

    def fill(self, run_node: dict[str, Any], run: RunCache) -> None:
        """Add every file to the directory of a run.

        Args:
            run_node: the run directory to fill.
            run: the run, which keeps its moved tags so a run that grows
                logs each move once.
        """
        self._separate_tensor_payloads()
        tags, paths, leaves, history = self._resolve_paths()
        tags.log(run)
        if self._insert_files(run_node, paths, leaves):
            history = _PathHistory(frozenset(), frozenset(), frozenset(), True)
        if history is not None:
            run_node["children"] = _RunChildren(run_node["children"], history)
        if self.name_map:
            run_node["children"][NAME_MAP] = {
                "type": "name_map",
                "data": self.name_map,
            }

    def _tag_names(self) -> dict[str, None]:
        return dict.fromkeys(file.tag for file in self.files if file.tag is not None)

    def _resolve_paths(
        self,
    ) -> tuple[TagPaths, list[_Path], set[_Path], _PathHistory | None]:
        blocked: dict[_Path, str] = {}
        scopes: set[_TagScope] = set()
        historical_dirs: set[_Path] = set()
        tag_names = self._tag_names()
        while True:
            tags = TagPaths(tag_names, blocked, self.name_map)
            paths = [file.path(tags, self.name_map) for file in self.files]
            leaves = set(paths)
            found = self.blocked_dirs(paths, leaves)
            if found.keys() <= blocked.keys():
                break
            for scope, path in self._scoped_paths(paths):
                scopes.add(scope)
                historical_dirs.update(
                    path[:end] for end in range(len(scope[0]) + 1, len(path))
                )
            blocked |= found
        history = None
        if blocked:
            history = _PathHistory(
                frozenset(historical_dirs),
                frozenset(scopes),
                frozenset(tags.renamed),
                False,
            )
        return tags, paths, leaves, history

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
    ) -> bool:
        directories: dict[_Path, dict[str, Any] | None] = {}
        for path in leaves:
            parent = path[:-1]
            while parent and parent not in directories:
                directories[parent] = None
                parent = parent[:-1]
        collisions: dict[tuple[str, ...], dict[str, _RunFile]] = {
            path: {} for path in leaves & directories.keys()
        }
        unstable = False
        for path, file in zip(paths, self.files, strict=True):
            if file.identity:
                collisions.setdefault(path, {})
            if path[0] == "raw" or file.nonstandard_tensor():
                unstable = True
        for path, file in zip(paths, self.files, strict=True):
            sources = collisions.get(path)
            if sources is None:
                parent = path[:-1]
                directory = directories.get(parent)
                if directory is None:
                    directory = directories[parent] = _Paths.mkdirs(run_node, parent)
                file.store(directory, path[-1])
            else:
                sources[file.identity] = file
        moved = self._insert_collisions(run_node, directories, leaves, collisions)
        return unstable or moved

    def _insert_collisions(
        self,
        run_node: dict[str, Any],
        directories: dict[_Path, dict[str, Any] | None],
        leaves: set[_Path],
        collisions: dict[_Path, dict[str, _RunFile]],
    ) -> bool:
        unstable = False
        for path, sources in collisions.items():
            directory = _Paths.mkdirs(run_node, path[:-1])
            number = 1
            for index, source in enumerate(sorted(sources)):
                resolved = path
                if index or path in directories:
                    unstable = True
                    resolved, number = self._free_leaf(
                        path, number, leaves, directories
                    )
                    leaves.add(resolved)
                sources[source].store(directory, resolved[-1])
        return unstable

    def _free_leaf(
        self,
        path: _Path,
        number: int,
        leaves: set[_Path],
        directories: dict[_Path, dict[str, Any] | None],
    ) -> tuple[_Path, int]:
        original = _Paths.decode_component(path[-1], name_map=self.name_map)
        while True:
            number += 1
            name = _Paths.fit(f"{path[-1]}~{number}", original, self.name_map)
            resolved = (*path[:-1], name)
            if resolved not in leaves and resolved not in directories:
                return resolved, number

    def blocked_dirs(
        self, paths: list[tuple[str, ...]], leaves: set[tuple[str, ...]]
    ) -> dict[tuple[str, ...], str]:
        """Return the tag directories that are also files, below their roots.

        Each value is the path of the file, for the warning.
        """
        found: dict[tuple[str, ...], str] = {}
        for (root, _, _), path in self._scoped_paths(paths):
            for end in range(len(root) + 1, len(path)):
                prefix = path[:end]
                if prefix in leaves:
                    found.setdefault(prefix[len(root) :], "/".join(prefix))
        return found

    def _scoped_paths(self, paths: list[_Path]) -> Iterator[tuple[_TagScope, _Path]]:
        """Yield one representative path per tagged scope in this resolution."""
        checked: set[_TagScope] = set()
        for path, file in zip(paths, self.files, strict=True):
            scope = file.scope()
            if scope is not None and scope not in checked:
                checked.add(scope)
                yield scope, path
