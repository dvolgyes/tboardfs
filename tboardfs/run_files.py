"""Collect the virtual files of one run, then give each tag a directory that no file has."""

from collections import defaultdict
from collections.abc import Container, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tboardfs.epoch_breakdown import _EpochBreakdown
from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model import BinaryEntry, RunCache
from tboardfs.model.packed_binaries import PackedBinaries
from tboardfs.model.packed_leaf import PackedLeaf
from tboardfs.model.packed_source import PackedSource
from tboardfs.paths import NAME_MAP, _Paths
from tboardfs.run_listing import RunListing, _LogicalFile
from tboardfs.run_tree_edit import _IMPLICIT_PRODUCERS, _RunLeaf, _TreeEdits
from tboardfs.tag_paths import TagPaths

type _Path = tuple[str, ...]
type _TagScope = tuple[_Path, str, bool]
type _BinaryValue = BinaryEntry | PackedLeaf


@dataclass(frozen=True, slots=True)
class _PathHistory:
    """Record cold-only resolutions that a local patch cannot freely change."""

    directories: frozenset[_Path]
    scopes: frozenset[_TagScope]
    renamed: frozenset[str]
    unstable: bool

    def permits(self, scope: _TagScope | None, path: _Path) -> bool:
        """Reject edits that could change a prior collision resolution."""
        if path in self.directories:
            return False
        return scope is None or (scope[1] not in self.renamed and scope in self.scopes)


class _RunChildren(RunListing):
    """A run listing with exceptional path history kept out of virtual nodes."""

    __slots__: ClassVar[tuple[str, ...]] = ("history",)

    history: _PathHistory

    def __init__(
        self, children: RunListing | dict[str, Any], history: _PathHistory
    ) -> None:
        if isinstance(children, RunListing):
            super().__init__(children.entries, logical=children.logical)
        else:
            super().__init__(children)
        self.history = history

    def copy(self) -> "_RunChildren":
        """Copy the listing and retain its immutable path history."""
        return _RunChildren(super().copy(), self.history)


def _listing(directory: dict[str, Any]) -> RunListing:
    children = directory["children"]
    if not isinstance(children, RunListing):
        children = directory["children"] = RunListing(children)
    return children


@dataclass(slots=True)
class _RunFile:
    """One virtual file: a root, a tag directory and a file name.

    A file without a tag is at root + name. A file named after its tag
    (``named``) is the last part of the tag with ``name`` as its extension.
    """

    root: tuple[str, ...]
    tag: str | None
    name: str
    node: LeafValue
    named: bool
    identity: str = ""
    producer: str = ""
    step: int | float | None = None

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
            and LeafSpec.kind_of(self.node) == "binary"
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
        children = _listing(directory)
        if self.producer in _IMPLICIT_PRODUCERS:
            children.logical.pop(name, None)
        else:
            children.logical[name] = self.logical()
        children.set_leaf(name, self.node)


def _embedded(name: str) -> str:
    stem, ext = name.rsplit(".", 1)
    return f"{stem}.embedded.{ext}"


@dataclass(slots=True)
class _BinaryGroup:
    """One indexed binary batch's files in a single tag directory."""

    root: _Path
    tag: str
    names: dict[str, _BinaryValue] = field(default_factory=dict)
    long_names: set[str] = field(default_factory=set)
    steps: dict[int, tuple[str, int | float]] | None = None

    def add(self, name: str, entry: _BinaryValue, step: float) -> None:
        """Keep the latest entry and its position in producer order."""
        size = len(name)
        can_shorten = size > 255 or (size > 63 and not name.isascii())
        if name in self.names:
            del self.names[name]
        self.names[name] = entry
        if self.steps is not None:
            self.steps[id(entry)] = (self.tag, step)
        if can_shorten:
            self.long_names.add(name)

    def scope(self) -> _TagScope:
        """Return the scope used by cold collision and append checks."""
        return self.root, self.tag, False

    def parent(self, tags: TagPaths) -> _Path:
        """Return the shared resolved parent of every entry in the group."""
        return (*self.root, *tags.dir_parts(self.tag))

    def resolve_names(self, views: set[str], name_map: dict[str, str]) -> None:
        """Fit long names and move embedded payloads beside generated views."""
        embedded = {name: _embedded(name) for name in views if name in self.names}
        if not embedded and not self.long_names:
            return
        changed: dict[str, str] = {}
        for name in self.long_names | embedded.keys():
            alias = embedded.get(name, name)
            resolved = _Paths.fit(alias, alias, name_map)
            if resolved != name:
                changed[name] = resolved
        if changed:
            self.names = {
                changed.get(name, name): entry for name, entry in self.names.items()
            }


type _Stage = _RunFile | _BinaryGroup
type _CollisionSource = _RunFile | _BinaryValue
type _SourceNames = dict[str, dict[str, _CollisionSource]]


class _Collisions(dict[_Path, _SourceNames]):
    """Reserved leaf names and their competing sources, grouped by parent."""

    def reserve(self, path: _Path) -> None:
        """Reserve a contested name before collecting its producer candidates."""
        self.setdefault(path[:-1], {}).setdefault(path[-1], {})


class _Directories(dict[_Path, dict[str, Any] | None]):
    """Known directory paths and their nodes once created during publication."""

    def remember(self, parent: _Path) -> None:
        """Register an intended directory and any unknown ancestors."""
        while parent and parent not in self:
            self[parent] = None
            parent = parent[:-1]

    def open(self, run_node: dict[str, Any], parent: _Path) -> dict[str, Any]:
        """Return the directory, creating it in the run on first use."""
        directory = self.get(parent)
        if directory is None:
            directory = self[parent] = _Paths.mkdirs(run_node, parent)
        return directory


@dataclass(slots=True)
class _LeafPaths:
    """Membership over exceptional paths and grouped ordinary binary names."""

    files: set[_Path] = field(default_factory=set)
    groups: defaultdict[_Path, list[dict[str, _BinaryValue]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def track(self, file: _Stage, tags: TagPaths, name_map: dict[str, str]) -> _Path:
        """Index a stage and return its resolved leaf or group parent path."""
        if isinstance(file, _BinaryGroup):
            parent = file.parent(tags)
            self.groups[parent].append(file.names)
            return parent
        path = file.path(tags, name_map)
        self.files.add(path)
        return path

    def contains(self, path: _Path) -> bool:
        """Check for a leaf without expanding grouped names into full paths."""
        if path in self.files:
            return True
        groups = self.groups.get(path[:-1])
        return groups is not None and any(path[-1] in names for names in groups)


class RunFiles:
    """Collect the virtual files of one run and build its directory tree.

    The tree is built only when all files are known. Then a tag directory
    that has the name of a file, such as the tag "a/000001.npy" beside the
    step file 000001.npy of the tag "a", gets "~2" in every tab, and no file
    hides another. A later observation of the same logical file replaces it.

    Attributes:
        files: ordered producer stages; indexed binaries share one stage per tag.
    """

    files: list[_Stage]
    name_map: dict[str, str]
    epoch_breakdown: _EpochBreakdown | None

    def __init__(self, *, epoch_breakdown: _EpochBreakdown | None = None) -> None:
        self.files = []
        self.name_map = {}
        self.epoch_breakdown = epoch_breakdown

    def add(
        self,
        root: tuple[str, ...],
        tag: str,
        name: str,
        node: LeafValue,
        *,
        producer: str = "",
        step: float | None = None,
    ) -> None:
        """Add a file to the directory of a tag below a root, such as a tab."""
        self.files.append(
            _RunFile(
                root,
                tag,
                name,
                node,
                named=False,
                producer=producer,
                step=step if self.epoch_breakdown is not None else None,
            )
        )

    def add_named(
        self,
        root: tuple[str, ...],
        tag: str,
        extension: str,
        node: LeafValue,
        *,
        producer: str = "",
    ) -> None:
        """Add a file named after its tag, such as /scalars/<tag>.json."""
        self.files.append(
            _RunFile(root, tag, extension, node, named=True, producer=producer)
        )

    def add_fixed(
        self, parts: tuple[str, ...], node: LeafValue, *, identity: str = ""
    ) -> None:
        """Add a generated file or a sidecar with a distinct source identity."""
        self.files.append(_RunFile(parts[:-1], None, parts[-1], node, False, identity))

    def add_binaries(
        self, batches: Iterable[list[BinaryEntry] | PackedBinaries], step_digits: int
    ) -> None:
        """Resolve names in file order without expanding packed entries."""
        groups: dict[tuple[str, str], _BinaryGroup] = {}
        for batch in batches:
            for chunk in (
                batch.chunks() if isinstance(batch, PackedBinaries) else (batch,)
            ):
                if isinstance(chunk, list):
                    self._add_entries(groups, chunk, step_digits)
                else:
                    self._add_packed(groups, chunk, step_digits)

    def _binary_group(
        self, groups: dict[tuple[str, str], _BinaryGroup], kind: str, tag: str
    ) -> _BinaryGroup:
        key = kind, tag
        group = groups.get(key)
        if group is None:
            groups[key] = group = _BinaryGroup(
                (kind,), tag, steps={} if self.epoch_breakdown is not None else None
            )
            self.files.append(group)
        return group

    def _add_entries(
        self,
        groups: dict[tuple[str, str], _BinaryGroup],
        entries: list[BinaryEntry],
        step_digits: int,
    ) -> None:
        group = None
        for entry in entries:
            if group is None or entry.tag != group.tag or entry.kind != group.root[0]:
                group = self._binary_group(groups, entry.kind, entry.tag)
            name = (
                f"{_Paths.step_name(entry.step, step_digits)}{entry.suffix}"
                f".{entry.extension}"
            )
            group.add(name, entry, entry.step)

    def _add_packed(
        self,
        groups: dict[tuple[str, str], _BinaryGroup],
        source: PackedSource,
        step_digits: int,
    ) -> None:
        previous = None
        group = None
        ending = ""
        for label, step, wall_time, index in source.project():
            if group is None or label is not previous:
                tag, kind, extension, suffix = label
                group = self._binary_group(groups, kind, tag)
                ending = f"{suffix}.{extension}"
                previous = label
            name = _Paths.step_name(step, step_digits) + ending
            group.add(name, PackedLeaf(source, index, wall_time), step)

    def patch(
        self, tree: dict[str, Any], run: RunCache, *, raw: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Publish changed logical files, or decline if cold resolution can differ."""
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
            if not self._patch_stage(edits, file, tags, history):
                return None
        return edits

    def _patch_stage(
        self,
        edits: _TreeEdits,
        file: _Stage,
        tags: TagPaths,
        history: _PathHistory | None,
    ) -> bool:
        if isinstance(file, _BinaryGroup):
            return self._patch_binaries(edits, file, tags, history)
        path = file.path(tags, self.name_map)
        if self.name_map or not file.patchable_name(path):
            return False
        if history is not None and not history.permits(file.scope(), path):
            return False
        return edits.add(path, _RunLeaf(file.node, file.logical()))

    def _patch_binaries(
        self,
        edits: _TreeEdits,
        group: _BinaryGroup,
        tags: TagPaths,
        history: _PathHistory | None,
    ) -> bool:
        parent = group.parent(tags)
        if self.name_map or parent[0] in ("raw", NAME_MAP):
            return False
        scope = group.scope()
        for name, entry in group.names.items():
            if group.root == ("tensors",) and ".embedded." in name:
                return False
            path = (*parent, _Paths.fit(name, name, self.name_map))
            if self.name_map or (
                history is not None and not history.permits(scope, path)
            ):
                return False
            logical = (group.root, group.tag, name, False, "", "binary")
            if not edits.add(path, _RunLeaf(entry, logical)):
                return False
        return True

    def fill(self, run_node: dict[str, Any], run: RunCache) -> None:
        """Add every file to the directory of a run.

        Args:
            run_node: the run directory to fill.
            run: the run, which keeps its moved tags so a run that grows
                logs each move once.
        """
        self._prepare_names()
        tags, paths, leaves, history = self._resolve_paths()
        tags.log(run)
        if self._insert_files(run_node, paths, leaves):
            history = _PathHistory(frozenset(), frozenset(), frozenset(), True)
        if history is not None:
            run_node["children"] = _RunChildren(run_node["children"], history)
        if self.name_map:
            _listing(run_node).set_leaf(
                NAME_MAP, {"type": "name_map", "data": self.name_map}
            )
        if self.epoch_breakdown is not None:
            self._track_steps(self.epoch_breakdown)

    def _track_steps(self, epochs: _EpochBreakdown) -> None:
        for file in self.files:
            if isinstance(file, _RunFile) and file.tag is not None:
                epochs.track(file.node, file.tag, file.step)
            elif isinstance(file, _BinaryGroup) and file.steps is not None:
                for value in file.names.values():
                    tag, step = file.steps[id(value)]
                    epochs.track(value, tag, step)

    def _tag_names(self) -> dict[str, None]:
        return dict.fromkeys(file.tag for file in self.files if file.tag is not None)

    def _resolve_paths(
        self,
    ) -> tuple[TagPaths, list[_Path], _LeafPaths, _PathHistory | None]:
        blocked: dict[_Path, str] = {}
        scopes: set[_TagScope] = set()
        historical_dirs: set[_Path] = set()
        tag_names = self._tag_names()
        while True:
            tags = TagPaths(tag_names, blocked, self.name_map)
            leaves = _LeafPaths()
            paths = [leaves.track(file, tags, self.name_map) for file in self.files]
            found = self._blocked_dirs(paths, leaves)
            if found.keys() <= blocked.keys():
                break
            self._record_history(paths, scopes, historical_dirs)
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

    def _record_history(
        self, paths: list[_Path], scopes: set[_TagScope], directories: set[_Path]
    ) -> None:
        for scope, parent in self._scoped_parents(paths):
            scopes.add(scope)
            directories.update(self._tag_prefixes(scope[0], parent))

    def _prepare_names(self) -> None:
        """Separate embedded tensor views and fit ordinary binary names once."""
        views = self._matching_tensor_views()
        empty: set[str] = set()
        for file in self.files:
            if isinstance(file, _BinaryGroup):
                file.resolve_names(
                    views.get(file.tag, empty) if file.root == ("tensors",) else empty,
                    self.name_map,
                )
            elif (
                views
                and file.root == ("tensors",)
                and file.name in views.get(file.tag, empty)
                and LeafSpec.kind_of(file.node) == "binary"
            ):
                file.name = _embedded(file.name)

    def _matching_tensor_views(self) -> dict[str | None, set[str]]:
        payloads = self._tensor_payload_names()
        views: defaultdict[str | None, set[str]] = defaultdict(set)
        if not payloads:
            return views
        for file in self.files:
            if not isinstance(file, _RunFile) or file.root != ("tensors",):
                continue
            names = payloads.get(file.tag)
            if (
                names is not None
                and any(file.name in batch for batch in names)
                and LeafSpec.kind_of(file.node) != "binary"
            ):
                views[file.tag].add(file.name)
        return views

    def _tensor_payload_names(self) -> dict[str | None, list[Container[str]]]:
        payloads: defaultdict[str | None, list[Container[str]]] = defaultdict(list)
        for file in self.files:
            if file.root != ("tensors",):
                continue
            if isinstance(file, _BinaryGroup):
                payloads[file.tag].append(file.names)
            elif LeafSpec.kind_of(file.node) == "binary":
                payloads[file.tag].append((file.name,))
        return payloads

    def _insert_files(
        self,
        run_node: dict[str, Any],
        paths: list[_Path],
        leaves: _LeafPaths,
    ) -> bool:
        directories = _Directories()
        for path, file in zip(paths, self.files, strict=True):
            directories.remember(self._stage_parent(file, path))
        collisions = self._find_collisions(paths, leaves, directories)
        for path, file in zip(paths, self.files, strict=True):
            self._store_stage(run_node, file, path, directories, collisions)
        moved = self._insert_collisions(run_node, directories, leaves, collisions)
        return moved or any(
            path[0] == "raw"
            or (isinstance(file, _RunFile) and file.nonstandard_tensor())
            for path, file in zip(paths, self.files, strict=True)
        )

    def _find_collisions(
        self, paths: list[_Path], leaves: _LeafPaths, directories: _Directories
    ) -> _Collisions:
        collisions = _Collisions()
        for path in directories:
            if leaves.contains(path):
                collisions.reserve(path)
        for path, file in zip(paths, self.files, strict=True):
            if isinstance(file, _RunFile) and file.identity:
                collisions.reserve(path)
        return collisions

    def _store_stage(
        self,
        run_node: dict[str, Any],
        file: _Stage,
        path: _Path,
        directories: _Directories,
        collisions: _Collisions,
    ) -> None:
        parent = self._stage_parent(file, path)
        names = collisions.get(parent)
        if isinstance(file, _BinaryGroup):
            directory = directories.open(run_node, parent)
            self._store_group(_listing(directory), file, names)
            return
        sources = None if names is None else names.get(path[-1])
        if sources is not None:
            sources[file.identity] = file
            return
        file.store(directories.open(run_node, parent), path[-1])

    @staticmethod
    def _store_group(
        listing: RunListing,
        group: _BinaryGroup,
        collisions: _SourceNames | None,
    ) -> None:
        logical = listing.logical
        if collisions is None:
            for name in group.names.keys() & logical.keys() if logical else ():
                del logical[name]
            listing.entries.update(group.names)
            return
        for name, entry in group.names.items():
            sources = collisions.get(name)
            if sources is not None:
                sources[""] = entry
                continue
            logical.pop(name, None)
            listing.set_leaf(name, entry)

    def _insert_collisions(
        self,
        run_node: dict[str, Any],
        directories: _Directories,
        leaves: _LeafPaths,
        collisions: _Collisions,
    ) -> bool:
        moved = False
        for parent, names in collisions.items():
            directory = _Paths.mkdirs(run_node, parent)
            for name, sources in names.items():
                moved |= self._insert_sources(
                    directory, (*parent, name), sources, leaves, directories
                )
        return moved

    def _insert_sources(
        self,
        directory: dict[str, Any],
        path: _Path,
        sources: dict[str, _CollisionSource],
        leaves: _LeafPaths,
        directories: _Directories,
    ) -> bool:
        number = 1
        for index, identity in enumerate(sorted(sources)):
            resolved = path
            if index or path in directories:
                resolved, number = self._free_leaf(path, number, leaves, directories)
                leaves.files.add(resolved)
            self._store_source(directory, resolved[-1], sources[identity])
        return number != 1

    @staticmethod
    def _store_source(
        directory: dict[str, Any], name: str, source: _CollisionSource
    ) -> None:
        if isinstance(source, _RunFile):
            source.store(directory, name)
            return
        listing = _listing(directory)
        listing.logical.pop(name, None)
        listing.set_leaf(name, source)

    def _free_leaf(
        self,
        path: _Path,
        number: int,
        leaves: _LeafPaths,
        directories: _Directories,
    ) -> tuple[_Path, int]:
        original = _Paths.decode_component(path[-1], name_map=self.name_map)
        while True:
            number += 1
            name = _Paths.fit(f"{path[-1]}~{number}", original, self.name_map)
            resolved = (*path[:-1], name)
            if not leaves.contains(resolved) and resolved not in directories:
                return resolved, number

    def _blocked_dirs(self, paths: list[_Path], leaves: _LeafPaths) -> dict[_Path, str]:
        """Return the tag directories that are also files, below their roots.

        Each value is the path of the file, for the warning.
        """
        found: dict[tuple[str, ...], str] = {}
        for root, prefix in self._scoped_prefixes(paths):
            if leaves.contains(prefix):
                found.setdefault(prefix[len(root) :], "/".join(prefix))
        return found

    def _scoped_prefixes(self, paths: list[_Path]) -> Iterator[tuple[_Path, _Path]]:
        for (root, _, _), parent in self._scoped_parents(paths):
            yield from ((root, prefix) for prefix in self._tag_prefixes(root, parent))

    def _scoped_parents(self, paths: list[_Path]) -> Iterator[tuple[_TagScope, _Path]]:
        """Yield one parent per tagged scope in this resolution."""
        checked: set[_TagScope] = set()
        for path, file in zip(paths, self.files, strict=True):
            scope = file.scope()
            if scope is not None and scope not in checked:
                checked.add(scope)
                yield scope, self._stage_parent(file, path)

    @staticmethod
    def _tag_prefixes(root: _Path, parent: _Path) -> Iterator[_Path]:
        return (parent[:end] for end in range(len(root) + 1, len(parent) + 1))

    @staticmethod
    def _stage_parent(file: _Stage, path: _Path) -> _Path:
        return path if isinstance(file, _BinaryGroup) else path[:-1]
