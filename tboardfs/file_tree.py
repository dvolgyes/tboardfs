"""Select, read, and copy exports from an event file or a directory tree."""

import os
from collections.abc import Iterator
from errno import ENOENT
from functools import partial
from pathlib import Path
from typing import Any, cast

from tboardfs.constants import DEFAULT_HISTOGRAM_FORMATS, DEFAULT_SCALAR_FORMATS
from tboardfs.export_formats import FormatList, _ExportFormats
from tboardfs.filesystem import TensorBoardFS
from tboardfs.host_publish import OutputPublisher, _OutputConflictError
from tboardfs.indexer import _EventIndexer
from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model import RunCache, TFRecordError
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.raw_frames import RawFrames, _RawFrameChildren
from tboardfs.render_node import render_node
from tboardfs.run_listing import RunListing
from tboardfs.run_tree import RunTree


def _render_leaf(value: LeafValue) -> bytes:
    return render_node(LeafSpec.node(value))


class _CopyError(OSError):
    """A failed publication with the paths already published or skipped."""

    copied_paths: list[str]
    skipped_paths: list[str]

    def __init__(
        self,
        target: Path,
        cause: OSError | TFRecordError,
        copied: list[str],
        skipped: list[str],
    ) -> None:
        message = (
            f"output already exists: {target}\nUse --force for forced overwriting."
            if isinstance(cause, _OutputConflictError)
            else f"could not copy to {target}: {cause}"
        )
        super().__init__(message)
        self.copied_paths = copied
        self.skipped_paths = skipped


class _NoEventError(ValueError):
    """A source file in which not one TensorBoard record can be read."""

    def __init__(self, source: Path) -> None:
        super().__init__(f"no TensorBoard event in {source}")


class SingleEventTree:
    """Expose one event file through the shared virtual tree implementation.

    A run or log directory is read as filesystem mode shows it, with its
    run directories and sidecars.

    :ivar tree: root virtual directory node
    """

    tree: dict[str, Any]

    def __init__(
        self,
        source: str | Path,
        *,
        step_digits: int = 6,
        scalar_format: FormatList = DEFAULT_SCALAR_FORMATS,
        histogram_format: FormatList = DEFAULT_HISTOGRAM_FORMATS,
        pr_curve_npy: bool = False,
        epoch_breakdown: bool = False,
    ) -> None:
        source_path = Path(source)
        if source_path.is_dir():
            filesystem = TensorBoardFS(
                source_path,
                step_digits=step_digits,
                scalar_formats=scalar_format,
                histogram_formats=histogram_format,
                pr_curve_npy=pr_curve_npy,
                epoch_breakdown=epoch_breakdown,
                cache_size_mb=0,
                raw=True,
            )
            self.tree = filesystem.virtual_tree()
            return
        formats = _ExportFormats.parse(scalar_format, histogram_format, pr_curve_npy)
        run = RunCache(())
        cache = _EventIndexer.parse_event_file(
            source_path, ignore_truncated=True, strict_framing=True
        )
        if cache.progress.processed_pos == 0:
            raise _NoEventError(source_path)
        run.files[source_path] = cache
        self.tree = RunTree.build(
            run, formats, step_digits, epoch_breakdown=epoch_breakdown
        )

    @classmethod
    def from_framing(cls, source: str | Path) -> "SingleEventTree":
        """Build a single source's raw tree without decoding event payloads."""
        source_path = Path(source)
        if source_path.is_dir():
            raise IsADirectoryError("--framing-only requires a single event file")
        raw = RawFrames.index_source(source_path)
        if not raw["children"]:
            raise _NoEventError(source_path)
        tree = cls.__new__(cls)
        tree.tree = _DirectoryNode()
        tree.tree["children"]["raw"] = raw
        return tree

    def iter_file_paths(
        self, *, prefix: str = "/", view: str | None = None
    ) -> Iterator[str]:
        """Yield selected files in virtual-path order."""
        node = self._lookup_path(prefix)
        if view is None:
            view = (
                "raw" if node.get("view") == "raw" or node["type"] == "raw" else "typed"
            )
        for parts, _ in self._iter_files(node, _Paths.path_parts(prefix), view):
            yield _Paths.join_path(parts)

    def list_file_paths(
        self, *, prefix: str = "/", view: str | None = None
    ) -> list[str]:
        """Return virtual file paths below a prefix."""
        return list(self.iter_file_paths(prefix=prefix, view=view))

    def read_file(self, path: str, *, view: str = "both") -> bytes:
        """Return virtual file bytes."""
        node = self._lookup_path(path)
        if node["type"] == "dir":
            raise IsADirectoryError(path)
        if not self._selected(node, view):
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), path)
        return render_node(node)

    def copy_all(
        self, outdir: str | Path, *, existing: str = "fail", view: str = "typed"
    ) -> tuple[int, list[str]]:
        """Copy selected files, reporting completed paths if publication fails."""
        outdir_path = Path(outdir)
        copied: list[str] = []
        skipped: list[str] = []
        target = outdir_path
        try:
            with OutputPublisher.directory(outdir_path) as root:
                for parts, node in self._iter_files(self.tree, (), view):
                    target = outdir_path.joinpath(*parts)
                    written = OutputPublisher.publish(
                        root, parts, partial(_render_leaf, node), existing
                    )
                    (copied if written else skipped).append(_Paths.join_path(parts))
        except (OSError, TFRecordError) as error:
            raise _CopyError(target, error, copied, skipped) from error
        return len(copied), skipped

    def _lookup_path(self, path: str) -> dict[str, Any]:
        return self._lookup_parts(_Paths.path_parts(path))

    def _lookup_parts(self, parts: tuple[str, ...]) -> dict[str, Any]:
        node = self.tree
        for part in parts:
            if node["type"] != "dir" or (child := node["children"].get(part)) is None:
                raise FileNotFoundError(
                    ENOENT, os.strerror(ENOENT), _Paths.join_path(parts)
                )
            node = cast(dict[str, Any], child)
        return node

    @staticmethod
    def _selected(node: LeafValue, view: str) -> bool:
        return view == "both" or view == (
            "raw" if LeafSpec.kind_of(node) == "raw" else "typed"
        )

    def _iter_files(
        self, node: LeafValue, parts: tuple[str, ...], view: str
    ) -> Iterator[tuple[tuple[str, ...], LeafValue]]:
        if not isinstance(node, dict) or node["type"] != "dir":
            if self._selected(node, view):
                yield parts, node
            return
        path = list(parts)
        stack = [self._children(node, view)]
        while stack:
            for name, child in stack[-1]:
                if isinstance(child, dict) and child["type"] == "dir":
                    path.append(name)
                    stack.append(self._children(child, view))
                    break
                if self._selected(child, view):
                    yield (*path, name), child
            else:
                stack.pop()
                if stack:
                    path.pop()

    @staticmethod
    def _children(node: dict[str, Any], view: str) -> Iterator[tuple[str, LeafValue]]:
        """Yield ordered child names and compact entries without expanding a subtree."""
        if view == "typed" and node.get("view") == "raw":
            return
        children = node["children"]
        if isinstance(children, _RawFrameChildren):
            yield from children.iter_nodes()
            return
        entries = children.entries if isinstance(children, RunListing) else children
        for name in sorted(
            entries,
            key=lambda n: n + ("/" if LeafSpec.kind_of(entries[n]) == "dir" else ""),
        ):
            yield name, entries[name]
