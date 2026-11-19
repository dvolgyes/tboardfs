"""Select, read, and copy exports from an event file or a directory tree."""

from collections.abc import Iterator
from errno import ENOENT
import os
from functools import partial
from pathlib import Path
from typing import Any, cast

from tboardfs.constants import DEFAULT_SCALAR_FORMATS
from tboardfs.filesystem import TensorBoardFS
from tboardfs.host_publish import OutputPublisher, _OutputConflictError
from tboardfs.indexer import _EventIndexer
from tboardfs.model import RunCache, TFRecordError
from tboardfs.paths import _Paths
from tboardfs.raw_frames import _RawFrameChildren
from tboardfs.render_node import render_node
from tboardfs.run_tree import RunTree


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
        scalar_format: str = "json,tsv,npz",
    ) -> None:
        source_path = Path(source)
        scalar_formats = (
            _Paths.normalize_formats(scalar_format) or DEFAULT_SCALAR_FORMATS
        )
        if source_path.is_dir():
            filesystem = TensorBoardFS(
                source_path,
                step_digits=step_digits,
                scalar_formats=scalar_formats,
                cache_size_mb=0,
            )
            self.tree = filesystem.virtual_tree()
            return
        run = RunCache(())
        cache = _EventIndexer.parse_event_file(
            source_path, ignore_truncated=True, strict_framing=True
        )
        if cache.progress.processed_pos == 0:
            raise _NoEventError(source_path)
        run.files[source_path] = cache
        self.tree = RunTree.build(run, scalar_formats, step_digits)

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
                        root, parts, partial(render_node, node), existing
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
            if node["type"] != "dir" or part not in node["children"]:
                raise FileNotFoundError(
                    ENOENT, os.strerror(ENOENT), _Paths.join_path(parts)
                )
            node = cast(dict[str, Any], node["children"][part])
        return node

    @staticmethod
    def _selected(node: dict[str, Any], view: str) -> bool:
        return view == "both" or view == ("raw" if node["type"] == "raw" else "typed")

    def _iter_files(
        self, node: dict[str, Any], parts: tuple[str, ...], view: str
    ) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
        if node["type"] != "dir":
            if self._selected(node, view):
                yield parts, node
            return
        if view == "typed" and node.get("view") == "raw":
            return
        children = node["children"]
        if isinstance(children, _RawFrameChildren):
            for name, child in children.iter_nodes():
                yield (*parts, name), child
            return
        for name in sorted(
            children, key=lambda n: n + ("/" if children[n]["type"] == "dir" else "")
        ):
            yield from self._iter_files(children[name], (*parts, name), view)
