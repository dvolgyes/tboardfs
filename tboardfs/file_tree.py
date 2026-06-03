from pathlib import Path
from typing import Any, cast

from tboardfs.constants import DEFAULT_SCALAR_FORMATS
from tboardfs.filesystem import _build_run_virtual_tree, _materialize_node_bytes
from tboardfs.indexer import _EventIndexer
from tboardfs.model import RunCache
from tboardfs.paths import _Paths


class _CopyConflictError(FileExistsError):
    """File copy conflict with the successful virtual paths so far.

    :ivar copied_paths: virtual paths copied before the conflict
    :ivar target: filesystem path that already exists
    """

    copied_paths: list[str]
    target: Path

    def __init__(self, target: Path, copied_paths: list[str]) -> None:
        super().__init__(str(target))
        self.target = target
        self.copied_paths = copied_paths


class SingleEventTree:
    """Expose one event file through the shared virtual tree implementation.

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
        run = RunCache(())
        run.files[source_path] = _EventIndexer.parse_event_file(
            source_path, ignore_truncated=True
        )
        scalar_formats = (
            _Paths.normalize_formats(scalar_format) or DEFAULT_SCALAR_FORMATS
        )
        self.tree = _build_run_virtual_tree(
            run, scalar_formats, step_digits, include_sidecars=False
        )

    def list_file_paths(self, *, prefix: str = "/") -> list[str]:
        """Return virtual file paths below a prefix."""
        node = self._lookup_path(prefix)
        if node["type"] != "dir":
            return [_Paths.norm_virtual_path(prefix)]
        return [
            _Paths.join_path(path)
            for path in self._iter_file_parts(node, _Paths.path_parts(prefix))
        ]

    def read_file(self, path: str) -> bytes:
        """Return virtual file bytes."""
        node = self._lookup_path(path)
        if node["type"] == "dir":
            raise IsADirectoryError(path)
        return _materialize_node_bytes(node)

    def copy_all(
        self, outdir: str | Path, *, existing: str = "fail"
    ) -> tuple[int, list[str]]:
        """Copy every virtual file into a directory."""
        outdir_path = Path(outdir)
        paths = self._iter_file_parts(self.tree, ())
        targets = [(path, outdir_path.joinpath(*path)) for path in paths]
        copied_paths: list[str] = []
        skipped_paths: list[str] = []
        for path, target in targets:
            virtual_path = _Paths.join_path(path)
            if target.exists() and existing != "overwrite":
                if existing == "skip":
                    skipped_paths.append(virtual_path)
                    continue
                raise _CopyConflictError(target, copied_paths)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_materialize_node_bytes(self._lookup_parts(path)))
            copied_paths.append(virtual_path)
        return len(copied_paths), skipped_paths

    def _lookup_path(self, path: str) -> dict[str, Any]:
        return self._lookup_parts(_Paths.path_parts(path))

    def _lookup_parts(self, parts: tuple[str, ...]) -> dict[str, Any]:
        node = self.tree
        for part in parts:
            if node["type"] != "dir" or part not in node["children"]:
                raise FileNotFoundError(_Paths.join_path(parts))
            node = cast(dict[str, Any], node["children"][part])
        return node

    def _iter_file_parts(
        self, node: dict[str, Any], prefix_parts: tuple[str, ...]
    ) -> list[tuple[str, ...]]:
        if node["type"] != "dir":
            return [prefix_parts]
        paths = []
        for name in sorted(node["children"]):
            child = node["children"][name]
            paths.extend(self._iter_file_parts(child, (*prefix_parts, name)))
        return paths
