import time
from typing import Any

import numpy as np

from tboardfs.array_json import array_to_json
from tboardfs.custom_scalars import custom_scalar_layout_json
from tboardfs.hparams import hparams_json
from tboardfs.mesh import mesh_nodes
from tboardfs.model import JsonEntry
from tboardfs.paths import _Paths
from tboardfs.tensor import tensor_to_array


def add_typed_plugin_files(
    run_node: dict[str, Any],
    entries: list[JsonEntry],
    scalars: dict[str, dict[str, np.ndarray]],
    step_digits: int,
) -> set[int]:
    """Add typed plugin files and return consumed entry identities."""
    consumed: set[int] = set()
    _TypedPluginFiles.add_hparams(run_node, entries, scalars, consumed)
    _TypedPluginFiles.add_custom_scalars(run_node, entries, consumed)
    _TypedPluginFiles.add_meshes(run_node, entries, step_digits, consumed)
    for entry in entries:
        marker = id(entry)
        if marker in consumed:
            continue
        if entry.tab == "pr_curves":
            consumed.update(
                _TypedPluginFiles.add_tensor_table(run_node, entry, step_digits)
            )
        elif entry.tab == "tensors":
            consumed.update(_TypedPluginFiles.add_tensor(run_node, entry, step_digits))
        elif entry.tab == "meshes" or entry.tab == "videos":
            consumed.add(marker)
    return consumed


class _TypedPluginFiles:
    """Add typed plugin records to the virtual tree."""

    @staticmethod
    def add_hparams(
        run_node: dict[str, Any],
        entries: list[JsonEntry],
        scalars: dict[str, dict[str, np.ndarray]],
        consumed: set[int],
    ) -> None:
        """Add a merged hparams summary file."""
        hparams = [entry for entry in entries if entry.tab == "hparams"]
        if not hparams:
            return
        _TypedPluginFiles.add_file(
            run_node,
            ("hparams", "hparams.json"),
            {
                "type": "json",
                "data": hparams_json(hparams, scalars),
                "mtime": max(e.wall_time for e in hparams),
            },
        )
        consumed.update(id(entry) for entry in hparams)

    @staticmethod
    def add_custom_scalars(
        run_node: dict[str, Any], entries: list[JsonEntry], consumed: set[int]
    ) -> None:
        """Add a merged custom scalar layout file."""
        layouts = [entry for entry in entries if entry.tab == "custom_scalars"]
        if not layouts:
            return
        _TypedPluginFiles.add_file(
            run_node,
            ("custom_scalars", "layout.json"),
            {
                "type": "json",
                "data": custom_scalar_layout_json(layouts),
                "mtime": max(e.wall_time for e in layouts),
            },
        )
        consumed.update(id(entry) for entry in layouts)

    @staticmethod
    def add_tensor_table(
        run_node: dict[str, Any], entry: JsonEntry, step_digits: int
    ) -> set[int]:
        """Add table, JSON, and NPY views for tensor-backed plugin records."""
        data = tensor_to_array(entry.payload.get("tensor") or {})
        if data.size == 0:
            return set()
        step = _Paths.step_name(entry.step, step_digits)
        table = _TypedPluginFiles.array_table(data)
        rel = ("pr_curves", *_Paths.tag_dir_parts(entry.tag))
        for fmt in ("json", "tsv"):
            _TypedPluginFiles.add_file(
                run_node,
                (*rel, f"{step}.{fmt}"),
                {
                    "type": "table",
                    "series": table,
                    "format": fmt,
                    "mtime": entry.wall_time,
                },
            )
        _TypedPluginFiles.add_file(
            run_node,
            (*rel, f"{step}.npy"),
            {"type": "npy", "data": data, "mtime": entry.wall_time},
        )
        return {id(entry)}

    @staticmethod
    def add_tensor(
        run_node: dict[str, Any], entry: JsonEntry, step_digits: int
    ) -> set[int]:
        """Add JSON and NPY views for one tensor record."""
        data = tensor_to_array(entry.payload.get("tensor") or {})
        if data.size == 0:
            return set()
        step = _Paths.step_name(entry.step, step_digits)
        rel = ("tensors", *_Paths.tag_dir_parts(entry.tag))
        _TypedPluginFiles.add_file(
            run_node,
            (*rel, f"{step}.json"),
            {"type": "json", "data": array_to_json(data), "mtime": entry.wall_time},
        )
        _TypedPluginFiles.add_file(
            run_node,
            (*rel, f"{step}.npy"),
            {"type": "npy", "data": data, "mtime": entry.wall_time},
        )
        return {id(entry)}

    @staticmethod
    def add_meshes(
        run_node: dict[str, Any],
        entries: list[JsonEntry],
        step_digits: int,
        consumed: set[int],
    ) -> None:
        """Add grouped mesh component archives."""
        nodes, used = mesh_nodes(entries, step_digits)
        for rel, node in nodes:
            _TypedPluginFiles.add_file(run_node, rel, node)
        consumed.update(used)

    @staticmethod
    def add_file(
        root: dict[str, Any], rel_parts: tuple[str, ...], node: dict[str, Any]
    ) -> None:
        """Add one virtual file to a directory tree."""
        parent = _Paths.mkdirs(root, rel_parts[:-1])
        node.setdefault("mtime", time.time())
        parent["children"][rel_parts[-1]] = node

    @staticmethod
    def array_table(data: np.ndarray) -> dict[str, np.ndarray]:
        """Return a column-oriented table from a tensor array."""
        values = (
            data.reshape((-1, data.shape[-1]))
            if data.ndim > 1
            else data.reshape((-1, 1))
        )
        return {f"value_{index}": values[:, index] for index in range(values.shape[1])}
