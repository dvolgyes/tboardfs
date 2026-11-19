from functools import partial

import numpy as np

from tboardfs.array_json import array_to_json
from tboardfs.constants import DT_STRING
from tboardfs.custom_scalars import custom_scalar_layout_json
from tboardfs.hparams import hparams_json
from tboardfs.mesh import mesh_nodes
from tboardfs.model import JsonEntry
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles
from tboardfs.skip_failure import skip_failure
from tboardfs.tensor import _TensorDType, tensor_to_array

# Row order of the pr_curves tensor, from the TensorBoard pr_curve plugin.
PR_CURVE_ROWS = (
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "precision",
    "recall",
)


def add_typed_plugin_files(
    files: RunFiles,
    entries: list[JsonEntry],
    scalars: dict[str, dict[str, np.ndarray]],
    step_digits: int,
) -> set[int]:
    """Add typed plugin files and return consumed entry identities."""
    consumed: set[int] = set()
    skip_failure(
        "the hparams file",
        partial(_TypedPluginFiles.add_hparams, files, entries, scalars, consumed),
    )
    skip_failure(
        "the custom scalars layout",
        partial(_TypedPluginFiles.add_custom_scalars, files, entries, consumed),
    )
    skip_failure(
        "the mesh files",
        partial(_TypedPluginFiles.add_meshes, files, entries, step_digits, consumed),
    )
    for entry in entries:
        marker = id(entry)
        if marker in consumed:
            continue
        builder = {
            "pr_curves": _TypedPluginFiles.add_tensor_table,
            "tensors": _TypedPluginFiles.add_tensor,
        }.get(entry.tab)
        if builder is not None:
            consumed.update(
                skip_failure(
                    f"{entry.tab} record {entry.tag!r} at step {entry.step}",
                    partial(builder, files, entry, step_digits),
                )
                or set()
            )
        elif entry.tab == "meshes" or entry.tab == "videos":
            consumed.add(marker)
    return consumed


class _TypedPluginFiles:
    """Add typed plugin records to the virtual tree."""

    @staticmethod
    def add_hparams(
        files: RunFiles,
        entries: list[JsonEntry],
        scalars: dict[str, dict[str, np.ndarray]],
        consumed: set[int],
    ) -> None:
        """Add a merged hparams summary file."""
        hparams = [entry for entry in entries if entry.tab == "hparams"]
        if not hparams:
            return
        files.add_fixed(
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
        files: RunFiles, entries: list[JsonEntry], consumed: set[int]
    ) -> None:
        """Add a merged custom scalar layout file."""
        layouts = [entry for entry in entries if entry.tab == "custom_scalars"]
        if not layouts:
            return
        files.add_fixed(
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
        files: RunFiles, entry: JsonEntry, step_digits: int
    ) -> set[int]:
        """Add table, JSON, and NPY views for tensor-backed plugin records."""
        data = tensor_to_array(entry.payload.get("tensor") or {})
        if data is None or data.size == 0:
            return set()
        step = _Paths.step_name(entry.step, step_digits)
        table = _TypedPluginFiles.pr_curve_table(data)
        for fmt in ("json", "tsv"):
            files.add(
                ("pr_curves",),
                entry.tag,
                f"{step}.{fmt}",
                {
                    "type": "table",
                    "series": table,
                    "format": fmt,
                    "mtime": entry.wall_time,
                },
            )
        files.add(
            ("pr_curves",),
            entry.tag,
            f"{step}.npy",
            {"type": "npy", "data": data, "mtime": entry.wall_time},
        )
        return {id(entry)}

    @staticmethod
    def add_tensor(files: RunFiles, entry: JsonEntry, step_digits: int) -> set[int]:
        """Add JSON and NPY views for one tensor record."""
        tensor = entry.payload.get("tensor") or {}
        if tensor.get("dtype") == DT_STRING:
            preview = _TensorDType.string_preview(tensor)
            if preview is None:
                return set()
            cells, dtype = preview
            json_data = array_to_json(cells, dtype=dtype)
            data = cells.astype(dtype)
        else:
            data = tensor_to_array(tensor, copy_content=False)
            if data is None:
                return set()
            json_data = array_to_json(data)
        step = _Paths.step_name(entry.step, step_digits)
        files.add(
            ("tensors",),
            entry.tag,
            f"{step}.json",
            {"type": "json", "data": json_data, "mtime": entry.wall_time},
        )
        files.add(
            ("tensors",),
            entry.tag,
            f"{step}.npy",
            {"type": "npy", "data": data, "mtime": entry.wall_time},
        )
        return {id(entry)}

    @staticmethod
    def add_meshes(
        files: RunFiles,
        entries: list[JsonEntry],
        step_digits: int,
        consumed: set[int],
    ) -> None:
        """Add grouped mesh component archives."""
        nodes, used = mesh_nodes(entries, step_digits)
        for tag, name, node in nodes:
            files.add(("meshes",), tag, name, node)
        consumed.update(used)

    @staticmethod
    def pr_curve_table(data: np.ndarray) -> dict[str, np.ndarray]:
        """Return one row per threshold, or generic columns for another shape.

        A [6, N] tensor has the rows of PR_CURVE_ROWS. TensorBoard spaces the
        N thresholds evenly in [0, 1].
        """
        if data.ndim != 2 or data.shape[0] != len(PR_CURVE_ROWS):
            return _TypedPluginFiles.array_table(data)
        table = {"threshold": np.linspace(0.0, 1.0, data.shape[1])}
        table.update(zip(PR_CURVE_ROWS, data, strict=True))
        return table

    @staticmethod
    def array_table(data: np.ndarray) -> dict[str, np.ndarray]:
        """Return a column-oriented table from a tensor array."""
        values = (
            data.reshape((-1, data.shape[-1]))
            if data.ndim > 1
            else data.reshape((-1, 1))
        )
        return {f"value_{index}": values[:, index] for index in range(values.shape[1])}
