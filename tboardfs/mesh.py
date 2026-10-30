from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

import numpy as np

from tboardfs.model import JsonEntry
from tboardfs.paths import _Paths
from tboardfs.proto_schema import _ProtoParser
from tboardfs.tensor import tensor_to_array

_MESH_COLOR = 3
_MESH_FACE = 2
_MESH_UNDEFINED = 0
_MESH_VERTEX = 1
_MESH_COMPONENTS = (
    (_MESH_VERTEX, "vertices"),
    (_MESH_FACE, "faces"),
    (_MESH_COLOR, "colors"),
)


@dataclass
class _MeshGroup:
    """TensorBoard mesh tensors grouped by mesh name and step."""

    components: dict[int, np.ndarray]
    entries: list[JsonEntry]
    metadata: Any
    mtime: float


def mesh_nodes(
    entries: list[JsonEntry],
    step_digits: int,
) -> tuple[list[tuple[tuple[str, ...], dict[str, Any]]], set[int]]:
    """Return virtual files for grouped TensorBoard mesh components."""
    nodes: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    consumed: set[int] = set()
    for (name, step_number), group in sorted(_MeshFiles.groups(entries).items()):
        if _MESH_VERTEX not in group.components:
            continue
        step = _Paths.step_name(step_number, step_digits)
        rel = ("meshes", *_Paths.tag_dir_parts(name))
        nodes.extend(_MeshFiles.files(rel, step, group))
        consumed.update(id(entry) for entry in group.entries)
    return nodes, consumed


class _MeshFiles:
    """Build the OBJ, NPZ, and metadata files of TensorBoard meshes.

    The OBJ file keeps every batch element as one object, vertex colors as
    r g b values in [0, 1] after the coordinates, and point clouds as
    vertex lines without faces.
    """

    @staticmethod
    def groups(entries: list[JsonEntry]) -> dict[tuple[str, int], _MeshGroup]:
        """Group mesh component tensors by TensorBoard mesh name and step."""
        groups: dict[tuple[str, int], _MeshGroup] = {}
        for entry in entries:
            if entry.tab != "meshes":
                continue
            content = entry.payload.get("plugin_content")
            data = tensor_to_array(entry.payload.get("tensor") or {})
            if not isinstance(content, bytes) or data.size == 0:
                continue
            metadata = cast(Any, _ProtoParser.mesh_plugin_data(content))
            if metadata.content_type == _MESH_UNDEFINED:
                continue
            group_key = (metadata.name or entry.tag.rsplit("_", 1)[0], int(entry.step))
            group = groups.get(group_key)
            if group is None:
                group = _MeshGroup({}, [], metadata, entry.wall_time)
                groups[group_key] = group
            group.components[metadata.content_type] = data
            group.entries.append(entry)
            group.mtime = max(group.mtime, entry.wall_time)
        return groups

    @staticmethod
    def files(
        rel: tuple[str, ...], step: str, group: _MeshGroup
    ) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
        """Return OBJ, NPZ, and metadata files for one mesh group."""
        mesh = {
            label: group.components[key]
            for key, label in _MESH_COMPONENTS
            if key in group.components
        }
        nodes = []
        obj = _MeshFiles.obj_text(mesh)
        if obj is not None:
            nodes.append(
                (
                    (*rel, f"{step}.obj"),
                    {"type": "obj", "data": obj, "mtime": group.mtime},
                )
            )
        handle = BytesIO()
        np.savez(handle, **mesh)
        nodes.append(
            (
                (*rel, f"{step}.npz"),
                {"type": "bytes", "data": handle.getvalue(), "mtime": group.mtime},
            )
        )
        nodes.append(
            (
                (*rel, f"{step}.json"),
                {
                    "type": "json",
                    "data": _MeshFiles.metadata(group),
                    "mtime": group.mtime,
                },
            )
        )
        return nodes

    @staticmethod
    def metadata(group: _MeshGroup) -> dict[str, Any]:
        """Return JSON-safe metadata for one mesh group."""
        metadata = group.metadata
        names = dict(_MESH_COMPONENTS)
        return {
            "name": metadata.name,
            "content_types": [
                names[key] for key in sorted(group.components) if key in names
            ],
            "shape": list(metadata.shape),
            "json_config": metadata.json_config,
        }

    @staticmethod
    def obj_text(mesh: dict[str, np.ndarray]) -> str | None:
        """Return Wavefront OBJ text, or None when the vertices are not [B,] N x 3.

        OBJ face indices count every vertex of the file, so the faces of a
        later batch element are offset by the vertices before it.
        """
        vertices = _MeshFiles.batched(mesh["vertices"])
        if vertices is None:
            return None
        faces = _MeshFiles.batched(mesh.get("faces"))
        colors = _MeshFiles.batched(mesh.get("colors"))
        if colors is not None and colors.shape != vertices.shape:
            colors = None
        lines = ["# TensorBoard mesh export"]
        offset = 0
        for index, points in enumerate(vertices):
            if len(vertices) > 1:
                lines.append(f"o mesh_{index}")
            for point_index, (x, y, z) in enumerate(points):
                color = (
                    ""
                    if colors is None
                    else _MeshFiles.color(colors[index, point_index])
                )
                lines.append(f"v {x} {y} {z}{color}")
            if faces is not None and index < len(faces):
                lines.extend(
                    f"f {offset + int(a) + 1} {offset + int(b) + 1} {offset + int(c) + 1}"
                    for a, b, c in faces[index]
                )
            offset += len(points)
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def batched(array: np.ndarray | None) -> np.ndarray | None:
        """Return a [B, N, 3] view of a component, or None for another shape."""
        if array is None or array.ndim not in (2, 3) or array.shape[-1] != 3:
            return None
        return array[None] if array.ndim == 2 else array

    @staticmethod
    def color(rgb: np.ndarray) -> str:
        """Return 0..255 color channels as the r g b suffix of an OBJ vertex."""
        return "".join(f" {int(channel) / 255:.6g}" for channel in rgb)
