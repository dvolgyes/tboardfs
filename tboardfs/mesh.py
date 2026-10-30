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
# The tag suffixes of TensorBoard's mesh summary_v2 and of PyTorch.
_MESH_TAG_SUFFIXES = (
    (_MESH_VERTEX, "_VERTEX"),
    (_MESH_FACE, "_FACE"),
    (_MESH_COLOR, "_COLOR"),
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
) -> tuple[list[tuple[str, str, dict[str, Any]]], set[int]]:
    """Return the mesh tag, file name and node of each mesh file."""
    nodes: list[tuple[str, str, dict[str, Any]]] = []
    consumed: set[int] = set()
    for (tag, step_number), group in sorted(_MeshFiles.groups(entries).items()):
        if _MESH_VERTEX not in group.components:
            continue
        step = _Paths.step_name(step_number, step_digits)
        nodes.extend((tag, *file) for file in _MeshFiles.files(step, group))
        consumed.update(id(entry) for entry in group.entries)
    return nodes, consumed


class _MeshFiles:
    """Build the OBJ, NPZ, and metadata files of TensorBoard meshes.

    The OBJ file keeps every batch element as one object, vertex colors as
    r g b values in [0, 1] after the coordinates, and point clouds as
    vertex lines without faces.
    """

    @staticmethod
    def tag(entry: JsonEntry) -> str | None:
        """Return the mesh tag of a component, or None when it is not one.

        The mesh tag is the tag without the suffix of its content type, so it
        keeps the name scope that the mesh name does not have. A tag without
        that suffix falls back to the mesh name, then to the tag.
        """
        content = entry.payload.get("plugin_content")
        if entry.tab != "meshes" or not isinstance(content, bytes):
            return None
        metadata = cast(Any, _ProtoParser.mesh_plugin_data(content))
        if metadata.content_type == _MESH_UNDEFINED:
            return None
        suffix = dict(_MESH_TAG_SUFFIXES).get(metadata.content_type)
        if suffix is not None and entry.tag.endswith(suffix):
            return entry.tag.removesuffix(suffix)
        return cast(str, metadata.name) or entry.tag

    @staticmethod
    def groups(entries: list[JsonEntry]) -> dict[tuple[str, int], _MeshGroup]:
        """Group mesh component tensors by mesh tag and step."""
        groups: dict[tuple[str, int], _MeshGroup] = {}
        for entry in entries:
            tag = _MeshFiles.tag(entry)
            if tag is None:
                continue
            data = tensor_to_array(entry.payload.get("tensor") or {})
            if data is None or data.size == 0:
                continue
            content = cast(bytes, entry.payload["plugin_content"])
            metadata = cast(Any, _ProtoParser.mesh_plugin_data(content))
            group_key = (tag, int(entry.step))
            group = groups.get(group_key)
            if group is None:
                group = _MeshGroup({}, [], metadata, entry.wall_time)
                groups[group_key] = group
            group.components[metadata.content_type] = data
            group.entries.append(entry)
            group.mtime = max(group.mtime, entry.wall_time)
        return groups

    @staticmethod
    def files(step: str, group: _MeshGroup) -> list[tuple[str, dict[str, Any]]]:
        """Return the names and nodes of the OBJ, NPZ, and metadata files."""
        mesh = {
            label: group.components[key]
            for key, label in _MESH_COMPONENTS
            if key in group.components
        }
        nodes = []
        obj = _MeshFiles.obj_text(mesh)
        if obj is not None:
            nodes.append(
                (f"{step}.obj", {"type": "obj", "data": obj, "mtime": group.mtime})
            )
        handle = BytesIO()
        np.savez(handle, **mesh)
        nodes.append(
            (
                f"{step}.npz",
                {"type": "bytes", "data": handle.getvalue(), "mtime": group.mtime},
            )
        )
        nodes.append(
            (
                f"{step}.json",
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
