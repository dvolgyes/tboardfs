from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Any, cast

import numpy as np

from tboardfs.leaf_spec import LeafSpec
from tboardfs.mesh_ply import ply_bytes
from tboardfs.model import JsonEntry
from tboardfs.paths import _Paths
from tboardfs.proto_schema import _ProtoParser
from tboardfs.skip_failure import skip_failure
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
) -> tuple[list[tuple[str, str, LeafSpec, int]], set[int]]:
    """Return each mesh file with its raw tag and recorded step."""
    nodes: list[tuple[str, str, LeafSpec, int]] = []
    consumed: set[int] = set()
    for (tag, step_number), group in sorted(_MeshFiles.groups(entries).items()):
        if _MESH_VERTEX not in group.components:
            continue
        step = _Paths.step_name(step_number, step_digits)
        group_files = skip_failure(
            f"mesh {tag!r} at step {step_number}",
            partial(_MeshFiles.files, step, group),
        )
        if group_files is None:
            continue
        nodes.extend((tag, *file, step_number) for file in group_files)
        consumed.update(id(entry) for entry in group.entries)
    return nodes, consumed


class _MeshFiles:
    """Build the OBJ, PLY, NPZ, and metadata files of TensorBoard meshes.

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
    def files(step: str, group: _MeshGroup) -> list[tuple[str, LeafSpec]]:
        """Return the names and nodes of the OBJ, PLY, NPZ, and metadata files."""
        mesh = {
            label: group.components[key]
            for key, label in _MESH_COMPONENTS
            if key in group.components
        }
        nodes: list[tuple[str, LeafSpec]] = []
        obj = _MeshFiles.obj_text(mesh)
        if obj is not None:
            nodes.append((f"{step}.obj", LeafSpec("obj", group.mtime, "data", obj)))
        ply = _MeshFiles.ply(mesh)
        if ply is not None:
            nodes.append((f"{step}.ply", LeafSpec("bytes", group.mtime, "data", ply)))
        handle = BytesIO()
        np.savez(handle, **mesh)
        nodes.append(
            (f"{step}.npz", LeafSpec("bytes", group.mtime, "data", handle.getvalue()))
        )
        nodes.append(
            (
                f"{step}.json",
                LeafSpec("json", group.mtime, "data", _MeshFiles.metadata(group)),
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
        faces = _MeshFiles.indices(
            _MeshFiles.batched(mesh.get("faces")), vertices.shape[1]
        )
        colors = _MeshFiles.rgb(mesh, vertices)
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
    def ply(mesh: dict[str, np.ndarray]) -> bytes | None:
        """Return a binary PLY file, or None when the vertices are not [B,] N x 3.

        Faces that are not vertex indices and colors that are not 0..255
        bytes of each vertex are left out; the NPZ file keeps them.
        """
        vertices = _MeshFiles.batched(mesh["vertices"])
        if vertices is None or vertices.dtype.kind not in "biuf":
            return None
        faces = _MeshFiles.indices(
            _MeshFiles.batched(mesh.get("faces")), vertices.shape[1]
        )
        return ply_bytes(vertices, faces, _MeshFiles.rgb(mesh, vertices))

    @staticmethod
    def rgb(mesh: dict[str, np.ndarray], vertices: np.ndarray) -> np.ndarray | None:
        """Return the [B, N, 3] colors, or None when they are not RGB bytes.

        Colors must have one whole number from 0 to 255 per channel of each
        vertex; the NPZ file keeps colors that do not.
        """
        colors = _MeshFiles.batched(mesh.get("colors"))
        if colors is None or colors.shape != vertices.shape:
            return None
        return colors if _MeshFiles.rgb_bytes(colors) else None

    @staticmethod
    def rgb_bytes(colors: np.ndarray) -> bool:
        """Return whether colors are whole numbers from 0 to 255."""
        return (
            colors.dtype.kind in "iuf"
            and _MeshFiles.finite(colors)
            and bool((colors == np.round(colors)).all())
            and (colors.size == 0 or (colors.min() >= 0 and colors.max() <= 255))
        )

    @staticmethod
    def batched(array: np.ndarray | None) -> np.ndarray | None:
        """Return a [B, N, 3] view of a component, or None for another shape."""
        if array is None or array.ndim not in (2, 3) or array.shape[-1] != 3:
            return None
        return array[None] if array.ndim == 2 else array

    @staticmethod
    def finite(array: np.ndarray) -> bool:
        """Return whether an array holds only finite real numbers."""
        return array.dtype.kind in "biuf" and bool(np.isfinite(array).all())

    @staticmethod
    def indices(faces: np.ndarray | None, count: int) -> np.ndarray | None:
        """Return the faces as vertex indices, or None when one is not valid.

        A valid index is a whole number from 0 to the vertex count less one.
        """
        if faces is None or not _MeshFiles.finite(faces):
            return None
        if faces.size and (
            (faces != np.round(faces)).any() or faces.min() < 0 or faces.max() >= count
        ):
            return None
        return faces.astype(np.int64)

    @staticmethod
    def color(rgb: np.ndarray) -> str:
        """Return 0..255 color channels as the r g b suffix of an OBJ vertex."""
        return "".join(f" {int(channel) / 255:.6g}" for channel in rgb)
