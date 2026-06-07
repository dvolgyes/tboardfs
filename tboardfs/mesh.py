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
    for (name, step_number), group in sorted(_mesh_groups(entries).items()):
        if _MESH_VERTEX not in group.components:
            continue
        step = _Paths.step_name(step_number, step_digits)
        rel = ("meshes", *_Paths.tag_dir_parts(name))
        nodes.extend(_mesh_files(rel, step, group))
        consumed.update(id(entry) for entry in group.entries)
    return nodes, consumed


def _mesh_groups(entries: list[JsonEntry]) -> dict[tuple[str, int], _MeshGroup]:
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


def _mesh_files(
    rel: tuple[str, ...], step: str, group: _MeshGroup
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Return OBJ, NPZ, and metadata files for one mesh group."""
    mesh = _mesh_arrays(group.components)
    nodes = _obj_nodes(rel, step, mesh, group.mtime)
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
            {"type": "json", "data": _mesh_metadata(group), "mtime": group.mtime},
        )
    )
    return nodes


def _mesh_arrays(components: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Return named component arrays for one mesh group."""
    return {
        label: components[key] for key, label in _MESH_COMPONENTS if key in components
    }


def _mesh_metadata(group: _MeshGroup) -> dict[str, Any]:
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


def _obj_nodes(
    rel: tuple[str, ...],
    step: str,
    mesh: dict[str, np.ndarray],
    mtime: float,
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Return an OBJ virtual file when a mesh has face data."""
    if "faces" not in mesh:
        return []
    vertices = mesh["vertices"][0] if mesh["vertices"].ndim == 3 else mesh["vertices"]
    faces = mesh["faces"][0] if mesh["faces"].ndim == 3 else mesh["faces"]
    obj_lines = ["# TensorBoard mesh export"]
    obj_lines.extend(f"v {x} {y} {z}" for x, y, z in vertices)
    obj_lines.extend(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}" for a, b, c in faces)
    obj_lines.append("")
    return [
        (
            (*rel, f"{step}.obj"),
            {"type": "obj", "data": "\n".join(obj_lines), "mtime": mtime},
        )
    ]
