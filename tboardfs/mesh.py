from io import BytesIO
from typing import Any

import numpy as np
from tensorboard.plugins.mesh import plugin_data_pb2 as mesh_plugin_data_pb2

from tboardfs.model import JsonEntry
from tboardfs.paths import _Paths
from tboardfs.tensor import tensor_to_array


def mesh_nodes(
    entries: list[JsonEntry],
    step_digits: int,
) -> tuple[list[tuple[tuple[str, ...], dict[str, Any]]], set[int]]:
    """Return virtual files for grouped TensorBoard mesh components."""
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    consumed: set[int] = set()
    for entry in entries:
        if entry.tab != "meshes":
            continue
        content = entry.payload.get("plugin_content")
        data = tensor_to_array(entry.payload.get("tensor") or {})
        if not isinstance(content, bytes) or data.size == 0:
            continue
        metadata = mesh_plugin_data_pb2.MeshPluginData()
        metadata.ParseFromString(content)
        if metadata.content_type == mesh_plugin_data_pb2.MeshPluginData.UNDEFINED:
            continue
        group_key = (metadata.name or entry.tag.rsplit("_", 1)[0], int(entry.step))
        group = groups.setdefault(
            group_key,
            {
                "components": {},
                "entries": [],
                "metadata": metadata,
                "mtime": entry.wall_time,
            },
        )
        group["components"][metadata.content_type] = data
        group["entries"].append(entry)
        group["mtime"] = max(group["mtime"], entry.wall_time)

    nodes: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    names = {
        mesh_plugin_data_pb2.MeshPluginData.VERTEX: "vertices",
        mesh_plugin_data_pb2.MeshPluginData.FACE: "faces",
        mesh_plugin_data_pb2.MeshPluginData.COLOR: "colors",
    }
    for (name, step_number), group in sorted(groups.items()):
        components = group["components"]
        if mesh_plugin_data_pb2.MeshPluginData.VERTEX not in components:
            continue
        mesh = {
            label: components[key] for key, label in names.items() if key in components
        }
        step = _Paths.step_name(step_number, step_digits)
        rel = ("meshes", *_Paths.tag_dir_parts(name))
        metadata = group["metadata"]
        json_data = {
            "name": metadata.name,
            "content_types": [names[key] for key in sorted(components) if key in names],
            "shape": list(metadata.shape),
            "json_config": metadata.json_config,
        }
        if "faces" in mesh:
            vertices = (
                mesh["vertices"][0] if mesh["vertices"].ndim == 3 else mesh["vertices"]
            )
            faces = mesh["faces"][0] if mesh["faces"].ndim == 3 else mesh["faces"]
            obj_lines = ["# TensorBoard mesh export"]
            obj_lines.extend(f"v {x} {y} {z}" for x, y, z in vertices)
            obj_lines.extend(
                f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}" for a, b, c in faces
            )
            obj_lines.append("")
            nodes.append(
                (
                    (*rel, f"{step}.obj"),
                    {
                        "type": "obj",
                        "data": "\n".join(obj_lines),
                        "mtime": group["mtime"],
                    },
                )
            )
        handle = BytesIO()
        np.savez(handle, **mesh)
        nodes.append(
            (
                (*rel, f"{step}.npz"),
                {"type": "bytes", "data": handle.getvalue(), "mtime": group["mtime"]},
            )
        )
        nodes.append(
            (
                (*rel, f"{step}.json"),
                {"type": "json", "data": json_data, "mtime": group["mtime"]},
            )
        )
        consumed.update(id(entry) for entry in group["entries"])
    return nodes, consumed
