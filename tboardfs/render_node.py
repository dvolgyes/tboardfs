"""Render the bytes of one virtual file node."""

import json
from collections.abc import Callable, Mapping
from errno import EIO, EOPNOTSUPP
from types import MappingProxyType
from typing import Any, Final

from tboardfs.array_json import array_to_json
from tboardfs.errors import fuse_error
from tboardfs.extract import _extract_graph_blob, extract_binary_blob
from tboardfs.meta_graph import meta_graph_graph_def
from tboardfs.npy_export import array_to_npy
from tboardfs.open3d_asset import OPEN3D_PLUGIN, Open3DAsset
from tboardfs.raw_frames import RawFrames
from tboardfs.run_metadata_text import run_metadata_text
from tboardfs.summary_graph import summary_graph_def
from tboardfs.tables import _TableExport, export_table
from tboardfs.tensor import _read_tensor_array


def render_node(node: dict[str, Any]) -> bytes:
    """Return the byte representation for one virtual file node."""
    render = _RENDERERS.get(node["type"])
    if render is None:
        raise fuse_error(EOPNOTSUPP)
    return render(node)


def _derived(data: bytes | None) -> bytes:
    """Return bytes derived from an event; None means the file changed after indexing."""
    if data is None:
        raise fuse_error(EIO)
    return data


def _sidecar(node: dict[str, Any]) -> bytes:
    entry = node["entry"]
    if entry.tab == OPEN3D_PLUGIN:
        return Open3DAsset.read(entry)
    return bytes(entry.source_path.read_bytes())


_RENDERERS: Final[Mapping[str, Callable[[dict[str, Any]], bytes]]] = MappingProxyType(
    {
        "bytes": lambda node: bytes(node["data"]),
        "obj": lambda node: str(node["data"]).encode(),
        "npy": lambda node: array_to_npy(
            node["data"] if "data" in node else _read_tensor_array(node["tensor_entry"])
        ),
        "json": lambda node: _TableExport.json_document(
            node["data"]
            if "data" in node
            else array_to_json(_read_tensor_array(node["tensor_entry"]))
        ),
        "name_map": lambda node: json.dumps(
            node["data"], ensure_ascii=True, sort_keys=True, indent=2
        ).encode("ascii"),
        "raw": RawFrames.read,
        "scalar": lambda node: export_table(node["series"], node["format"]),
        "table": lambda node: export_table(node["series"], node["format"]),
        "binary": lambda node: extract_binary_blob(node["entry"]),
        "graph": lambda node: _extract_graph_blob(node["entry"]),
        "graph_def": lambda node: _derived(
            summary_graph_def(node["entry"], extract_binary_blob(node["entry"]))
        ),
        "meta_graph_graph": lambda node: _derived(
            meta_graph_graph_def(node["entry"], _extract_graph_blob(node["entry"]))
        ),
        "run_metadata_text": lambda node: _derived(
            run_metadata_text(node["entry"], extract_binary_blob(node["entry"]))
        ),
        "sidecar": _sidecar,
    }
)
