"""Describe typed files without building their renderer nodes."""

from dataclasses import dataclass, field
from errno import EIO
from typing import Any, Literal, cast

from tboardfs.errors import fuse_error
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.json_entry import JsonEntry

type LeafValue = BinaryEntry | LeafSpec | dict[str, Any]


@dataclass(slots=True, eq=False)
class LeafSpec:
    """Store a file's renderer fields without a node header."""

    kind: str
    mtime: float
    key: Literal["data", "entry", "series", "tensor_entry"]
    payload: object
    format: str | None = field(default=None, kw_only=True)

    @staticmethod
    def kind_of(value: LeafValue) -> str:
        """Return the virtual node type without constructing a file node."""
        if isinstance(value, dict):
            return cast(str, value["type"])
        return value.kind if isinstance(value, LeafSpec) else "binary"

    @staticmethod
    def node(value: LeafValue) -> dict[str, Any]:
        """Return an eager node or construct one from a compact entry."""
        if isinstance(value, dict):
            return value
        if isinstance(value, BinaryEntry):
            return {"type": "binary", "entry": value, "mtime": value.wall_time}
        node: dict[str, Any] = {
            value.key: value.payload,
            "type": value.kind,
            "mtime": value.mtime,
        }
        if value.key == "tensor_entry":
            source = cast(JsonEntry, value.payload).tensor_source
            if source is None:
                raise fuse_error(EIO)
            node["source"] = source.source_path
        if value.format is not None:
            node["format"] = value.format
        return node
