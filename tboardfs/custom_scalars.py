from typing import Any, cast

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError

from tboardfs.model import JsonEntry
from tboardfs.proto_schema import protobuf_message_from_bytes


def custom_scalar_layout_json(entries: list[JsonEntry]) -> dict[str, Any]:
    """Return the first TensorBoard custom scalar layout in JSON-safe form."""
    for entry in entries:
        tensor = entry.payload.get("tensor") or {}
        blobs = tensor.get("string_val") or []
        for blob in blobs:
            if not isinstance(blob, bytes):
                continue
            try:
                layout = protobuf_message_from_bytes(blob, "tensorboard.Layout")
            except DecodeError:
                continue
            return cast(
                dict[str, Any],
                MessageToDict(layout, preserving_proto_field_name=True),
            )
    return {}
