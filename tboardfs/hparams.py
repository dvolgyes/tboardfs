from typing import Any, cast

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Value
import numpy as np

from tboardfs.model import JsonEntry
from tboardfs.proto_schema import _ProtoParser
from tboardfs.tables import _TableExport


def hparams_json(
    entries: list[JsonEntry],
    scalars: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Return structured hparams metadata and final metric values."""
    parsed = [_parse_hparams(entry) for entry in entries]
    return {
        "experiment": next(
            (item["experiment"] for item in parsed if item.get("experiment")),
            {},
        ),
        "session_start": next(
            (item["session_start"] for item in parsed if item.get("session_start")),
            {},
        ),
        "session_end": next(
            (item["session_end"] for item in parsed if item.get("session_end")),
            {},
        ),
        "metrics": {
            tag: _TableExport.json_safe(series["value"][-1])
            for tag, series in sorted(scalars.items())
            if tag.startswith("hparam/")
        },
    }


def _parse_hparams(entry: JsonEntry) -> dict[str, Any]:
    """Return one parsed hparams plugin record."""
    content = entry.payload.get("plugin_content")
    if not isinstance(content, bytes):
        return {}
    data = cast(Any, _ProtoParser.hparams_plugin_data(content))
    out: dict[str, Any] = {}
    if data.HasField("experiment"):
        out["experiment"] = cast(
            dict[str, Any],
            MessageToDict(data.experiment, preserving_proto_field_name=True),
        )
    if data.HasField("session_start_info"):
        hparams: dict[str, object] = {}
        for name, value in sorted(data.session_start_info.hparams.items()):
            typed_value = cast(Value, value)
            kind = typed_value.WhichOneof("kind")
            values = {
                "number_value": typed_value.number_value,
                "string_value": typed_value.string_value,
                "bool_value": typed_value.bool_value,
                "null_value": None,
            }
            if kind in values:
                hparams[name] = values[kind]
            else:
                hparams[name] = cast(
                    object,
                    MessageToDict(typed_value, preserving_proto_field_name=True),
                )
        out["session_start"] = {
            "group_name": data.session_start_info.group_name,
            "start_time_secs": data.session_start_info.start_time_secs,
            "hparams": hparams,
        }
    if data.HasField("session_end_info"):
        out["session_end"] = cast(
            dict[str, Any],
            MessageToDict(data.session_end_info, preserving_proto_field_name=True),
        )
    return out
