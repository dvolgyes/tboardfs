import math
from typing import Any, cast

import numpy as np
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from google.protobuf.struct_pb2 import Value

from tboardfs.model import JsonEntry
from tboardfs.proto_schema import _ProtoParser
from tboardfs.tables import _TableExport


def hparams_json(
    entries: list[JsonEntry],
    scalars: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Return structured hparams metadata and final metric values.

    The rules follow the TensorBoard hparams plugin for one run. The first
    experiment, session start and session end records win; ``sessions`` lists
    every session. The declared metric_infos name the metrics, or all scalar
    tags when the run declares no experiment. A metric value is the last scalar
    value, or None without data. Metrics of a sub-run group have no value here.
    """
    parsed = [_HparamsRecords.parse(entry) for entry in entries]
    experiments = [item["experiment"] for item in parsed if "experiment" in item]
    experiment: dict[str, Any] = experiments[0] if experiments else {}
    return {
        "experiment": experiment,
        "session_start": next(
            (item["session_start"] for item in parsed if "session_start" in item),
            {},
        ),
        "session_end": next(
            (item["session_end"] for item in parsed if "session_end" in item),
            {},
        ),
        "sessions": _HparamsRecords.sessions(parsed),
        "metrics": _HparamsRecords.metrics(experiment, bool(experiments), scalars),
    }


class _HparamsRecords:
    """Parse hparams plugin records and combine them as TensorBoard does."""

    @staticmethod
    def sessions(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pair each session start with the end record that follows it."""
        sessions: list[dict[str, Any]] = []
        for item in parsed:
            if "session_start" in item:
                sessions.append(
                    {"session_start": item["session_start"], "session_end": {}}
                )
            if "session_end" in item:
                if sessions and not sessions[-1]["session_end"]:
                    sessions[-1]["session_end"] = item["session_end"]
                else:
                    sessions.append(
                        {"session_start": {}, "session_end": item["session_end"]}
                    )
        return sessions

    @staticmethod
    def metrics(
        experiment: dict[str, Any],
        declared: bool,
        scalars: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, object]:
        """Return the last value of each metric of this run, or None."""
        if declared:
            names = [
                info.get("name", {}) for info in experiment.get("metric_infos", [])
            ]
            tags = [name.get("tag", "") for name in names if not name.get("group")]
        else:
            tags = sorted(scalars)
        return {
            tag: _TableExport.json_safe(scalars[tag]["value"][-1])
            if tag in scalars
            else None
            for tag in tags
        }

    @staticmethod
    def spell_non_finite(message: Message) -> None:
        """Replace each NaN or infinite struct Value number by its JSON spelling.

        MessageToDict cannot write such a google.protobuf.Value, because the
        JSON text would parse as a string; the hparams JSON files spell them
        "NaN", "Infinity" and "-Infinity" as strings.
        """
        for field, value in message.ListFields():
            if field.message_type is None:
                continue
            if field.message_type.GetOptions().map_entry:
                items = list(value.values())
            elif field.is_repeated:
                items = list(value)
            else:
                items = [value]
            for item in items:
                if (
                    isinstance(item, Value)
                    and item.WhichOneof("kind") == "number_value"
                    and not math.isfinite(item.number_value)
                ):
                    item.string_value = cast(
                        str, _TableExport.json_value(item.number_value, "NaN")
                    )
                elif isinstance(item, Message):
                    _HparamsRecords.spell_non_finite(item)

    @staticmethod
    def parse(entry: JsonEntry) -> dict[str, Any]:
        """Return one parsed hparams plugin record."""
        content = entry.payload.get("plugin_content")
        if not isinstance(content, bytes):
            return {}
        data = cast(Any, _ProtoParser.hparams_plugin_data(content))
        _HparamsRecords.spell_non_finite(data)
        out: dict[str, Any] = {}
        if data.HasField("experiment"):
            out["experiment"] = cast(
                dict[str, Any],
                MessageToDict(
                    data.experiment,
                    always_print_fields_with_no_presence=True,
                    preserving_proto_field_name=True,
                ),
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
                MessageToDict(
                    data.session_end_info,
                    always_print_fields_with_no_presence=True,
                    preserving_proto_field_name=True,
                ),
            )
        return out
