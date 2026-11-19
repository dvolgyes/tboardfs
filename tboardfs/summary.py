from typing import Any, cast

from tboardfs.proto_schema import _ProtoParser
from tboardfs.constants import DT_STRING
from tboardfs.tensor_summary import TAGGED_RUN_METADATA

# SessionLog.SessionStatus.START in the vendored event proto.
START = 1


def parse_event(data: bytes) -> dict[str, Any]:
    """Parse a TensorBoard Event protobuf with vendored protobuf classes."""
    return _SummaryParser.parse_event(data)


class _SummaryParser:
    """Convert TensorBoard protobuf messages to parser dictionaries."""

    @staticmethod
    def parse_event(data: bytes) -> dict[str, Any]:
        """Parse one Event protobuf payload."""
        event_message = cast(Any, _ProtoParser.event(data))
        event: dict[str, Any] = {
            "wall_time": event_message.wall_time,
            "step": event_message.step,
        }
        kind = event_message.WhichOneof("what")
        if kind == "summary":
            event["values"] = [
                _SummaryParser.parse_summary_value(value)
                for value in event_message.summary.value
            ]
        elif kind == "tagged_run_metadata":
            event["values"] = [
                _SummaryParser.tagged_run_metadata_value(
                    event_message.tagged_run_metadata
                )
            ]
        elif kind in ("file_version", "graph_def", "meta_graph_def"):
            event[kind] = getattr(event_message, kind)
        elif kind == "session_log" and event_message.session_log.status == START:
            event["session_start"] = True
        return event

    @staticmethod
    def tagged_run_metadata_value(tagged_message: object) -> dict[str, Any]:
        """Return a TF1 tagged RunMetadata as the summary value TensorBoard reads.

        TensorBoard's dataclass_compat migrates it to a string tensor that
        holds the serialized RunMetadata, with plugin graph_tagged_run_metadata.
        """
        message = cast(Any, tagged_message)
        return {
            "tag": message.tag,
            "tensor": {
                "dtype": DT_STRING,
                "shape": [1],
                "string_val": [message.run_metadata],
            },
            "plugin_name": TAGGED_RUN_METADATA,
        }

    @staticmethod
    def parse_summary_value(value_message: object) -> dict[str, Any]:
        """Convert a SummaryValue protobuf message to the internal shape."""
        message = cast(Any, value_message)
        value: dict[str, Any] = {"tag": message.tag}
        # The value fields are one oneof, so one call finds the field set.
        kind = message.WhichOneof("value")
        if kind == "simple_value":
            value["simple_value"] = message.simple_value
        elif kind == "image":
            value["image"] = _SummaryParser.parse_image(message.image)
        elif kind == "histo":
            value["histo"] = _SummaryParser.parse_histogram(message.histo)
        elif kind == "audio":
            value["audio"] = _SummaryParser.parse_audio(message.audio)
        elif kind == "tensor":
            if not message.tag:
                value["tag"] = message.node_name
            value["tensor"] = _SummaryParser.parse_tensor(message.tensor)
        if message.HasField("metadata"):
            value.update(_SummaryParser.parse_summary_metadata(message.metadata))
        return value

    @staticmethod
    def parse_image(image_message: object) -> dict[str, bytes]:
        """Convert an Image protobuf message to the internal shape."""
        message = cast(Any, image_message)
        return {"encoded_image_string": message.encoded_image_string}

    @staticmethod
    def parse_audio(audio_message: object) -> dict[str, bytes]:
        """Convert an Audio protobuf message to the internal shape."""
        message = cast(Any, audio_message)
        return {"encoded_audio_string": message.encoded_audio_string}

    @staticmethod
    def parse_histogram(histo_message: object) -> dict[str, Any]:
        """Convert a HistogramProto protobuf message to the internal shape."""
        message = cast(Any, histo_message)
        histo: dict[str, Any] = {
            "bucket_limit": list(message.bucket_limit),
            "bucket": list(message.bucket),
        }
        for name in ("min", "max", "num", "sum", "sum_squares"):
            histo[name] = getattr(message, name)
        return histo

    @staticmethod
    def parse_summary_metadata(metadata_message: object) -> dict[str, Any]:
        """Convert SummaryMetadata protobuf plugin fields to internal metadata."""
        message = cast(Any, metadata_message)
        if not message.HasField("plugin_data"):
            return {}
        plugin_data = message.plugin_data
        return {
            "plugin_name": plugin_data.plugin_name,
            "plugin_content": plugin_data.content,
        }

    @staticmethod
    def parse_tensor(tensor_message: object) -> dict[str, Any]:
        """Convert TensorProto protobuf scalar/blob fields to internal shape."""
        message = cast(Any, tensor_message)
        tensor: dict[str, Any] = {"dtype": message.dtype}
        if message.HasField("tensor_shape"):
            tensor["shape"] = [dim.size for dim in message.tensor_shape.dim]
            if message.tensor_shape.unknown_rank:
                tensor["unknown_rank"] = True
        if content := message.tensor_content:
            tensor["tensor_content"] = content
        values = _SummaryParser.parse_tensor_values(message)
        if values:
            tensor["values"] = values
        if message.string_val:
            tensor["string_val"] = list(message.string_val)
        return tensor

    @staticmethod
    def parse_tensor_values(tensor_message: object) -> dict[int, list[Any]]:
        """Convert TensorProto repeated scalar fields to legacy field-number keys."""
        message = cast(Any, tensor_message)
        values: dict[int, list[Any]] = {}
        field_map = (
            (5, "float_val"),
            (6, "double_val"),
            (7, "int_val"),
            (9, "scomplex_val"),
            (10, "int64_val"),
            (11, "bool_val"),
            (12, "dcomplex_val"),
            (13, "half_val"),
            (16, "uint32_val"),
            (17, "uint64_val"),
        )
        for number, name in field_map:
            repeated = list(getattr(message, name))
            if repeated:
                values[number] = repeated
        return values
