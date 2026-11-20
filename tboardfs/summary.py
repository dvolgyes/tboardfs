from math import prod
from typing import Any, cast

from tboardfs.constants import DT_FLOAT, DT_STRING, LARGE_TENSOR_THRESHOLD
from tboardfs.proto_schema import _ProtoParser
from tboardfs.tensor_summary import TAGGED_RUN_METADATA

# SessionLog.SessionStatus.START in the vendored event proto.
START = 1

_TENSOR_VALUE_FIELDS = (
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
_PACKED_FIELDS = frozenset(("dtype", "shape", "tensor_content"))
_MAX_TENSOR_RANK = 254


def parse_event(data: bytes) -> dict[str, Any]:
    """Parse a TensorBoard Event protobuf with vendored protobuf classes."""
    return _SummaryParser.parse_event(data)


class _SummaryParser:
    """Convert TensorBoard protobuf messages to parser dictionaries."""

    @staticmethod
    def parse_event(data: bytes, *, indexing: bool = False) -> dict[str, Any]:
        """Parse one Event protobuf payload."""
        message = cast(Any, _ProtoParser.event(data))
        return _SummaryParser.convert_event(
            message, message.WhichOneof("what"), indexing=indexing
        )

    @staticmethod
    def convert_event(
        message: object, kind: str | None, *, indexing: bool = False
    ) -> dict[str, Any]:
        """Convert an already decoded event, including all values before indexing."""
        event_message = cast(Any, message)
        event: dict[str, Any] = {
            "wall_time": event_message.wall_time,
            "step": event_message.step,
        }
        if kind == "summary":
            event["values"] = [
                _SummaryParser.parse_summary_value(value, indexing=indexing)
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
    def parse_summary_value(
        value_message: object, *, indexing: bool = False
    ) -> dict[str, Any]:
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
            tensor = _SummaryParser.parse_tensor(message.tensor)
            value["tensor"] = tensor
            if (
                indexing
                and tensor["dtype"] == DT_FLOAT
                and _SummaryParser.packed_float32(tensor, message.tensor)
            ):
                value["packed_float32"] = True
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
            shape = message.tensor_shape
            tensor["shape"] = [dim.size for dim in shape.dim]
            if shape.unknown_rank:
                tensor["unknown_rank"] = True
        if content := message.tensor_content:
            tensor["tensor_content"] = content
        values = {
            number: list(repeated)
            for number, name in _TENSOR_VALUE_FIELDS
            if (repeated := getattr(message, name))
        }
        if values:
            tensor["values"] = values
        if strings := message.string_val:
            tensor["string_val"] = list(strings)
        return tensor

    @staticmethod
    def packed_float32(tensor: dict[str, Any], tensor_message: object) -> bool:
        """Trust the numeric layout only when it accounts for the whole raw export."""
        if tensor.keys() != _PACKED_FIELDS:
            return False
        size = len(tensor["tensor_content"])
        shape = tensor["shape"]
        if size < LARGE_TENSOR_THRESHOLD or not 0 < len(shape) <= _MAX_TENSOR_RANK:
            return False
        if min(shape) <= 0 or 4 * prod(shape) != size:
            return False
        message = cast(Any, tensor_message)
        return not (
            message.resource_handle_val or message.variant_val or message.float8_val
        )
