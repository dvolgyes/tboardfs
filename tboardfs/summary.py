from typing import Any, cast

from tboardfs.proto_schema import protobuf_message_from_bytes


def parse_event(data: bytes) -> dict[str, Any]:
    """Parse a TensorBoard Event protobuf with the local raw protobuf schema."""
    return _SummaryParser.parse_event(data)


class _SummaryParser:
    """Convert TensorBoard protobuf messages to parser dictionaries."""

    @staticmethod
    def parse_event(data: bytes) -> dict[str, Any]:
        """Parse one Event protobuf payload."""
        event_message = cast(Any, protobuf_message_from_bytes(data, "tensorflow.Event"))
        event: dict[str, Any] = {}
        if event_message.HasField("wall_time"):
            event["wall_time"] = event_message.wall_time
        if event_message.HasField("step"):
            event["step"] = event_message.step
        if event_message.HasField("graph_def"):
            event["graph_def"] = event_message.graph_def
        if event_message.HasField("summary"):
            event["values"] = [
                _SummaryParser.parse_summary_value(value)
                for value in event_message.summary.value
            ]
        return event

    @staticmethod
    def parse_summary_value(value_message: object) -> dict[str, Any]:
        """Convert a SummaryValue protobuf message to the internal shape."""
        message = cast(Any, value_message)
        value: dict[str, Any] = {}
        if message.HasField("tag"):
            value["tag"] = message.tag
        if message.HasField("simple_value"):
            value["simple_value"] = message.simple_value
        if message.HasField("image"):
            value["image"] = _SummaryParser.parse_image(message.image)
        if message.HasField("histo"):
            value["histo"] = _SummaryParser.parse_histogram(message.histo)
        if message.HasField("audio"):
            value["audio"] = _SummaryParser.parse_audio(message.audio)
        if message.HasField("tensor"):
            value["tensor"] = _SummaryParser.parse_tensor(message.tensor)
        if message.HasField("metadata"):
            value.update(_SummaryParser.parse_summary_metadata(message.metadata))
        return value

    @staticmethod
    def parse_image(image_message: object) -> dict[str, bytes]:
        """Convert an Image protobuf message to the internal shape."""
        message = cast(Any, image_message)
        image: dict[str, bytes] = {}
        if message.HasField("encoded_image_string"):
            image["encoded_image_string"] = message.encoded_image_string
        return image

    @staticmethod
    def parse_audio(audio_message: object) -> dict[str, bytes]:
        """Convert an Audio protobuf message to the internal shape."""
        message = cast(Any, audio_message)
        audio: dict[str, bytes] = {}
        if message.HasField("encoded_audio_string"):
            audio["encoded_audio_string"] = message.encoded_audio_string
        return audio

    @staticmethod
    def parse_histogram(histo_message: object) -> dict[str, Any]:
        """Convert a HistogramProto protobuf message to the internal shape."""
        message = cast(Any, histo_message)
        histo = {
            "bucket_limit": list(message.bucket_limit),
            "bucket": list(message.bucket),
        }
        for name in ("min", "max", "num", "sum", "sum_squares"):
            if message.HasField(name):
                histo[name] = getattr(message, name)
        return histo

    @staticmethod
    def parse_summary_metadata(metadata_message: object) -> dict[str, Any]:
        """Convert SummaryMetadata protobuf plugin fields to internal metadata."""
        message = cast(Any, metadata_message)
        metadata: dict[str, Any] = {}
        if not message.HasField("plugin_data"):
            return metadata
        plugin_data = message.plugin_data
        if plugin_data.HasField("plugin_name"):
            metadata["plugin_name"] = plugin_data.plugin_name
        if plugin_data.HasField("content"):
            metadata["plugin_content"] = plugin_data.content
        return metadata

    @staticmethod
    def parse_tensor(tensor_message: object) -> dict[str, Any]:
        """Convert TensorProto protobuf scalar/blob fields to internal shape."""
        message = cast(Any, tensor_message)
        tensor: dict[str, Any] = {}
        if message.HasField("dtype"):
            tensor["dtype"] = message.dtype
        if message.HasField("tensor_shape"):
            tensor["shape"] = [
                dim.size for dim in message.tensor_shape.dim if dim.HasField("size")
            ]
        if message.HasField("tensor_content"):
            tensor["tensor_content"] = message.tensor_content
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
            (10, "int64_val"),
            (11, "bool_val"),
            (16, "uint32_val"),
            (17, "uint64_val"),
        )
        for number, name in field_map:
            repeated = list(getattr(message, name))
            if repeated:
                values[number] = repeated
        return values
