from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def event_message_from_bytes(data: bytes) -> object:
    """Parse TensorBoard Event bytes with a local protobuf schema."""
    event_class = message_factory.GetMessageClass(
        _SchemaBuilder.pool().FindMessageTypeByName("tensorflow.Event")
    )
    event = event_class()
    event.ParseFromString(data)
    return event


class _SchemaBuilder:
    """Build the TensorBoard protobuf subset used by the parser."""

    @staticmethod
    def pool() -> descriptor_pool.DescriptorPool:
        """Build a descriptor pool for the local TensorBoard schema."""
        schema_pool = descriptor_pool.DescriptorPool()
        schema_pool.Add(_SchemaBuilder.file_descriptor())
        return schema_pool

    @staticmethod
    def file_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the local TensorBoard protobuf descriptor."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.name = "tboardfs_tensorboard_subset.proto"
        proto.package = "tensorflow"
        proto.syntax = "proto2"
        add = _SchemaBuilder.add_field

        event = proto.message_type.add(name="Event")
        add(event, "wall_time", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        add(event, "step", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
        add(event, "graph_def", 4, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
        add(
            event,
            "summary",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.Summary",
        )

        summary = proto.message_type.add(name="Summary")
        add(
            summary,
            "value",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorflow.SummaryValue",
        )

        value = proto.message_type.add(name="SummaryValue")
        add(value, "tag", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(value, "simple_value", 2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
        add(
            value,
            "image",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.Image",
        )
        add(
            value,
            "histo",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.HistogramProto",
        )
        add(
            value,
            "audio",
            6,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.Audio",
        )
        add(
            value,
            "tensor",
            8,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.TensorProto",
        )
        add(
            value,
            "metadata",
            9,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.SummaryMetadata",
        )

        metadata = proto.message_type.add(name="SummaryMetadata")
        add(
            metadata,
            "plugin_data",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.PluginData",
        )

        plugin = proto.message_type.add(name="PluginData")
        add(plugin, "plugin_name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(plugin, "content", 2, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)

        image = proto.message_type.add(name="Image")
        add(image, "height", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(image, "width", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(image, "colorspace", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(
            image,
            "encoded_image_string",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        )

        audio = proto.message_type.add(name="Audio")
        add(audio, "sample_rate", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
        add(audio, "num_channels", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
        add(audio, "length_frames", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
        add(
            audio,
            "encoded_audio_string",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        )
        add(audio, "content_type", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        histogram = proto.message_type.add(name="HistogramProto")
        for name, number in (
            ("min", 1),
            ("max", 2),
            ("num", 3),
            ("sum", 4),
            ("sum_squares", 5),
        ):
            add(
                histogram, name, number, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
            )
        add(
            histogram,
            "bucket_limit",
            6,
            descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )
        add(
            histogram,
            "bucket",
            7,
            descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )

        tensor = proto.message_type.add(name="TensorProto")
        add(tensor, "dtype", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(
            tensor,
            "tensor_shape",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorflow.TensorShapeProto",
        )
        add(
            tensor,
            "tensor_content",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
        )
        for name, number, field_type in (
            ("float_val", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
            ("double_val", 6, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
            ("int_val", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
            ("string_val", 8, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
            ("int64_val", 10, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
            ("bool_val", 11, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
            ("uint32_val", 16, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32),
            ("uint64_val", 17, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64),
        ):
            add(
                tensor,
                name,
                number,
                field_type,
                label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            )
        shape = proto.message_type.add(name="TensorShapeProto")
        add(
            shape,
            "dim",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorflow.TensorShapeDim",
        )
        dim = proto.message_type.add(name="TensorShapeDim")
        add(dim, "size", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
        return proto

    @staticmethod
    def add_field(
        message: descriptor_pb2.DescriptorProto,
        name: str,
        number: int,
        field_type: int,
        *,
        label: int = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type_name: str = "",
    ) -> None:
        """Add one field descriptor to a protobuf message descriptor."""
        field = message.field.add()
        field.name = name
        field.number = number
        field.label = label
        field.type = field_type
        if type_name:
            field.type_name = type_name
