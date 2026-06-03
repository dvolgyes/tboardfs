from typing import cast

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, struct_pb2


def protobuf_message_from_bytes(data: bytes, full_name: str) -> object:
    """Parse bytes into a protobuf message from the local TensorBoard schema."""
    message_class = _SchemaBuilder.message_class(full_name)
    message = message_class()
    message.ParseFromString(data)
    return message


class _SchemaBuilder:
    """Build the TensorBoard protobuf subset used by the parser."""

    @staticmethod
    def pool() -> descriptor_pool.DescriptorPool:
        """Build a descriptor pool for the local TensorBoard schema."""
        schema_pool = descriptor_pool.DescriptorPool()
        schema_pool.AddSerializedFile(struct_pb2.DESCRIPTOR.serialized_pb)
        for descriptor in _SchemaBuilder.file_descriptors():
            schema_pool.Add(descriptor)
        return schema_pool

    @staticmethod
    def message_class(full_name: str) -> type:
        """Return a dynamic protobuf message class by fully qualified name."""
        return cast(
            type,
            message_factory.GetMessageClass(
                _SchemaBuilder.pool().FindMessageTypeByName(full_name)
            ),
        )

    @staticmethod
    def file_descriptors() -> list[descriptor_pb2.FileDescriptorProto]:
        """Return all local TensorBoard protobuf descriptors."""
        return [
            _SchemaBuilder.event_descriptor(),
            _SchemaBuilder.custom_scalar_descriptor(),
            _SchemaBuilder.mesh_plugin_descriptor(),
            _SchemaBuilder.hparams_api_descriptor(),
            _SchemaBuilder.hparams_plugin_descriptor(),
        ]

    @staticmethod
    def event_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the local TensorBoard event protobuf descriptor."""
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
    def custom_scalar_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the TensorBoard custom scalar layout descriptor subset."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.name = "tensorboard/plugins/custom_scalar/layout.proto"
        proto.package = "tensorboard"
        proto.syntax = "proto3"
        add = _SchemaBuilder.add_field

        chart = proto.message_type.add(name="Chart")
        chart.oneof_decl.add(name="content")
        add(chart, "title", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        field = add(
            chart,
            "multiline",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.MultilineChartContent",
        )
        field.oneof_index = 0
        field = add(
            chart,
            "margin",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.MarginChartContent",
        )
        field.oneof_index = 0

        multiline = proto.message_type.add(name="MultilineChartContent")
        add(
            multiline,
            "tag",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )

        margin = proto.message_type.add(name="MarginChartContent")
        add(
            margin,
            "series",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.MarginChartContent.Series",
        )
        series = margin.nested_type.add(name="Series")
        add(series, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(series, "lower", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(series, "upper", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        category = proto.message_type.add(name="Category")
        add(category, "title", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            category,
            "chart",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.Chart",
        )
        add(category, "closed", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)

        layout = proto.message_type.add(name="Layout")
        add(layout, "version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(
            layout,
            "category",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.Category",
        )
        return proto

    @staticmethod
    def mesh_plugin_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the TensorBoard mesh plugin data descriptor subset."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.name = "tensorboard/plugins/mesh/plugin_data.proto"
        proto.package = "tensorboard.mesh"
        proto.syntax = "proto3"
        add = _SchemaBuilder.add_field

        mesh = proto.message_type.add(name="MeshPluginData")
        _SchemaBuilder.add_enum(
            mesh.enum_type.add(name="ContentType"),
            (
                ("UNDEFINED", 0),
                ("VERTEX", 1),
                ("FACE", 2),
                ("COLOR", 3),
            ),
        )
        add(mesh, "version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        add(mesh, "name", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            mesh,
            "content_type",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            type_name=".tensorboard.mesh.MeshPluginData.ContentType",
        )
        add(mesh, "json_config", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            mesh,
            "shape",
            6,
            descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        )
        add(mesh, "components", 7, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
        return proto

    @staticmethod
    def hparams_api_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the TensorBoard hparams API descriptor subset."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.name = "tensorboard/plugins/hparams/api.proto"
        proto.package = "tensorboard.hparams"
        proto.syntax = "proto3"
        proto.dependency.append("google/protobuf/struct.proto")
        add = _SchemaBuilder.add_field

        _SchemaBuilder.add_enum(
            proto.enum_type.add(name="DataType"),
            (
                ("DATA_TYPE_UNSET", 0),
                ("DATA_TYPE_STRING", 1),
                ("DATA_TYPE_BOOL", 2),
                ("DATA_TYPE_FLOAT64", 3),
            ),
        )
        _SchemaBuilder.add_enum(
            proto.enum_type.add(name="DatasetType"),
            (
                ("DATASET_UNKNOWN", 0),
                ("DATASET_TRAINING", 1),
                ("DATASET_VALIDATION", 2),
            ),
        )
        _SchemaBuilder.add_enum(
            proto.enum_type.add(name="Status"),
            (
                ("STATUS_UNKNOWN", 0),
                ("STATUS_SUCCESS", 1),
                ("STATUS_FAILURE", 2),
                ("STATUS_RUNNING", 3),
            ),
        )

        experiment = proto.message_type.add(name="Experiment")
        add(experiment, "name", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            experiment,
            "description",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )
        add(experiment, "user", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            experiment,
            "time_created_secs",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
        )
        add(
            experiment,
            "hparam_infos",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.hparams.HParamInfo",
        )
        add(
            experiment,
            "metric_infos",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.hparams.MetricInfo",
        )

        hparam = proto.message_type.add(name="HParamInfo")
        hparam.oneof_decl.add(name="domain")
        add(hparam, "name", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(hparam, "display_name", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(hparam, "description", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            hparam,
            "type",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            type_name=".tensorboard.hparams.DataType",
        )
        field = add(
            hparam,
            "domain_discrete",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".google.protobuf.ListValue",
        )
        field.oneof_index = 0
        field = add(
            hparam,
            "domain_interval",
            6,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.hparams.Interval",
        )
        field.oneof_index = 0
        add(hparam, "differs", 7, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)

        interval = proto.message_type.add(name="Interval")
        add(interval, "min_value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
        add(interval, "max_value", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

        metric_name = proto.message_type.add(name="MetricName")
        add(metric_name, "group", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(metric_name, "tag", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

        metric_info = proto.message_type.add(name="MetricInfo")
        add(
            metric_info,
            "name",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.hparams.MetricName",
        )
        add(
            metric_info,
            "display_name",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )
        add(
            metric_info,
            "description",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )
        add(
            metric_info,
            "dataset_type",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            type_name=".tensorboard.hparams.DatasetType",
        )
        return proto

    @staticmethod
    def hparams_plugin_descriptor() -> descriptor_pb2.FileDescriptorProto:
        """Return the TensorBoard hparams plugin data descriptor subset."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.name = "tensorboard/plugins/hparams/plugin_data.proto"
        proto.package = "tensorboard.hparams"
        proto.syntax = "proto3"
        proto.dependency.extend(
            [
                "tensorboard/plugins/hparams/api.proto",
                "google/protobuf/struct.proto",
            ]
        )
        add = _SchemaBuilder.add_field

        plugin = proto.message_type.add(name="HParamsPluginData")
        plugin.oneof_decl.add(name="data")
        add(plugin, "version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT32)
        field = add(
            plugin,
            "experiment",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.hparams.Experiment",
        )
        field.oneof_index = 0
        field = add(
            plugin,
            "session_start_info",
            3,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.hparams.SessionStartInfo",
        )
        field.oneof_index = 0
        field = add(
            plugin,
            "session_end_info",
            4,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".tensorboard.hparams.SessionEndInfo",
        )
        field.oneof_index = 0

        start = proto.message_type.add(name="SessionStartInfo")
        entry = start.nested_type.add(name="HparamsEntry")
        entry.options.map_entry = True
        add(entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            entry,
            "value",
            2,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=".google.protobuf.Value",
        )
        add(
            start,
            "hparams",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type_name=".tensorboard.hparams.SessionStartInfo.HparamsEntry",
        )
        add(start, "model_uri", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(start, "monitor_url", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(start, "group_name", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add(
            start,
            "start_time_secs",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
        )

        end = proto.message_type.add(name="SessionEndInfo")
        add(
            end,
            "status",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            type_name=".tensorboard.hparams.Status",
        )
        add(end, "end_time_secs", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
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
    ) -> descriptor_pb2.FieldDescriptorProto:
        """Add one field descriptor to a protobuf message descriptor."""
        field = message.field.add()
        field.name = name
        field.number = number
        field.label = label
        field.type = field_type
        if type_name:
            field.type_name = type_name
        return field

    @staticmethod
    def add_enum(
        enum: descriptor_pb2.EnumDescriptorProto,
        values: tuple[tuple[str, int], ...],
    ) -> None:
        """Add enum value descriptors."""
        for name, number in values:
            value = enum.value.add()
            value.name = name
            value.number = number
