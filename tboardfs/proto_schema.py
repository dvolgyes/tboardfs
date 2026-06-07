from typing import cast

from google.protobuf.message import Message

from tboardfs.proto import (
    custom_scalar_layout_pb2,
    event_pb2,
    hparams_plugin_data_pb2,
    mesh_plugin_data_pb2,
)


class _ProtoParser:
    """Parse vendored TensorBoard protobuf messages."""

    @staticmethod
    def event(data: bytes) -> Message:
        """Parse a TensorFlow Event message."""
        message = cast(Message, getattr(event_pb2, "Event")())
        message.ParseFromString(data)
        return message

    @staticmethod
    def custom_scalar_layout(data: bytes) -> Message:
        """Parse a TensorBoard custom scalars Layout message."""
        message = cast(Message, getattr(custom_scalar_layout_pb2, "Layout")())
        message.ParseFromString(data)
        return message

    @staticmethod
    def mesh_plugin_data(data: bytes) -> Message:
        """Parse a TensorBoard mesh MeshPluginData message."""
        message = cast(Message, getattr(mesh_plugin_data_pb2, "MeshPluginData")())
        message.ParseFromString(data)
        return message

    @staticmethod
    def hparams_plugin_data(data: bytes) -> Message:
        """Parse a TensorBoard hparams HParamsPluginData message."""
        message = cast(Message, getattr(hparams_plugin_data_pb2, "HParamsPluginData")())
        message.ParseFromString(data)
        return message
