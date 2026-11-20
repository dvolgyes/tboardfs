from google.protobuf.message import Message

from tboardfs.proto import _messages


class _ProtoParser:
    """Parse vendored TensorBoard protobuf messages."""

    @staticmethod
    def event(data: bytes) -> Message:
        """Parse a TensorFlow Event message."""
        message: Message = _messages.EVENT()
        message.ParseFromString(data)
        return message

    @staticmethod
    def custom_scalar_layout(data: bytes) -> Message:
        """Parse a TensorBoard custom scalars Layout message."""
        message: Message = _messages.LAYOUT()
        message.ParseFromString(data)
        return message

    @staticmethod
    def mesh_plugin_data(data: bytes) -> Message:
        """Parse a TensorBoard mesh MeshPluginData message."""
        message: Message = _messages.MESH_PLUGIN_DATA()
        message.ParseFromString(data)
        return message

    @staticmethod
    def hparams_plugin_data(data: bytes) -> Message:
        """Parse a TensorBoard hparams HParamsPluginData message."""
        message: Message = _messages.HPARAMS_PLUGIN_DATA()
        message.ParseFromString(data)
        return message
