"""Load the vendored descriptors and resolve their generated message classes."""

from importlib import import_module

from google.protobuf import descriptor_pool, message_factory
from google.protobuf.descriptor import Descriptor
from google.protobuf.message import Message

from tboardfs.proto import (
    custom_scalar_layout_pb2,
    hparams_plugin_data_pb2,
    mesh_plugin_data_pb2,
)


def _message_type(descriptor: Descriptor) -> type[Message]:
    """Return the concrete class already built for this descriptor."""
    cls: type[Message] = message_factory.GetMessageClass(descriptor)
    return cls


import_module("tboardfs.proto.compat_pb2")
_pool = descriptor_pool.Default()
_PACKAGE = "tboardfs.vendored.tensorboard"

EVENT = _message_type(_pool.FindMessageTypeByName(f"{_PACKAGE}.Event"))
GRAPH_DEF = _message_type(_pool.FindMessageTypeByName(f"{_PACKAGE}.GraphDef"))
META_GRAPH_DEF = _message_type(_pool.FindMessageTypeByName(f"{_PACKAGE}.MetaGraphDef"))
RUN_METADATA = _message_type(_pool.FindMessageTypeByName(f"{_PACKAGE}.RunMetadata"))
LAYOUT = _message_type(
    custom_scalar_layout_pb2.DESCRIPTOR.message_types_by_name["Layout"]
)
MESH_PLUGIN_DATA = _message_type(
    mesh_plugin_data_pb2.DESCRIPTOR.message_types_by_name["MeshPluginData"]
)
HPARAMS_PLUGIN_DATA = _message_type(
    hparams_plugin_data_pb2.DESCRIPTOR.message_types_by_name["HParamsPluginData"]
)
DT_FLOAT: int = (
    _pool.FindEnumTypeByName(f"{_PACKAGE}.DataType").values_by_name["DT_FLOAT"].number
)
