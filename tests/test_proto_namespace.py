"""Vendored descriptors must coexist with TensorBoard and TensorFlow protos."""

import subprocess
import sys

UPSTREAM_REGISTRATION = """
from google.protobuf import descriptor_pb2, descriptor_pool

import tboardfs.proto.compat_pb2
import tboardfs.proto.custom_scalar_layout_pb2
import tboardfs.proto.hparams_api_pb2
import tboardfs.proto.hparams_plugin_data_pb2
import tboardfs.proto.mesh_plugin_data_pb2

UPSTREAM = [
    ("tensorflow/core/util/event.proto", "tensorflow", "Event"),
    ("tensorflow/core/framework/summary.proto", "tensorflow", "Summary"),
    ("tensorflow/core/framework/graph.proto", "tensorflow", "GraphDef"),
    ("tensorflow/core/framework/node_def.proto", "tensorflow", "NodeDef"),
    ("tensorflow/core/framework/attr_value.proto", "tensorflow", "AttrValue"),
    ("tensorflow/core/framework/function.proto", "tensorflow", "FunctionDef"),
    ("tensorflow/core/framework/op_def.proto", "tensorflow", "OpDef"),
    ("tensorflow/core/framework/versions.proto", "tensorflow", "VersionDef"),
    ("tensorflow/core/protobuf/config.proto", "tensorflow", "RunMetadata"),
    ("tensorboard/plugins/hparams/api.proto", "tensorboard.hparams", "Experiment"),
    (
        "tensorboard/plugins/hparams/plugin_data.proto",
        "tensorboard.hparams",
        "HParamsPluginData",
    ),
    ("tensorboard/plugins/mesh/plugin_data.proto", "tensorboard.mesh", "MeshPluginData"),
    ("tensorboard/plugins/custom_scalar/layout.proto", "tensorboard", "Layout"),
    ("tensorboard/compat/proto/event.proto", "tensorboard", "Event"),
    ("tensorboard/compat/proto/summary.proto", "tensorboard", "Summary"),
    ("tensorboard/compat/proto/config.proto", "tensorboard", "RunMetadata"),
    ("tensorboard/compat/proto/meta_graph.proto", "tensorboard", "MetaGraphDef"),
    ("tensorboard/compat/proto/graph.proto", "tensorboard", "GraphDef"),
]
pool = descriptor_pool.Default()
for name, package, message in UPSTREAM:
    proto = descriptor_pb2.FileDescriptorProto(name=name, package=package, syntax="proto3")
    proto.message_type.add(name=message)
    pool.AddSerializedFile(proto.SerializeToString())
print("ok")
"""


def test_vendored_descriptors_do_not_claim_upstream_names() -> None:
    """Upstream TensorBoard/TensorFlow descriptors register after tboardfs."""
    result = subprocess.run(
        [sys.executable, "-c", UPSTREAM_REGISTRATION],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
