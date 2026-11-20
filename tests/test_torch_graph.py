"""Read the PyTorch add_graph fixture without importing PyTorch or TensorBoard."""

from test_tf_producer_cases import CORPUS

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


def test_native_pytorch_trace_preserves_classifier_dataflow() -> None:
    fs = TensorBoardFS(CORPUS / "torch-graph")
    graph = compat_pb2.GraphDef.FromString(fs.read("/graphs/graph.pb", 65536, 0))
    nodes = {node.name: node for node in graph.node}
    (linear,) = (node for node in graph.node if node.op == "aten::linear")
    (relu,) = (node for node in graph.node if node.op == "aten::relu")
    (output,) = (node for node in graph.node if node.name.startswith("output/"))
    source, weight, bias = (nodes[name] for name in linear.input)
    assert source.name.startswith("input/") and source.op == "IO Node"
    assert weight.op == "prim::GetAttr" and weight.name.endswith("/weight")
    assert bias.op == "prim::Constant"
    assert list(relu.input) == [linear.name]
    assert list(output.input) == [relu.name]
    for node, shape in (
        (source, [2, 2]),
        (linear, [2, 1]),
        (relu, [2, 1]),
        (output, [2, 1]),
    ):
        (tensor,) = node.attr["_output_shapes"].list.shape
        assert [dim.size for dim in tensor.dim] == shape


def test_native_pytorch_trace_exports_cpu_profile_tag() -> None:
    fs = TensorBoardFS(CORPUS / "torch-graph")
    stem = "/graphs/step1/000000.run_metadata"
    metadata = compat_pb2.RunMetadata.FromString(fs.read(f"{stem}.pb", 65536, 0))
    (device,) = metadata.step_stats.dev_stats
    assert device.device == "/device:CPU:0"
    assert not device.node_stats
    assert (
        fs.read(f"{stem}.pbtxt", 65536, 0)
        == b'step_stats {\n  dev_stats {\n    device: "/device:CPU:0"\n  }\n}\n'
    )
