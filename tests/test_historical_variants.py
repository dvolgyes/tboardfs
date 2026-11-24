"""Exercise the additional pinned TF/TB 1.15 and 2.0 native writer outputs."""

import json
import struct
from io import BytesIO
from pathlib import Path
from pickletools import genops

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).parent / "data/producer-cases/historical-variants"


@pytest.fixture(params=["tf115", "tf200"])
def run(request: pytest.FixtureRequest) -> Path:
    return ROOT / str(request.param)


def test_historical_unweighted_and_precomputed_pr(run: Path) -> None:
    fs = TensorBoardFS(run)
    for tag, middle in [
        ("unweighted", (0.5, 1, 1, 1, 1, 0.5, 0.5)),
        ("precomputed", (0.5, 2, 0, 2, 0, 1, 1)),
    ]:
        rows = json.loads(fs.read(f"/pr_curves/{tag}/pr_curves/000007.json", 4096, 0))
        assert [
            (
                row["threshold"],
                row["true_positives"],
                row["false_positives"],
                row["true_negatives"],
                row["false_negatives"],
                row["precision"],
                row["recall"],
            )
            for row in rows
        ] == [(0, 2, 2, 0, 0, 0.5, 1), middle, (1, 0, 0, 2, 2, 0, 0)]


def test_historical_mesh_and_cloud_batches(run: Path) -> None:
    fs = TensorBoardFS(run)
    vertices = [[[0, 0, 0], [2, 0, 0], [0, 1, 0]], [[0, 0, 1], [3, 0, 1], [0, 2, 1]]]
    with np.load(BytesIO(fs.read("/meshes/surface_batch/000007.npz", 8192, 0))) as mesh:
        assert set(mesh.files) == {"vertices", "faces", "colors"}
        np.testing.assert_array_equal(mesh["vertices"], vertices)
        np.testing.assert_array_equal(mesh["faces"], [[[0, 1, 2]], [[0, 2, 1]]])
        np.testing.assert_array_equal(
            mesh["colors"], [[[255, 0, 0]] * 3, [[0, 0, 255]] * 3]
        )
    obj = fs.read("/meshes/surface_batch/000007.obj", 8192, 0).splitlines()
    assert [line for line in obj if line.startswith(b"f ")] == [b"f 1 2 3", b"f 4 6 5"]
    data = fs.read("/meshes/cloud_batch/000007.ply", 8192, 0)
    header, body = data.split(b"end_header\n")
    assert b"element vertex 6" in header and b"element face" not in header
    assert list(struct.iter_unpack("<3f", body)) == [
        tuple(v) for batch in vertices for v in batch
    ]


def test_historical_rank_three_text_and_exact_tensor_payloads(run: Path) -> None:
    fs = TensorBoardFS(run, raw=True)
    cube = json.loads(fs.read("/text/cube/000007.json", 4096, 0))
    assert cube["shape"] == [2, 1, 2]
    assert cube["values"] == [[["one", "two"]], [["三", "x|y"]]]
    for tag, expected in [
        ("wide", np.array([1 + 2**-40, -2.5], np.float64)),
        ("phase", np.array([1 + 2j, -3 - 4j], np.complex64)),
    ]:
        array = np.load(
            BytesIO(fs.read(f"/tensors/{tag}/000007.npy", 4096, 0)), allow_pickle=False
        )
        assert array.dtype == expected.dtype
        np.testing.assert_array_equal(array, expected)
    raw = "/raw/events.out.tfevents.summaries"
    (opaque,) = (
        value
        for name in fs.readdir(raw)[2:]
        for value in compat_pb2.Event.FromString(
            fs.read(f"{raw}/{name}", 16384, 0)[12:-4]
        ).summary.value
        if value.tag == "opaque"
    )
    assert opaque.tensor.dtype == compat_pb2.DT_STRING
    assert list(opaque.tensor.string_val) == [b"a\x00b", b"\xff\x80", b""]


def test_historical_hparams_margin_and_sequential_model(run: Path) -> None:
    fs = TensorBoardFS(run)
    hparams = json.loads(fs.read("/hparams/hparams.json", 8192, 0))
    params = {entry["name"]: entry for entry in hparams["experiment"]["hparam_infos"]}
    assert params["kind"]["domain_discrete"] == ["linear", "square"]
    assert params["enabled"]["domain_discrete"] == [False, True]
    assert params["rate"]["domain_interval"] == {"min_value": 0.125, "max_value": 0.5}
    assert params["bins"]["domain_interval"] == {"min_value": 2, "max_value": 5}
    assert hparams["session_start"] == {
        "group_name": "sweep",
        "start_time_secs": 11,
        "hparams": {"kind": "square", "enabled": True, "rate": 0.25, "bins": 3},
    }
    assert hparams["session_end"] == {"status": "STATUS_SUCCESS", "end_time_secs": 12}
    layout = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))
    assert layout["category"] == [
        {
            "title": "validation",
            "chart": [
                {
                    "title": "range",
                    "margin": {
                        "series": [{"value": "score", "lower": "low", "upper": "high"}]
                    },
                }
            ],
        }
    ]
    model = json.loads(fs.read("/graphs/model/000007.json", 8192, 0))
    assert model["class_name"] == "Sequential"
    assert model["config"]["name"] == "historical_sequential"
    assert [
        (layer["config"]["name"], layer["config"]["units"])
        for layer in model["config"]["layers"]
        if layer["class_name"] == "Dense"
    ] == [("hidden", 3), ("estimate", 1)]
    graph = compat_pb2.GraphDef.FromString(
        fs.read("/graphs/model/000007.conceptual.pb", 8192, 0)
    )
    hidden, estimate = graph.node
    assert hidden.attr["keras_class"].s == estimate.attr["keras_class"].s == b"Dense"
    assert list(estimate.input) == [hidden.name]


def test_historical_checkpoint_projector_and_cpu_timeline(run: Path) -> None:
    fs = TensorBoardFS(run)
    for name in (
        "projector_config.pbtxt",
        "labels.tsv",
        "bookmarks.json",
        "sprite.png",
    ):
        assert fs.read(f"/projector/{name}", 8192, 0) == (run / name).read_bytes()
    config = fs.read("/projector/projector_config.pbtxt", 8192, 0)
    assert b'tensor_name: "coordinates"' in config and b"tensor_path:" not in config
    assert b'bookmarks_path: "bookmarks.json"' in config
    assert b'image_path: "sprite.png"' in config
    (scene,) = json.loads(fs.read("/projector/bookmarks.json", 8192, 0))
    assert scene["label"] == "Left point"
    assert scene["dataSetDimensions"] == [2, 2] and scene["selectedPoints"] == [0]
    assert scene["pcaComponentDimensions"] == [0, 1]
    dimensions = fs.read("/projector/sprite.png", 24, 0)[16:24]
    assert struct.unpack(">II", dimensions) == (2, 2)
    (state, *_) = (run / "checkpoint").read_text(encoding="utf-8").splitlines()
    assert state == 'model_checkpoint_path: "coordinates"'
    assert b"coordinates" in (run / "coordinates.index").read_bytes()
    vectors = (run / "coordinates.data-00000-of-00001").read_bytes()
    assert struct.unpack("<4f", vectors) == (1, 2, 3, 4)
    trace = json.loads(fs.read("/profile/capture/cpu.trace.json", 16384, 0))[
        "traceEvents"
    ]
    assert any("CPU:0" in item.get("args", {}).get("name", "") for item in trace)
    assert any(
        item["ph"] == "X" and "watched_square" in item.get("args", {}).get("name", "")
        for item in trace
    )


def test_historical_native_watched_tensor_dump(run: Path) -> None:
    directory = run / "plugins/debugger/capture"
    (tensor_path,) = directory.glob("_tfdbg_device_*/*DebugIdentity_*")
    (graph_path,) = tensor_path.parent.glob("_tfdbg_graph_hash*")
    (metadata_path,) = directory.glob("_tfdbg_core_metadata_sessionrun*")
    (value,) = compat_pb2.Event.FromString(tensor_path.read_bytes()).summary.value
    assert value.node_name == "watched_square:0:DebugIdentity"
    assert value.metadata.plugin_data.plugin_name == "debugger"
    assert json.loads(value.metadata.plugin_data.content)["device"].endswith(
        "/device:CPU:0"
    )
    assert value.tensor.dtype == compat_pb2.DT_FLOAT
    assert struct.unpack("<2f", value.tensor.tensor_content) == (9, 4)
    graph = compat_pb2.GraphDef.FromString(
        compat_pb2.Event.FromString(graph_path.read_bytes()).graph_def
    )
    assert any(
        node.name == "watched_square" and node.op == "Square" for node in graph.node
    )
    metadata = json.loads(
        compat_pb2.Event.FromString(metadata_path.read_bytes()).log_message.message
    )
    assert metadata["input_names"] == ["samples:0"]
    assert metadata["output_names"] == ["watched_square:0"]


def test_tf200_eager_function_trace() -> None:
    fs = TensorBoardFS(ROOT / "tf200")
    path = "/graphs/squared_function/000007"
    native = compat_pb2.RunMetadata.FromString(
        fs.read(f"{path}.run_metadata.pb", 8192, 0)
    )
    assert any(
        node.op == "Square"
        for fn in native.function_graphs
        for node in fn.pre_optimization_graph.node
    )
    graph = compat_pb2.GraphDef.FromString(fs.read(f"{path}.pb", 8192, 0))
    assert any(node.op == "Square" for node in graph.node)


def test_tf200_native_beholder_sections_and_pixels() -> None:
    directory = ROOT / "tf200/plugins/beholder"
    section_info = {
        arg
        for op, arg, _ in genops((directory / "section-info.pkl").read_bytes())
        if op.name == "BINUNICODE"
    }
    assert {
        "(2, 3)",
        "(8,)",
        "-3.000e+00",
        "3.333e-01",
        "3.500e+00",
        "7.000e+00",
    } <= section_info
    (frame,) = compat_pb2.Summary.FromString(
        (directory / "frame.summary").read_bytes()
    ).value
    assert frame.tag == "beholder-frame"
    shape = tuple(dim.size for dim in frame.tensor.tensor_shape.dim)
    assert shape == (261, 768, 1)
    pixels = np.frombuffer(frame.tensor.tensor_content, np.uint8).reshape(shape)
    np.testing.assert_array_equal(pixels[0, ::256, 0], [0, 127, 255])
    assert not pixels[128:133].any()
    np.testing.assert_array_equal(
        pixels[133, ::96, 0], [0, 36, 72, 109, 145, 182, 218, 255]
    )
