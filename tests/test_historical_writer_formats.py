"""Read records from actual TensorBoard 1.15.0 and 2.0.0 without either writer."""

from base64 import b64decode
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

from test_incremental_parse import _files
from test_proto3_reference import _fields


@pytest.fixture(scope="module", params=("history-115", "history-200"))
def run(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "data/producer-cases" / request.param


@pytest.fixture(scope="module")
def fs(run: Path) -> TensorBoardFS:
    return TensorBoardFS(run)


def test_scalar_and_histogram_percentiles(fs: TensorBoardFS) -> None:
    (scalar,) = json.loads(fs.read("/scalars/accuracy.json", 4096, 0))
    assert (scalar["step"], scalar["wall_time"], scalar["value"]) == (4, 5, 0.75)
    for tag in ("classic", "scores"):
        bins = json.loads(fs.read(f"/histograms/{tag}/000004.json", 4096, 0))
        assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in bins] == [
            (0, 2, 2),
            (2, 4, 3),
        ]
        rows = json.loads(fs.read(f"/distributions/{tag}/000004.json", 4096, 0))
        p = np.array([0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100])
        np.testing.assert_array_equal([r["percentile"] for r in rows], p)
        np.testing.assert_allclose(
            [r["value"] for r in rows], np.interp(p, [0, 40, 100], [0, 2, 4])
        )


def test_classifier_precision_and_recall(fs: TensorBoardFS) -> None:
    rows = json.loads(fs.read("/pr_curves/classifier/pr_curves/000004.json", 4096, 0))
    expected = {
        "threshold": [0, 0.25, 0.5, 0.75, 1],
        "true_positives": [2, 2, 1, 1, 0],
        "false_positives": [2, 1, 1, 0, 0],
        "true_negatives": [0, 1, 1, 2, 2],
        "false_negatives": [0, 0, 1, 1, 2],
        "precision": [0.5, 2 / 3, 0.5, 1, 0],
        "recall": [1, 1, 0.5, 0.5, 0],
    }
    for name, values in expected.items():
        np.testing.assert_allclose([r[name] for r in rows], values)


def test_triangle_and_point_cloud(fs: TensorBoardFS) -> None:
    data = PlyData.read(BytesIO(fs.read("/meshes/triangle/000004.ply", 4096, 0)))
    vertices = data["vertex"]
    np.testing.assert_array_equal(
        np.column_stack([vertices[k] for k in "xyz"]), [[0, 0, 0], [2, 0, 0], [0, 3, 0]]
    )
    np.testing.assert_array_equal(
        np.column_stack([vertices[k] for k in ("red", "green", "blue")]),
        np.diag([220, 180, 200]),
    )
    np.testing.assert_array_equal(np.stack(data["face"]["vertex_indices"]), [[0, 1, 2]])
    lines = fs.read("/meshes/cloud/000004.obj", 4096, 0).decode().splitlines()
    assert lines[1:] == ["v 0.0 0.0 1.0", "v 2.0 0.0 1.0", "v 0.0 3.0 1.0"]


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_numeric_tensor_dtype_and_values(fs: TensorBoardFS, dtype: str) -> None:
    actual = np.load(
        BytesIO(fs.read(f"/tensors/{dtype}/000004.npy", 4096, 0)), allow_pickle=False
    )
    expected = np.array([[0, 1], [3, 2]], dtype=dtype)
    if dtype.startswith("complex"):
        expected += 2j
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_text_and_untagged_strings(fs: TensorBoardFS) -> None:
    labels = json.loads(fs.read("/text/labels/000004.json", 4096, 0))
    assert labels["shape"] == [2, 2]
    assert labels["values"] == [["cat", "dog"], ["owl", "fox"]]
    assert json.loads(fs.read("/text/words/000004.json", 4096, 0))["values"] == [
        "birch",
        "fir",
    ]


def test_hparams_and_custom_scalar_charts(fs: TensorBoardFS) -> None:
    data = json.loads(fs.read("/hparams/hparams.json", 4096, 0))
    (hp,) = data["experiment"]["hparam_infos"]
    assert (hp["name"], hp["domain_discrete"]) == ("batch", [8, 16])
    assert data["experiment"]["metric_infos"][0]["name"]["tag"] == "accuracy"
    assert data["session_start"] == {
        "group_name": "baseline",
        "start_time_secs": 2,
        "hparams": {"batch": 16},
    }
    assert data["session_end"] == {"status": "STATUS_SUCCESS", "end_time_secs": 10}
    (category,) = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))[
        "category"
    ]
    assert category == {
        "title": "classification",
        "chart": [{"title": "validation", "multiline": {"tag": ["accuracy"]}}],
    }


def test_graph_meta_graph_and_execution_metadata(fs: TensorBoardFS) -> None:
    graph = compat_pb2.GraphDef.FromString(fs.read("/graphs/graph.pb", 4096, 0))
    assert [(n.name, n.op, list(n.input)) for n in graph.node] == [
        ("input", "Placeholder", []),
        ("energy", "Square", ["input"]),
    ]
    assert graph.node[1].attr["T"].type == 1
    meta = compat_pb2.MetaGraphDef.FromString(fs.read("/graphs/meta_graph.pb", 4096, 0))
    assert list(meta.meta_info_def.tags) == ["serve"]
    assert meta.graph_def == graph
    metadata = compat_pb2.RunMetadata.FromString(
        fs.read("/graphs/worker/000004.run_metadata.pb", 4096, 0)
    )
    (dev,) = metadata.step_stats.dev_stats
    (op,) = dev.node_stats
    assert (dev.device, op.node_name, op.all_start_micros) == (
        "/device:CPU:0",
        "energy",
        12,
    )


def test_status_and_obsolete_fields_survive_raw_mount(fs: TensorBoardFS) -> None:
    frames = b"".join(fs.read(p, 8192, 0) for p in sorted(_files(fs, "/raw")))
    records = [
        compat_pb2.Event.FromString(r.payload) for r in iter_tfrecords(BytesIO(frames))
    ]
    assert records[0].file_version == "brain.Event:2"
    start, checkpoint, stop = (
        r.session_log for r in records if r.HasField("session_log")
    )
    assert (start.status, start.msg) == (1, "training")
    assert (checkpoint.status, checkpoint.checkpoint_path) == (3, "model-4")
    assert (stop.status, stop.msg) == (2, "finished")
    (log,) = (r.log_message for r in records if r.HasField("log_message"))
    assert (log.level, log.message) == (30, "slow batch")
    values = {v.tag: v for record in records for v in record.summary.value}
    assert values["obsolete"].obsolete_old_style_histogram == b"\x00\xfflegacy"
    assert values["metadata-only"].node_name == "validation/input"


def test_numeric_debugger_and_bare_beholder_frame(fs: TensorBoardFS, run: Path) -> None:
    debug = json.loads(fs.read("/plugins/debugger/activations/000004.json", 4096, 0))
    assert json.loads(debug["plugin_content"]) == {"device": "CPU:0", "outputSlot": 0}
    tensor = debug["tensor"]
    assert (tensor["dtype"], tensor["shape"]) == (2, [15])
    pill = np.frombuffer(b64decode(tensor["tensor_content"]["base64"]), dtype="<f8")
    np.testing.assert_allclose(
        pill, [1, 4, 1, 0, 1, 1, 1, 0, -1, 3, 2 / 3, 26 / 9, -1, 1, 4]
    )
    (value,) = compat_pb2.Summary.FromString(
        (run / "plugins/beholder/frame.summary").read_bytes()
    ).value
    assert value.tag == "beholder-frame"
    assert [dim.size for dim in value.tensor.tensor_shape.dim] == [16, 16, 3]
    rgb = np.frombuffer(value.tensor.tensor_content, np.uint8).reshape(16, 16, 3)
    np.testing.assert_array_equal(rgb[8, :, 0], np.arange(16) * 17)
    np.testing.assert_array_equal(rgb[:, 8, 1], np.arange(16) * 17)
    np.testing.assert_array_equal(
        rgb[0, :, 2], [0] * 4 + [255] * 4 + [0] * 4 + [255] * 4
    )
    assert fs.readdir("/plugins")[2:] == ["debugger"]
    assert len(fs.readdir("/raw")[2:]) == 1


def test_projector_assets_and_legacy_profile(fs: TensorBoardFS) -> None:
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0).decode()
    for declaration in (
        'tensor_name: "birds"',
        'tensor_path: "vectors.tsv"',
        'metadata_path: "birds.tsv"',
    ):
        assert declaration in config
    assert fs.read("/projector/birds.tsv", 4096, 0).splitlines() == [b"owl", b"finch"]
    vectors = np.loadtxt(
        BytesIO(fs.read("/projector/vectors.tsv", 4096, 0)), delimiter="\t"
    )
    np.testing.assert_array_equal(vectors, np.eye(2))
    (op,) = _fields(fs.read("/profile/trial/cpu.trace", 4096, 0), 4)
    assert [_fields(op, field) for field in (1, 2, 3, 9, 10)] == [
        [1],
        [2],
        [b"Square"],
        [8_000_000],
        [2_500_000],
    ]
