"""Verify materialized files from native TensorFlow 2.21 writer APIs."""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData
from test_proto3_reference import _fields

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

ROOT = Path(__file__).parent / "data" / "producer-cases" / "tf221-extra"


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        (
            "weighted",
            [
                [7, 10, 0, 0, 7 / 17, 1],
                [2, 3, 7, 5, 0.4, 2 / 7],
                [2, 0, 10, 5, 1, 2 / 7],
            ],
        ),
        (
            "prepared",
            [[4, 4, 0, 0, 0.5, 1], [3, 1, 3, 1, 0.75, 0.75], [1, 0, 4, 3, 1, 0.25]],
        ),
    ],
)
def test_graph_precision_recall(name: str, rows: list[list[float]]) -> None:
    fs = TensorBoardFS(ROOT / "graph")
    path = f"/pr_curves/{name}/pr_curves/000007"
    table = json.loads(fs.read(f"{path}.json", 4096, 0))
    fields = (
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
    )
    for threshold, raw, row in zip([0, 0.5, 1], rows, table, strict=True):
        assert row == pytest.approx(
            {"threshold": threshold, **dict(zip(fields, raw, strict=True))}
        )
    array = np.load(BytesIO(fs.read(f"{path}.npy", 4096, 0)))
    np.testing.assert_array_equal(
        array, np.array(rows, dtype=np.float32).T, strict=True
    )


@pytest.mark.parametrize("surface", [True, False])
@pytest.mark.parametrize("colored", [True, False])
def test_graph_batched_geometry(surface: bool, colored: bool) -> None:
    fs = TensorBoardFS(ROOT / "graph")
    name = ("surface" if surface else "cloud") + ("-rgb" if colored else "")
    prefix = f"/meshes/{name}/000007"
    vertices = np.array(
        [
            [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 2, 1], [1, 0, 1]],
        ]
    )
    colors = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
        ]
    )
    expected = {"vertices": vertices}
    if colored:
        expected["colors"] = colors
    if surface:
        expected["faces"] = np.array([[[0, 1, 2]], [[0, 2, 1]]])
    with np.load(BytesIO(fs.read(f"{prefix}.npz", 4096, 0))) as arrays:
        assert set(arrays.files) == expected.keys()
        for component, values in expected.items():
            np.testing.assert_array_equal(arrays[component], values)

    ply = PlyData.read(BytesIO(fs.read(f"{prefix}.ply", 4096, 0)))
    np.testing.assert_array_equal(
        np.column_stack([ply["vertex"][axis] for axis in "xyz"]),
        vertices.reshape(-1, 3),
    )
    assert ("face" in ply) == surface
    if surface:
        np.testing.assert_array_equal(
            np.stack(ply["face"]["vertex_indices"]), [[0, 1, 2], [3, 5, 4]]
        )
    rgb = ("red", "green", "blue")
    assert ply["vertex"].data.dtype.names == tuple("xyz") + (rgb if colored else ())
    if colored:
        np.testing.assert_array_equal(
            np.column_stack([ply["vertex"][c] for c in rgb]), colors.reshape(-1, 3)
        )


@pytest.mark.parametrize(
    ("run", "tag", "rank"),
    [("graph", "directions", 2), ("graph", "planes", 3), ("eager", "planes", 3)],
)
def test_native_text_tables(run: str, tag: str, rank: int) -> None:
    fs = TensorBoardFS(ROOT / run)
    prefix = f"/text/{tag}/000007"
    table = [["north", "east"], ["south", "west"]]
    values = table if rank == 2 else [table, [["silver", "gold"], ["green", "blue"]]]
    record = json.loads(fs.read(f"{prefix}.json", 4096, 0))
    assert record["shape"] == [2] * rank
    assert record["values"] == values
    text = fs.read(f"{prefix}.txt", 4096, 0).decode()
    markdown = "|   |   |\n| --- | --- |\n| north | east |\n| south | west |\n"
    if rank == 2:
        assert text == markdown
    else:
        warning, rendered = text.split("\n\n", 1)
        assert warning.startswith("**Warning:** this text summary has 3 dimensions")
        assert rendered == markdown


@pytest.mark.parametrize("run", ["graph", "eager"])
@pytest.mark.parametrize(
    ("tag", "dtype", "values"),
    [
        ("wide", "float64", [1 + 2**-48, -2.5]),
        ("complex", "complex64", [2 - 3j, -1 + 0.5j]),
    ],
)
def test_native_wide_and_complex_tensors(
    run: str, tag: str, dtype: str, values: list[float] | list[complex]
) -> None:
    fs = TensorBoardFS(ROOT / run)
    data = np.load(BytesIO(fs.read(f"/tensors/{tag}/000007.npy", 4096, 0)))
    np.testing.assert_array_equal(data, np.array(values, dtype=dtype), strict=True)


def test_graph_binary_strings_preserve_exact_raw_bytes() -> None:
    fs = TensorBoardFS(ROOT / "graph")
    data = json.loads(fs.read("/text/binary/000007.json", 4096, 0))
    assert (data["shape"], data["values"]) == ([2], ["a\0b", "\ufffd\ufffd"])
    directory = "/raw/events.out.tfevents.graph-extra"
    tensors = []
    for name in fs.readdir(directory):
        if name.endswith(".tfrecord"):
            frame = fs.read(f"{directory}/{name}", 4096, 0)
            event = compat_pb2.Event.FromString(frame[12:-4])
            tensors.extend(
                value.tensor for value in event.summary.value if value.tag == "binary"
            )
    (tensor,) = tensors
    assert list(tensor.string_val) == [b"a\0b", b"\xff\x80"]


def test_graph_hparams_mixed_domains_and_session_success() -> None:
    fs = TensorBoardFS(ROOT / "graph")
    data = json.loads(fs.read("/hparams/hparams.json", 4096, 0))
    domains = {info["name"]: info for info in data["experiment"]["hparam_infos"]}
    assert domains["mode"]["domain_discrete"] == ["accurate", "fast"]
    assert domains["enabled"]["domain_discrete"] == [False, True]
    assert domains["rate"]["domain_interval"] == {"min_value": 0.125, "max_value": 0.5}
    assert data["session_start"] == {
        "group_name": "bounds",
        "start_time_secs": 31,
        "hparams": {"mode": "fast", "enabled": True, "rate": 0.25},
    }
    assert data["session_end"] == {"status": "STATUS_SUCCESS", "end_time_secs": 32}
    assert data["metrics"] == {"score": 2.0}


def test_graph_margin_links_logged_series() -> None:
    fs = TensorBoardFS(ROOT / "graph")
    layout = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))
    (category,) = layout["category"]
    assert category["title"] == "measurements"
    (chart,) = category["chart"]
    assert chart["title"] == "bounds"
    (series,) = chart["margin"]["series"]
    assert series == {"value": "score", "lower": "minimum", "upper": "maximum"}
    for key, expected in {"value": 2.0, "lower": 1.0, "upper": 4.0}.items():
        (row,) = json.loads(fs.read(f"/scalars/{series[key]}.json", 4096, 0))
        assert (row["step"], row["value"]) == (7, expected)


def test_graph_checkpoint_projector_assets() -> None:
    run = ROOT / "graph"
    fs = TensorBoardFS(run)
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0)
    assert config == (run / "projector_config.pbtxt").read_bytes()
    for line in (
        b'tensor_name: "vectors"',
        b'metadata_path: "labels.tsv"',
        b'bookmarks_path: "bookmarks.json"',
        b'image_path: "sprite.png"',
    ):
        assert line in config
    assert config.count(b"single_image_dim: 1") == 2
    assert fs.read("/projector/labels.tsv", 4096, 0) == b"red\ngreen\n"
    sprite = fs.read("/projector/sprite.png", 4096, 0)
    assert sprite == (run / "sprite.png").read_bytes()
    assert sprite[16:24] == b"\0\0\0\x02\0\0\0\x01"
    (view,) = json.loads(fs.read("/projector/bookmarks.json", 4096, 0))
    assert view["label"] == "First point"
    assert view["selectedPoints"] == [0]
    assert view["dataSetDimensions"] == [2, 2]
    assert view["projections"] == [
        {"pca-0": -3.4, "pca-1": 0.0},
        {"pca-0": 3.4, "pca-1": 0.0},
    ]
    assert 'model_checkpoint_path: "positions"' in (run / "checkpoint").read_text()
    tensor = np.fromfile(run / "positions.data-00000-of-00001", dtype="<f4")
    np.testing.assert_array_equal(tensor.reshape(2, 2), [[1.0, -2.0], [3.0, 4.5]])
    assert set(fs.readdir("/projector")[2:]) == {
        "bookmarks.json",
        "labels.tsv",
        "projector_config.pbtxt",
        "sprite.png",
    }


def test_live_eager_debugger_records_cpu_square_without_training_events() -> None:
    run = ROOT / "eager"
    dump = run / "tfdbg"
    with (dump / "tfdbg_events.capture.execution").open("rb") as handle:
        (execution,) = iter_tfrecords(handle)
    (op,) = _fields(execution.payload, 9)
    assert _fields(op, 1) == [b"Square"]
    (payload,) = _fields(op, 7)
    tensor = compat_pb2.TensorProto.FromString(payload)
    assert tensor.dtype == 1
    assert [dim.size for dim in tensor.tensor_shape.dim] == [2]
    np.testing.assert_array_equal(
        np.frombuffer(tensor.tensor_content, dtype="<f4"), [9, 4]
    )
    with (dump / "tfdbg_events.capture.graphs").open("rb") as handle:
        (registration,) = iter_tfrecords(handle)
    (device,) = _fields(registration.payload, 12)
    assert _fields(device, 1) == [b"/job:localhost/replica:0/task:0/device:CPU:0"]
    assert _fields(op, 9) == [bytes(_fields(device, 2))]

    fs = TensorBoardFS(run)
    assert fs.readdir("/raw")[2:] == ["events.out.tfevents.eager-extra"]
    assert fs.readdir("/tensors")[2:] == ["complex", "wide"]
