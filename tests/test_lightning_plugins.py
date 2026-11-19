"""Read Lightning profiler output and its logger's delegated PyTorch summaries offline."""

from hashlib import sha256
from io import BytesIO
import json

import numpy as np
from plyfile import PlyData
import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(CORPUS / "lightning-plugins")


def test_weighted_classifier(fs: TensorBoardFS) -> None:
    rows = json.loads(fs.read("/pr_curves/classifier/000004.json", 4096, 0))
    assert [r["threshold"] for r in rows] == [0, 0.5, 1]
    for field, expected in {
        "true_positives": [7, 3, 3],
        "false_positives": [3, 2, 0],
        "true_negatives": [0, 1, 3],
        "false_negatives": [0, 4, 4],
        "precision": [0.7, 0.6, 1],
        "recall": [1, 3 / 7, 3 / 7],
    }.items():
        np.testing.assert_allclose([r[field] for r in rows], expected)


@pytest.mark.parametrize("tag", ["surface", "scan"])
def test_geometry(fs: TensorBoardFS, tag: str) -> None:
    data = PlyData.read(BytesIO(fs.read(f"/meshes/{tag}/000004.ply", 8192, 0)))
    xyz = np.column_stack([data["vertex"][c] for c in "xyz"])
    np.testing.assert_array_equal(
        xyz, [[-1, 0, 0.5], [1, 0, 0.5], [0, 2, 1], [0, 1, -1]]
    )
    if tag == "surface":
        np.testing.assert_array_equal(
            np.stack(data["face"]["vertex_indices"]), [[0, 1, 2], [1, 0, 3]]
        )
    else:
        assert "face" not in [element.name for element in data.elements]
        rgb = np.column_stack([data["vertex"][c] for c in ("red", "green", "blue")])
        np.testing.assert_array_equal(
            rgb, [[0, 64, 255], [255, 32, 0], [0, 255, 16], [128, 0, 32]]
        )


def test_double_precision_generic_tensor(fs: TensorBoardFS) -> None:
    data = np.load(BytesIO(fs.read("/tensors/calibration/000004.npy", 4096, 0)))
    np.testing.assert_array_equal(
        data, np.array([[1 + 2**-45, -3.125]], dtype=np.float64), strict=True
    )


def test_margin_uses_logged_series(fs: TensorBoardFS) -> None:
    (category,) = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))[
        "category"
    ]
    assert category["title"] == "estimates"
    (chart,) = category["chart"]
    assert chart["title"] == "interval"
    (series,) = chart["margin"]["series"]
    assert series == {"value": "estimate", "lower": "low", "upper": "high"}
    for role, value in {"value": 2.25, "lower": 1.5, "upper": 3}.items():
        (row,) = json.loads(fs.read(f"/scalars/{series[role]}.json", 4096, 0))
        assert (row["step"], row["value"]) == (4, value)


def test_embedding_rows_and_sprite(fs: TensorBoardFS) -> None:
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0).decode()
    for role, name in [
        ("tensor_path", "tensors.tsv"),
        ("metadata_path", "metadata.tsv"),
        ("image_path", "sprite.png"),
    ]:
        assert f'{role}: "00004/tracks/{name}"' in config
    prefix = "/projector/00004/tracks"
    assert fs.read(f"{prefix}/metadata.tsv", 4096, 0).splitlines() == [
        b"left",
        b"center",
        b"right",
    ]
    xyz = np.loadtxt(BytesIO(fs.read(f"{prefix}/tensors.tsv", 4096, 0)), delimiter="\t")
    np.testing.assert_array_equal(xyz, [[-1, 3], [0.25, 1], [2, -0.5]])


def test_profile_is_from_lightning(fs: TensorBoardFS) -> None:
    data = json.loads(fs.read("/profile/capture/lightning.pt.trace.json", 8192, 0))
    events = {e["name"]: e for e in data["traceEvents"]}
    annotation = events.pop("[pl][profile]calibration_training_step")
    assert annotation["cat"] == "user_annotation" and annotation["ts"] == 0
    assert set(events) == {"aten::square", "aten::sum", "aten::item"}
    for op in events.values():
        assert set(op) == {"name", "cat", "dur", "ph", "pid", "tid", "ts"}
        assert (op["cat"], op["pid"], op["tid"], op["ph"]) == ("cpu_op", 1, 1, "X")
        assert 0 <= op["ts"] < op["ts"] + op["dur"] <= annotation["dur"]


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        (
            "/videos/tracking/000004.gif",
            "e25724357429e39796fb5c52779a8906196be7fbc56c45c4215c5b708be68cea",
        ),
        (
            "/projector/00004/tracks/sprite.png",
            "e4fa5921008ee4d184a55b8ae7a06c37a86058e192cde9bf472ba295ee7daf99",
        ),
    ],
)
def test_tracking_images(fs: TensorBoardFS, path: str, digest: str) -> None:
    """Preserve every sprite tile and ordered GIF frame decoded during generation."""
    assert sha256(fs.read(path, 4096, 0)).hexdigest() == digest
