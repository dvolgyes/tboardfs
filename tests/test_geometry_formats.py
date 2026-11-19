"""Read native TensorBoardX geometry, weighted PR and raw statistics offline."""

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
import pytest

from tboardfs import TensorBoardFS


RUN = Path(__file__).parent / "data" / "producer-cases" / "geometry"
FIELDS = (
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "precision",
    "recall",
)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (
            "weighted-pr",
            [
                [6, 6, 2, 2, 0],
                [8, 3, 3, 3, 0],
                [0, 5, 5, 5, 8],
                [0, 0, 4, 4, 6],
                [3 / 7, 2 / 3, 2 / 5, 2 / 5, 0],
                [1, 1, 1 / 3, 1 / 3, 0],
            ],
        ),
        (
            "raw-pr",
            [[4, 3, 2], [1, 2, 3], [5, 4, 3], [0, 1, 2], [0.8, 0.6, 0.4], [1, 0.5, 0]],
        ),
    ],
)
def test_writer_pr_curve(tag: str, expected: list[list[float]]) -> None:
    fs = TensorBoardFS(RUN)
    prefix = f"/pr_curves/{tag}/000004"
    tensor = np.load(BytesIO(fs.read(f"{prefix}.npy", 8192, 0)))
    np.testing.assert_array_equal(
        tensor, np.array(expected, dtype=np.float32), strict=True
    )
    rows = json.loads(fs.read(f"{prefix}.json", 8192, 0))
    thresholds = np.linspace(0, 1, len(expected[0]))
    np.testing.assert_array_equal([row["threshold"] for row in rows], thresholds)
    for field, values in zip(FIELDS, expected, strict=True):
        np.testing.assert_allclose([row[field] for row in rows], values)


def test_tensorboardx_mesh_has_float_components_and_integer_geometry() -> None:
    fs = TensorBoardFS(RUN)
    prefix = "/meshes/tetrahedron/000004"
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    with np.load(BytesIO(fs.read(f"{prefix}.npz", 8192, 0))) as arrays:
        for name, data in (("faces", faces), ("colors", colors)):
            np.testing.assert_array_equal(
                arrays[name], np.array([data], np.float32), strict=True
            )
    metadata = json.loads(fs.read(f"{prefix}.json", 8192, 0))
    assert json.loads(metadata["json_config"]) == {"material": {"wireframe": True}}
    obj = fs.read(f"{prefix}.obj", 8192, 0).decode().splitlines()
    assert obj[1:] == [
        "v 0.0 0.0 0.0 1 0 0",
        "v 1.0 0.0 0.0 0 1 0",
        "v 0.0 1.0 0.0 0 0 1",
        "v 0.0 0.0 1.0 1 1 1",
        "f 1 3 2",
        "f 1 2 4",
        "f 1 4 3",
        "f 2 3 4",
    ]
    ply = PlyData.read(BytesIO(fs.read(f"{prefix}.ply", 8192, 0)))
    np.testing.assert_array_equal(np.stack(ply["face"]["vertex_indices"]), faces)
    np.testing.assert_array_equal(
        np.column_stack([ply["vertex"][c] for c in ("red", "green", "blue")]), colors
    )
    assert fs.read("/meshes/cloud/000004.obj", 8192, 0).decode().splitlines()[1:] == [
        "v 2.0 3.0 5.0",
        "v -2.0 4.0 1.0",
    ]


def test_raw_histogram_and_interpolated_distribution() -> None:
    fs = TensorBoardFS(RUN, scalar_formats=("csv", "npz"))
    with np.load(BytesIO(fs.read("/histograms/interval/000004.npz", 8192, 0))) as data:
        np.testing.assert_array_equal(data["bucket_left"], [0.0])
        np.testing.assert_array_equal(data["bucket_right"], [2.0])
        np.testing.assert_array_equal(data["count"], [2.0])
    text = fs.read("/distributions/interval/000004.csv", 8192, 0).decode()
    rows = list(csv.DictReader(StringIO(text)))
    percentiles = [0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100]
    assert [float(row["percentile"]) for row in rows] == percentiles
    np.testing.assert_allclose(
        [float(row["value"]) for row in rows], np.array(percentiles) / 50
    )
