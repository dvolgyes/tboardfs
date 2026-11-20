"""Read pinned Torch, Lightning, and TensorBoardX outputs without the writers."""

import json
from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
from plyfile import PlyData
from test_tf_producer_cases import CORPUS

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords


@pytest.fixture(params=["torch-matrix", "lightning-matrix"])
def delegated(request: pytest.FixtureRequest) -> TensorBoardFS:
    return TensorBoardFS(CORPUS / request.param)


def _raw_values(producer: str) -> dict[str, compat_pb2.Summary.Value]:
    root = CORPUS / producer
    fs = TensorBoardFS(root)
    (event,) = root.glob("events.out.tfevents.*")
    raw = f"/raw/{event.name}"
    payload = b"".join(
        fs.read(f"{raw}/{name}", 4096, 0) for name in sorted(fs.readdir(raw)[2:])
    )
    assert payload == event.read_bytes()
    return {
        value.tag: value
        for frame in iter_tfrecords(BytesIO(payload))
        for value in compat_pb2.Event.FromString(frame.payload).summary.value
    }


def test_grouped_subruns_keep_shared_tag(delegated: TensorBoardFS) -> None:
    assert "score" not in delegated.readdir("/scalars")
    for child, expected in [("left", 1.25), ("right", 2.5)]:
        (row,) = json.loads(
            delegated.read(f"/score_split_{child}/scalars/score/split.json", 4096, 0)
        )
        assert (row["step"], row["value"]) == (4, expected)


@pytest.mark.parametrize(
    "producer", ["torch-matrix", "lightning-matrix", "tensorboardx-matrix"]
)
def test_pr_without_weights(producer: str) -> None:
    fs = TensorBoardFS(CORPUS / producer)
    path = "/pr_curves/unweighted/000004"
    data = np.load(BytesIO(fs.read(f"{path}.npy", 4096, 0)))
    rows = json.loads(fs.read(f"{path}.json", 4096, 0))
    assert [row["threshold"] for row in rows] == [0, 0.5, 1]
    assert [
        (
            row["true_positives"],
            row["false_positives"],
            row["true_negatives"],
            row["false_negatives"],
        )
        for row in rows
    ] == [(2, 2, 0, 0), (2, 1, 1, 0), (1, 0, 2, 1)]
    np.testing.assert_array_equal(
        data,
        np.array(
            [
                [2, 2, 1],
                [2, 1, 0],
                [0, 1, 2],
                [0, 0, 1],
                [0.5, 2 / 3, 1],
                [1, 1, 0.5],
            ],
            dtype=np.float32,
        ),
        strict=True,
    )


@pytest.mark.parametrize(
    ("producer", "tag", "faces", "colors"),
    [
        ("torch-matrix", "surface", True, False),
        ("torch-matrix", "plain-cloud", False, False),
        ("lightning-matrix", "surface", True, True),
        ("lightning-matrix", "plain-cloud", False, False),
        ("tensorboardx-matrix", "surface", True, False),
        ("tensorboardx-matrix", "colored-cloud", False, True),
    ],
)
def test_batched_geometry_optional_components(
    producer: str, tag: str, faces: bool, colors: bool
) -> None:
    fs = TensorBoardFS(CORPUS / producer)
    ply = PlyData.read(BytesIO(fs.read(f"/meshes/{tag}/000004.ply", 4096, 0)))
    assert [element.name for element in ply.elements] == (
        ["vertex", "face"] if faces else ["vertex"]
    )
    vertex = ply["vertex"].data
    assert vertex.dtype.names == (
        ("x", "y", "z", "red", "green", "blue") if colors else ("x", "y", "z")
    )
    np.testing.assert_array_equal(
        np.column_stack([vertex[axis] for axis in "xyz"]),
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [5, 5, 5],
            [7, 5, 5],
            [5, 6, 5],
        ],
    )
    if faces:
        assert [face.tolist() for face in ply["face"].data["vertex_indices"]] == [
            [0, 1, 2],
            [5, 4, 3],
        ]
    if colors:
        np.testing.assert_array_equal(
            np.column_stack([vertex[c] for c in ("red", "green", "blue")]),
            [
                [255, 0, 0],
                [0, 64, 0],
                [0, 0, 128],
                [0, 0, 128],
                [0, 64, 0],
                [255, 0, 0],
            ],
        )


def test_projector_independently_optional_assets(delegated: TensorBoardFS) -> None:
    config = delegated.read("/projector/projector_config.pbtxt", 4096, 0).decode()
    entries = config.split("embeddings {")[1:]
    assert len(entries) == 3
    for i, (tag, entry) in enumerate(
        zip(["vectors", "labels", "sprite"], entries, strict=True)
    ):
        parent = f"00004/{tag}"
        assert f'tensor_name: "{tag}:00004"' in entry
        assert f'tensor_path: "{parent}/tensors.tsv"' in entry
        assert ("metadata_path" in entry) == (tag == "labels")
        assert ("image_path" in entry) == (tag == "sprite")
        matrix = np.loadtxt(
            BytesIO(delegated.read(f"/projector/{parent}/tensors.tsv", 4096, 0))
        )
        np.testing.assert_array_equal(matrix, np.array([[1, -2], [4.5, 8]]) + i)
    assert (
        delegated.read("/projector/00004/labels/metadata.tsv", 4096, 0) == b"oak\nash\n"
    )
    sprite = delegated.read("/projector/00004/sprite/sprite.png", 4096, 0)
    assert (
        sha256(sprite).hexdigest()
        == "44b9f96054bbcae63c3727eb89508bf3a9c146c2a8e122d7416c3c42c51691bd"
    )


def test_explicit_discrete_child_run(delegated: TensorBoardFS) -> None:
    data = json.loads(delegated.read("/trial/hparams/hparams.json", 4096, 0))
    (optimizer,) = data["experiment"]["hparam_infos"]
    assert (optimizer["name"], optimizer["domain_discrete"]) == (
        "optimizer",
        ["adam", "sgd"],
    )
    assert data["session_start"]["hparams"] == {"optimizer": "sgd"}
    assert data["session_end"]["status"] == "STATUS_SUCCESS"
    (row,) = json.loads(delegated.read("/trial/scalars/accuracy.json", 4096, 0))
    assert (row["step"], row["value"]) == (4, 0.875)


@pytest.mark.parametrize(
    ("tag", "digest"),
    [
        ("grid", "8b66f634186b74a76958a021383caed6bc6bb282a37b8f7d4e6eab809d326128"),
        ("boxed", "cc6ed14557f1fdc2955738dd6a4affb19d0a042a916d46b2a01a705fc885cd10"),
    ],
)
def test_image_grid_and_annotated_raster(
    delegated: TensorBoardFS, tag: str, digest: str
) -> None:
    assert (
        sha256(delegated.read(f"/images/{tag}/000004.png", 4096, 0)).hexdigest()
        == digest
    )


def test_lightning_raw_buckets_and_precomputed_pr() -> None:
    fs = TensorBoardFS(CORPUS / "lightning-matrix")
    rows = json.loads(fs.read("/histograms/buckets/000004.json", 4096, 0))
    assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in rows] == [
        (0.5, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
    ]
    values = _raw_values("lightning-matrix")
    hist = values["buckets"].histo
    assert (hist.min, hist.max, hist.num, hist.sum, hist.sum_squares) == (
        0.5,
        3,
        3,
        5.5,
        13.25,
    )
    assert list(hist.bucket_limit) == [1, 2, 4]
    assert list(hist.bucket) == [1, 1, 1]
    data = np.load(BytesIO(fs.read("/pr_curves/precomputed/000004.npy", 4096, 0)))
    np.testing.assert_array_equal(
        data,
        np.array(
            [
                [4, 3, 1],
                [4, 1, 0],
                [0, 3, 4],
                [0, 1, 3],
                [0.5, 0.75, 1],
                [1, 0.75, 0.25],
            ],
            dtype=np.float32,
        ),
        strict=True,
    )


def test_lightning_multiline_resolves_logged_metrics() -> None:
    fs = TensorBoardFS(CORPUS / "lightning-matrix")
    assert json.loads(fs.read("/custom_scalars/layout.json", 4096, 0)) == {
        "category": [
            {
                "title": "accuracy",
                "chart": [
                    {"title": "comparison", "multiline": {"tag": ["train", "valid"]}},
                ],
            }
        ],
    }
    for tag, value in [("train", 0.75), ("valid", 0.625)]:
        (row,) = json.loads(fs.read(f"/scalars/{tag}.json", 4096, 0))
        assert (row["step"], row["value"]) == (4, value)


def test_lightning_scalar_precision_and_tensor_encodings() -> None:
    fs = TensorBoardFS(CORPUS / "lightning-matrix")
    values = _raw_values("lightning-matrix")
    scalar = values["precise"].tensor
    assert (scalar.dtype, list(scalar.double_val)) == (2, [16_777_219])
    (row,) = json.loads(fs.read("/scalars/precise.json", 4096, 0))
    assert (row["step"], row["value"]) == (4, 16_777_219)
    complex_tensor = values["complex64"].tensor
    assert (complex_tensor.dtype, list(complex_tensor.scomplex_val)) == (
        8,
        [1, 2, -3.5, -0.5],
    )
    array = np.load(BytesIO(fs.read("/tensors/complex64/000004.npy", 4096, 0)))
    np.testing.assert_array_equal(
        array, np.array([1 + 2j, -3.5 - 0.5j], dtype=np.complex64), strict=True
    )
    half = values["float16"].tensor
    assert (half.dtype, list(half.half_val)) == (19, [1069547520, -1073741824])
    assert fs.readdir("/tensors/float16")[2:] == ["000004.json"]
    fallback = json.loads(fs.read("/tensors/float16/000004.json", 4096, 0))["tensor"]
    assert fallback == {
        "dtype": 19,
        "shape": [2],
        "values": {"13": [1069547520, -1073741824]},
    }


def test_tensorboardx_file_writer_preserves_binary_strings() -> None:
    tensor = _raw_values("tensorboardx-matrix")["binary-strings"].tensor
    assert tensor.dtype == 7
    assert [dim.size for dim in tensor.tensor_shape.dim] == [3]
    assert list(tensor.string_val) == [b"", b"A\x00B", b"\xfe\xff"]
    fs = TensorBoardFS(CORPUS / "tensorboardx-matrix")
    text = json.loads(fs.read("/text/binary-strings/000004.json", 4096, 0))
    assert (text["shape"], text["values"]) == ([3], ["", "A\0B", "\ufffd\ufffd"])
