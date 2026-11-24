"""Read TF1 graph-mode writer outputs without TensorFlow or TensorBoard."""

import json
import struct
import wave
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).parent / "data" / "producer-cases" / "tf1-writer"


@pytest.fixture
def fs() -> TensorBoardFS:
    return TensorBoardFS(ROOT)


def test_graph_mode_scalar_histogram_and_distribution(fs: TensorBoardFS) -> None:
    (scalar,) = json.loads(fs.read("/scalars/loss.json", 4096, 0))
    assert (scalar["step"], scalar["value"]) == (3, 0.25)
    bins = json.loads(fs.read("/histograms/weights/000003.json", 8192, 0))
    occupied = [b for b in bins if b["count"]]
    assert [b["count"] for b in occupied] == [1] * 4
    for bucket, value in zip(occupied, (-2, 0, 2, 4), strict=True):
        assert bucket["bucket_left"] <= value <= bucket["bucket_right"]
    rows = json.loads(fs.read("/distributions/weights/000003.json", 8192, 0))
    assert [(r["percentile"], r["value"]) for r in (rows[0], rows[-1])] == [
        (0, -2),
        (100, 4),
    ]
    assert [r["value"] for r in rows] == sorted(r["value"] for r in rows)


def test_native_graph_mode_media(fs: TensorBoardFS) -> None:
    """The producer decoded the complete gradient before recording its PNG."""
    png = fs.read("/images/gradient/image/000003.png", 65536, 0)
    assert (
        sha256(png).hexdigest()
        == "ddcb9e4f24277696993058382f45e131079c2cd6fae0a844bbea423e877f1eb8"
    )
    with wave.open(BytesIO(fs.read("/audio/tone/audio/000003.wav", 8192, 0))) as wav:
        assert (
            wav.getframerate(),
            wav.getsampwidth(),
            wav.getnchannels(),
            wav.getnframes(),
        ) == (8000, 2, 1, 1600)
        pcm = np.frombuffer(wav.readframes(1600), dtype="<i2")
    expected = 0.6 * 32768 * np.sin(np.arange(1600) * (2 * np.pi * 440 / 8000))
    np.testing.assert_allclose(pcm, expected, atol=2)


def test_tensor_summary_and_text_op(fs: TensorBoardFS) -> None:
    nums = np.load(
        BytesIO(fs.read("/tensors/numeric/000003.npy", 4096, 0)), allow_pickle=False
    )
    assert nums.dtype == np.float32
    np.testing.assert_array_equal(nums, [[1, 2], [3, 4]])
    assert json.loads(fs.read("/text/words/000003.json", 4096, 0))["values"] == [
        "red",
        "blue",
    ]
    assert fs.read("/text/note/000003.txt", 4096, 0).splitlines()[2:] == [
        b"| graph-mode text |",
        b"| second row |",
    ]


def test_thresholds_and_geometry(fs: TensorBoardFS) -> None:
    rows = json.loads(fs.read("/pr_curves/binary/pr_curves/000003.json", 8192, 0))
    assert [
        (
            r["threshold"],
            r["true_positives"],
            r["false_positives"],
            r["true_negatives"],
            r["false_negatives"],
        )
        for r in rows
    ] == [
        (0, 2, 2, 0, 0),
        (0.25, 2, 1, 1, 0),
        (0.5, 2, 1, 1, 0),
        (0.75, 1, 0, 2, 1),
        (1, 0, 0, 2, 2),
    ]
    triangle = fs.read("/meshes/triangle/000003.obj", 4096, 0).splitlines()
    assert triangle[1:] == [
        b"v 0.0 0.0 0.0",
        b"v 1.0 0.0 0.0",
        b"v 0.0 1.0 0.0",
        b"f 1 2 3",
    ]
    header, vertices = fs.read("/meshes/cloud/000003.ply", 4096, 0).split(
        b"end_header\n"
    )
    assert b"element face" not in header
    assert list(struct.iter_unpack("<3f3B", vertices)) == [
        (0, 0, 0, 255, 0, 0),
        (1, 0, 0, 0, 255, 0),
        (0, 1, 0, 0, 0, 255),
    ]


def test_layout_hparams_and_embedding(fs: TensorBoardFS) -> None:
    layout = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))
    assert layout["category"] == [
        {
            "title": "training",
            "chart": [{"title": "objective", "multiline": {"tag": ["loss"]}}],
        }
    ]
    hp = json.loads(fs.read("/hparams/hparams.json", 8192, 0))
    assert hp["experiment"]["hparam_infos"][0]["domain_discrete"] == [0.25, 0.5]
    assert hp["session_start"]["hparams"] == {"rate": 0.25}
    assert hp["metrics"] == {"loss": 0.25}
    assert fs.read("/projector/labels.tsv", 4096, 0).splitlines() == [
        b"origin",
        b"right",
        b"up",
    ]
    np.testing.assert_array_equal(
        np.loadtxt(BytesIO(fs.read("/projector/vectors.tsv", 4096, 0))),
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    )


def test_run_metadata_and_separate_v1_tools(fs: TensorBoardFS) -> None:
    metadata = compat_pb2.RunMetadata.FromString(
        fs.read("/graphs/energy/000003.run_metadata.pb", 65536, 0)
    )
    assert {"squared", "energy"} <= {
        n.node_name for d in metadata.step_stats.dev_stats for n in d.node_stats
    }
    durations = {
        e["name"]: e["dur"]
        for e in json.loads((ROOT / "timeline_-1").read_bytes())["traceEvents"]
        if e["ph"] == "X"
    }
    assert durations["squared"] >= 0 and durations["energy"] >= 0
    (value,) = compat_pb2.Event.FromString(
        (ROOT / "tfdbg/squared.pb").read_bytes()
    ).summary.value
    assert value.node_name == "squared:0:DebugIdentity"
    np.testing.assert_array_equal(
        np.frombuffer(value.tensor.tensor_content, dtype="<f4"), [1, 4, 9]
    )
    root = fs.readdir("/")
    assert "tfdbg" not in root
    assert "profile" not in root
