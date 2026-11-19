"""Read native PyTorch plugins offline, with no producer or multimedia imports."""

from hashlib import sha256
from io import BytesIO
import json
import wave

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    return TensorBoardFS(CORPUS / "torch-plugins")


@pytest.mark.parametrize(
    ("tag", "matrix"),
    [
        (
            "weighted",
            [
                [7, 7, 7, 2, 0],
                [3, 1, 0, 0, 0],
                [0, 2, 3, 3, 3],
                [0, 0, 0, 5, 7],
                [0.7, 0.875, 1, 1, 0],
                [1, 1, 1, 2 / 7, 0],
            ],
        ),
        (
            "precomputed",
            [
                [4, 3, 1],
                [3, 1, 0],
                [0, 2, 3],
                [0, 1, 3],
                [4 / 7, 0.75, 1],
                [1, 0.75, 0.25],
            ],
        ),
    ],
)
def test_pr_curve(fs: TensorBoardFS, tag: str, matrix: list) -> None:
    array = np.load(BytesIO(fs.read(f"/pr_curves/{tag}/000003.npy", 4096, 0)))
    np.testing.assert_array_equal(
        array, np.asarray(matrix, dtype=np.float32), strict=True
    )
    rows = json.loads(fs.read(f"/pr_curves/{tag}/000003.json", 8192, 0))
    np.testing.assert_allclose(
        [r["threshold"] for r in rows], np.linspace(0, 1, len(matrix[0]))
    )


@pytest.mark.parametrize("tag", ["observations", "interval"])
def test_histogram_and_distribution(fs: TensorBoardFS, tag: str) -> None:
    hist = json.loads(fs.read(f"/histograms/{tag}/000004.json", 8192, 0))
    assert [(r["bucket_left"], r["bucket_right"], r["count"]) for r in hist] == (
        [(0, 2, 2), (2, 4, 3)] if tag == "observations" else [(0, 2, 2)]
    )
    rows = json.loads(fs.read(f"/distributions/{tag}/000004.json", 8192, 0))
    pct = [0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100]
    assert [r["percentile"] for r in rows] == pct
    expected = (
        np.interp(pct, [0, 40, 100], [0, 2, 4])
        if tag == "observations"
        else np.array(pct) / 50
    )
    np.testing.assert_allclose([r["value"] for r in rows], expected)


@pytest.mark.parametrize("tag", ["triangle", "colored-cloud"])
def test_colored_geometry(fs: TensorBoardFS, tag: str) -> None:
    obj = fs.read(f"/meshes/{tag}/000005.obj", 4096, 0).decode().splitlines()
    assert obj[1:] == [
        "v 0.0 0.0 0.0 1 0 0",
        "v 1.0 0.0 0.0 0 1 0",
        "v 0.0 1.0 0.0 0 0 1",
    ] + (["f 1 2 3"] if tag == "triangle" else [])


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        (
            "/videos/motion/000006.gif",
            "a806e8fa7f45b9c4791e86ea2747b4fd46d58da595e79009f487755ac48d411b",
        ),
        (
            "/images/still/000006.png",
            "d36a015dfcc2f6cd45d0c3cbf247584818043dbdfcfeb3c9419f1ab41dcd87dc",
        ),
        (
            "/projector/00008/features/sprite.png",
            "e99fdf8cffb291aa5fd6ef7df9928183a8b3d1ba96f93a96c144ce2f177f1d6d",
        ),
    ],
)
def test_displayable_media(fs: TensorBoardFS, path: str, digest: str) -> None:
    """Export the exact RGB frames and sprite validated during generation."""
    assert sha256(fs.read(path, 4096, 0)).hexdigest() == digest


def test_native_audio_is_playable_tone(fs: TensorBoardFS) -> None:
    with wave.open(BytesIO(fs.read("/audio/tone/000006.wav", 8192, 0))) as wav:
        assert (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getnframes(),
        ) == (1, 2, 8000, 2000)
        pcm = np.frombuffer(wav.readframes(2000), dtype="<i2")
    expected = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2000) / 8000)
    np.testing.assert_allclose(pcm / 32767, expected, atol=1 / 32767)


def test_text_and_chart_configuration(fs: TensorBoardFS) -> None:
    assert (
        fs.read("/text/notes/text_summary/000007.txt", 4096, 0)
        == b"Torch API text\nsecond line"
    )
    (category,) = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))[
        "category"
    ]
    assert category == {
        "title": "comparison",
        "chart": [
            {
                "title": "bounds",
                "margin": {
                    "series": [{"value": "score", "lower": "low", "upper": "high"}]
                },
            },
            {"title": "loss", "multiline": {"tag": ["loss/train", "loss/eval"]}},
        ],
    }
    for tag, value in {
        "score": 0.5,
        "low": 0.25,
        "high": 0.75,
        "loss/train": 1,
        "loss/eval": 1.5,
    }.items():
        (row,) = json.loads(fs.read(f"/scalars/{tag}.json", 4096, 0))
        assert (row["step"], row["value"]) == (7, value)


def test_embedding_config_and_vectors(fs: TensorBoardFS) -> None:
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0).decode()
    prefix = "00008/features"
    assert 'tensor_name: "features:00008"' in config
    for field, name in [
        ("metadata_path", "metadata.tsv"),
        ("tensor_path", "tensors.tsv"),
        ("image_path", "sprite.png"),
    ]:
        assert f'{field}: "{prefix}/{name}"' in config
    assert fs.read(f"/projector/{prefix}/metadata.tsv", 4096, 0).splitlines() == [
        b"red",
        b"green",
        b"blue",
    ]
    matrix = np.loadtxt(
        BytesIO(fs.read(f"/projector/{prefix}/tensors.tsv", 4096, 0)), delimiter="\t"
    )
    np.testing.assert_array_equal(matrix, [[0, 1], [2, 3], [4, 5]])


def test_hparams_child_run(fs: TensorBoardFS) -> None:
    data = json.loads(fs.read("/trial/hparams/hparams.json", 8192, 0))
    assert data["session_start"]["hparams"] == {"lr": 0.125, "optimizer": "sgd"}
    assert data["session_end"]["status"] == "STATUS_SUCCESS"
    assert data["experiment"]["metric_infos"][0]["name"]["tag"] == "accuracy"
    (row,) = json.loads(fs.read("/trial/scalars/accuracy.json", 4096, 0))
    assert (row["step"], row["value"]) == (9, 0.875)


def test_cpu_trace_contains_nested_real_timed_operations(fs: TensorBoardFS) -> None:
    trace = json.loads(fs.read("/profile/capture/cpu.pt.trace.json", 16384, 0))
    assert trace["schemaVersion"] == 1
    events = trace["traceEvents"]
    (annotation,) = (e for e in events if e["cat"] == "user_annotation")
    assert annotation["name"] == "torch_plugin_fixture_operation"
    ops = {e["name"] for e in events if e["cat"] == "cpu_op"}
    assert {"aten::arange", "aten::square", "aten::sum", "aten::item"} <= ops
    for e in events:
        assert set(e) == {"name", "cat", "ph", "dur", "ts", "pid", "tid"}
        assert (e["ph"], e["pid"], e["tid"]) == ("X", 1, 1)
        assert e["dur"] > 0 and 0 <= e["ts"] <= e["ts"] + e["dur"] <= annotation["dur"]
