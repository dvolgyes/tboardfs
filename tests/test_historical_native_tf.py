"""Read real TF 1.15 session and TF 2.0 eager summaries without either framework."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


ROOT = Path(__file__).parent / "data" / "producer-cases"


@pytest.mark.parametrize(
    ("producer", "step"), [("tf115-native", 12), ("tf200-native", 14)]
)
def test_signal_analysis(producer: str, step: int) -> None:
    """Check equal-width TF2 interpolation and TensorBoard 1.15 compressor reference values."""
    fs = TensorBoardFS(ROOT / producer)
    (scalar,) = json.loads(fs.read("/scalars/mean_square.json", 4096, 0))
    assert (scalar["step"], scalar["value"]) == (step, 5)
    values = np.load(BytesIO(fs.read(f"/tensors/energy/{step:06d}.npy", 4096, 0)))
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, [9, 1, 1, 9])
    text = json.loads(fs.read(f"/text/channels/{step:06d}.json", 4096, 0))
    assert text["values"] == ["left", "right"]
    bins = json.loads(fs.read(f"/histograms/weights/{step:06d}.json", 65536, 0))
    occupied = [b for b in bins if b["count"]]
    assert sum(b["count"] for b in occupied) == 4
    assert [b["count"] for b in occupied] == ([1] * 4 if step == 12 else [2, 2])
    for value in [-3, -1, 1, 3]:
        assert any(b["bucket_left"] <= value <= b["bucket_right"] for b in occupied)
    rows = json.loads(fs.read(f"/distributions/weights/{step:06d}.json", 4096, 0))
    assert [(r["percentile"], r["value"]) for r in (rows[0], rows[-1])] == [
        (0, -3),
        (100, 3),
    ]
    if step == 14:
        np.testing.assert_allclose(
            [r["value"] for r in rows],
            [-3 + 0.06 * r["percentile"] for r in rows],
            atol=1e-12,
        )
    else:
        expected = [
            -3,
            -2.967591894,
            -2.923006490,
            -0.987507425,
            0.917246401,
            0.987507425,
            2.923006490,
            2.967591894,
            3,
        ]
        np.testing.assert_allclose([r["value"] for r in rows], expected)


@pytest.mark.parametrize(
    ("producer", "path", "digest"),
    [
        (
            "tf115-native",
            "/images/checker/image/000012.png",
            "07604aee11bab41aea71f584cdd50dd71d483b3cf0a424eed6ce0592b05e06eb",
        ),
        (
            "tf200-native",
            "/images/quadrants/000014_0.png",
            "4d55a456a4fc58da53f26f78bc77917fbfcda384100c1a469c0526aca72a0cf0",
        ),
        (
            "tf200-native",
            "/images/quadrants/000014_1.png",
            "60149dd216509fa087fd1657b278784032a29af82ddd9655af56e3b173ee38a8",
        ),
    ],
)
def test_native_rgb(producer: str, path: str, digest: str) -> None:
    """Export the exact PNGs whose RGB pixels the native generators verified."""
    fs = TensorBoardFS(ROOT / producer)
    assert sha256(fs.read(path, 4096, 0)).hexdigest() == digest


@pytest.mark.parametrize(
    ("producer", "path", "frequency"),
    [
        ("tf115-native", "/audio/concert_a/audio/000012.wav", 440),
        ("tf200-native", "/audio/tone/000014.wav", 330),
    ],
)
def test_historical_audio(producer: str, path: str, frequency: int) -> None:
    fs = TensorBoardFS(ROOT / producer)
    with wave.open(BytesIO(fs.read(path, 8192, 0))) as wav:
        assert (
            wav.getframerate(),
            wav.getsampwidth(),
            wav.getnchannels(),
            wav.getnframes(),
        ) == (8000, 2, 1, 2000)
        pcm = np.frombuffer(wav.readframes(2000), dtype="<i2")
    expected = 16384 * np.sin(2 * np.pi * frequency * np.arange(2000) / 8000)
    np.testing.assert_allclose(pcm, expected, atol=1)


def test_v1_session_graph() -> None:
    fs = TensorBoardFS(ROOT / "tf115-native")
    graph = compat_pb2.GraphDef.FromString(fs.read("/graphs/graph.pb", 1 << 20, 0))
    nodes = {node.name: node for node in graph.node}
    assert (nodes["samples"].op, nodes["squared"].op, list(nodes["squared"].input)) == (
        "Placeholder",
        "Square",
        ["samples"],
    )


@pytest.mark.parametrize(
    ("producer", "step"), [("tf115-native", 12), ("tf200-native", 14)]
)
def test_weighted_classifier_and_geometry(producer: str, step: int) -> None:
    fs = TensorBoardFS(ROOT / producer)
    rows = json.loads(
        fs.read(f"/pr_curves/classifier/pr_curves/{step:06d}.json", 4096, 0)
    )
    for row, (threshold, tp, fp) in zip(
        rows,
        [(0, 3, 4), (0.25, 3, 1), (0.5, 2, 1), (0.75, 2, 0), (1, 0, 0)],
        strict=True,
    ):
        assert row == pytest.approx(
            {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": 3 - tp,
                "true_negatives": 4 - fp,
                "recall": tp / 3,
                "precision": tp / (tp + fp) if tp + fp else 0,
            }
        )
    obj = fs.read(f"/meshes/surface/{step:06d}.obj", 4096, 0).splitlines()
    assert obj[1:] == [b"v 0.0 0.0 0.0", b"v 2.0 0.0 0.0", b"v 0.0 1.0 0.0", b"f 1 2 3"]
    cloud = fs.read(f"/meshes/cloud/{step:06d}.ply", 4096, 0)
    header, body = cloud.split(b"end_header\n")
    assert b"element vertex 3" in header and b"element face" not in header
    assert list(struct.iter_unpack("<3f3B", body)) == [
        (0, 0, 0, 255, 0, 0),
        (2, 0, 0, 0, 255, 0),
        (0, 1, 0, 0, 0, 255),
    ]


@pytest.mark.parametrize("producer", ["tf115-native", "tf200-native"])
def test_dashboard_plugins(producer: str) -> None:
    fs = TensorBoardFS(ROOT / producer)
    layout = json.loads(fs.read("/custom_scalars/layout.json", 4096, 0))
    assert layout["category"] == [
        {
            "title": "signal",
            "chart": [{"title": "power", "multiline": {"tag": ["mean_square"]}}],
        }
    ]
    hparams = json.loads(fs.read("/hparams/hparams.json", 8192, 0))
    (rate,) = hparams["experiment"]["hparam_infos"]
    assert (rate["name"], rate["domain_discrete"]) == ("rate", [0.25, 0.5])
    assert hparams["session_start"] == {
        "group_name": "calibration",
        "start_time_secs": 2,
        "hparams": {"rate": 0.5},
    }
    assert hparams["metrics"] == {"mean_square": 5.0}
    config = fs.read("/projector/projector_config.pbtxt", 4096, 0)
    for field, value in [
        (b"tensor_name", b"corners"),
        (b"tensor_path", b"points.tsv"),
        (b"metadata_path", b"labels.tsv"),
    ]:
        assert field + b': "' + value + b'"' in config
    assert fs.read("/projector/labels.tsv", 4096, 0).splitlines() == [
        b"origin",
        b"right",
        b"up",
    ]
    np.testing.assert_array_equal(
        np.loadtxt(BytesIO(fs.read("/projector/points.tsv", 4096, 0))),
        [[0, 0], [2, 0], [0, 1]],
    )


@pytest.mark.parametrize(
    ("producer", "step", "names"),
    [("tf115-native", 12, {"squared"}), ("tf200-native", 14, {"Square", "Sum"})],
)
def test_native_cpu_profile(producer: str, step: int, names: set[str]) -> None:
    fs = TensorBoardFS(ROOT / producer)
    payload = fs.read(f"/graphs/signal/{step:06d}.run_metadata.pb", 8192, 0)
    metadata = compat_pb2.RunMetadata.FromString(payload)
    timed = {
        node.node_name
        for device in metadata.step_stats.dev_stats
        if "CPU" in device.device
        for node in device.node_stats
        if node.all_end_rel_micros > 0
    }
    assert names <= timed
    if step == 14:
        sidecar = fs.read("/profile/signal/local.trace", 65536, 0)
        assert (
            sidecar
            == (ROOT / producer / "plugins/profile/signal/local.trace").read_bytes()
        )
        assert all(name.encode() in sidecar for name in names)
