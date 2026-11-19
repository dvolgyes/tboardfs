"""Logs written by TensorFlow 2, Keras and PyTorch Lightning read correctly.

The fixtures come from scripts/generate_tf2_logs.py and
scripts/generate_lightning_logs.py.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tboardfs import TensorBoardFS

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    """Return the fixture log directory with 3-digit step names."""
    return TensorBoardFS(ROOT / "test-logs", step_digits=3)


def _json(fs: TensorBoardFS, path: str) -> Any:
    return json.loads(fs.read(path, 1 << 20, 0))


def test_keras_epochs_start_at_step_zero(fs: TensorBoardFS) -> None:
    """The TensorBoard callback writes epoch 0 at step 0, which has no step field."""
    for split in ("train", "validation"):
        rows = _json(fs, f"/tf2/keras/{split}/scalars/epoch_loss.json")
        assert [row["step"] for row in rows] == [0, 1]


def test_tf2_summary_scalars_text_and_hparams(fs: TensorBoardFS) -> None:
    """tf.summary scalars, text and hparams have their typed files."""
    rows = _json(fs, "/tf2/summary/scalars/loss.json")
    hparams = _json(fs, "/tf2/summary/hparams/hparams.json")

    assert [row["step"] for row in rows] == [1, 2, 3]
    assert fs.read("/tf2/summary/text/note/002.txt", 100, 0) == b"step **2**"
    assert hparams["session_start"]["hparams"] == {"lr": 0.01}
    assert hparams["metrics"] == {"loss": np.float32(1 / 3)}


def test_tf2_tensor_without_plugin_name_is_a_tensor_file(fs: TensorBoardFS) -> None:
    """tf.summary.write with no metadata writes a tensor with no plugin name."""
    blob = fs.read("/tf2/summary/tensors/raw/002.npy", 1 << 20, 0)

    np.testing.assert_array_equal(
        np.load(io.BytesIO(blob)), np.arange(6.0).reshape(2, 3) * 2
    )


def test_lightning_metrics_hparams_and_media(fs: TensorBoardFS) -> None:
    """TensorBoardLogger metrics, hparams, graph and SummaryWriter media."""
    run = "/lightning/version_0"
    hparams = _json(fs, f"{run}/hparams/hparams.json")

    assert [row["step"] for row in _json(fs, f"{run}/scalars/val_loss.json")] == [1, 3]
    assert hparams["session_start"]["hparams"] == {"lr": 0.1}
    assert hparams["metrics"] == {"hp_metric": -1.0}
    assert fs.read(f"{run}/hparams/hparams.yaml", 100, 0) == b"lr: 0.1\n"
    assert fs.readdir(f"{run}/images/pattern")[2:] == ["000.png", "001.png"]
    assert fs.read(f"{run}/images/pattern/000.png", 8, 0) == b"\x89PNG\r\n\x1a\n"
    assert fs.read(f"{run}/audio/tone/001.wav", 4, 0) == b"RIFF"
    # PyTorch add_graph also writes a tagged RunMetadata, which TensorBoard
    # shows as the profile of tag step1.
    assert fs.readdir(f"{run}/graphs")[2:] == ["graph.pb", "step1"]
    assert fs.readdir(f"{run}/graphs/step1")[2:] == [
        "000.run_metadata.pb",
        "000.run_metadata.pbtxt",
    ]
    profile = fs.read(f"{run}/graphs/step1/000.run_metadata.pbtxt", 1 << 20, 0)
    assert profile.startswith(b"step_stats {\n  dev_stats {\n")


def test_lightning_text_and_weight_percentiles(fs: TensorBoardFS) -> None:
    """Compare the sparse two-weight histogram to TensorBoard's reference reader."""
    run = "/lightning/version_0"
    assert fs.read(f"{run}/text/note/text_summary/001.txt", 256, 0) == b"epoch **1**"
    buckets = _json(fs, f"{run}/histograms/layer/weight/001.json")
    assert [row["count"] for row in buckets if row["count"]] == [1, 1]
    dist = _json(fs, f"{run}/distributions/layer/weight/001.json")
    np.testing.assert_array_equal(
        [r["percentile"] for r in dist],
        [0, 6.68, 15.87, 30.85, 50, 69.15, 84.13, 93.32, 100],
    )
    np.testing.assert_allclose(
        [r["value"] for r in dist],
        [
            0.37501847743988037,
            0.3768866772633304,
            0.3794568503737355,
            0.38364631644488545,
            0.7580548669750213,
            0.7676680901639027,
            0.775187990078756,
            0.779801333217441,
            0.7831546664237976,
        ],
        rtol=1e-7,
    )
