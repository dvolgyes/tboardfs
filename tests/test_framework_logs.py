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


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    """Return the fixture log directory with 3-digit step names."""
    return TensorBoardFS(Path("test-logs"), step_digits=3)


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
