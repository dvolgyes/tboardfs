import json
from pathlib import Path

from tboardfs import TensorBoardFS, find_tensorboard_files


FIXTURE_ROOT = Path("test-logs")


def test_generated_logs_include_both_producers() -> None:
    """Generated fixtures include both requested producer directories."""
    event_files = find_tensorboard_files(FIXTURE_ROOT)

    assert (FIXTURE_ROOT / "tensorboardx") in {path.parent for path in event_files}
    assert (FIXTURE_ROOT / "tensorboard") in {path.parent for path in event_files}


def test_tensorboardx_fixture_exposes_clean_representations() -> None:
    """TensorBoardX fixtures expose typed virtual files."""
    fs = TensorBoardFS(FIXTURE_ROOT, step_digits=3)

    assert fs.readdir("/tensorboardx/scalars/train")[-3:] == [
        "f1_score.json",
        "f1_score.npz",
        "f1_score.tsv",
    ]
    assert "loss.json" in fs.readdir("/tensorboardx/scalars")
    assert "\n  {" in fs.read("/tensorboardx/scalars/loss.json", 10000, 0).decode()
    loss_rows = json.loads(fs.read("/tensorboardx/scalars/loss.json", 10000, 0))
    assert [row["step"] for row in loss_rows] == [1, 2, 3]
    assert [row["epoch"] for row in loss_rows] == [0.0, 1.0, 2.0]
    assert fs.readdir("/tensorboardx/histograms/weights")[-3:] == [
        "003.json",
        "003.npz",
        "003.tsv",
    ]
    assert fs.readdir("/tensorboardx/distributions/weights")[-3:] == [
        "003.json",
        "003.npz",
        "003.tsv",
    ]
    assert fs.readdir("/tensorboardx/videos/clip")[-4:] == [
        "001.gif",
        "002.gif",
        "003.gif",
        "raw",
    ]
    assert fs.readdir("/tensorboardx/meshes/box")[-3:] == [
        "003.json",
        "003.npz",
        "003.obj",
    ]
    assert "001.npz" not in fs.readdir("/tensorboardx/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboardx/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboardx/tensors/activations")
    assert "layout.json" in fs.readdir("/tensorboardx/custom_scalars")
    assert "metadata.tsv" in fs.readdir("/tensorboardx/projector")
    assert "trace.json" in fs.readdir("/tensorboardx/profile")
    assert "hparams.json" in fs.readdir("/tensorboardx/hparams")
    assert not any(name[:1].isdigit() for name in fs.readdir("/tensorboardx"))


def test_tensorboard_fixture_exposes_clean_representations() -> None:
    """Raw TensorBoard fixtures expose typed virtual files."""
    fs = TensorBoardFS(FIXTURE_ROOT, step_digits=3)

    f1_rows = json.loads(fs.read("/tensorboard/scalars/train/f1_score.json", 10000, 0))
    assert [row["step"] for row in f1_rows] == [1, 2, 3]
    assert [row["relative_time"] for row in f1_rows] == [0.0, 1.0, 2.0]
    assert [row["epoch"] for row in f1_rows] == [0.0, 1.0, 2.0]
    assert fs.readdir("/tensorboard/images/eval/sample")[-3:] == [
        "001.png",
        "002.png",
        "003.png",
    ]
    assert fs.readdir("/tensorboard/audio/tone")[-3:] == [
        "001.wav",
        "002.wav",
        "003.wav",
    ]
    assert fs.readdir("/tensorboard/videos/clip")[-3:] == [
        "001.gif",
        "002.gif",
        "003.gif",
    ]
    assert fs.readdir("/tensorboard/text/notes")[-3:] == [
        "001.txt",
        "002.txt",
        "003.txt",
    ]
    assert "001.tsv" in fs.readdir("/tensorboard/histograms/weights")
    assert "001.tsv" in fs.readdir("/tensorboard/distributions/weights")
    assert "graph.pb" in fs.readdir("/tensorboard/graphs")
    assert "metadata.tsv" in fs.readdir("/tensorboard/projector")
    assert "hparams.json" in fs.readdir("/tensorboard/hparams")
    assert "001.npz" in fs.readdir("/tensorboard/meshes/box")
    assert "001.obj" in fs.readdir("/tensorboard/meshes/box")
    assert "001.npz" not in fs.readdir("/tensorboard/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboard/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboard/tensors/activations")
    assert "layout.json" in fs.readdir("/tensorboard/custom_scalars")
    assert "trace.json" in fs.readdir("/tensorboard/profile")
    assert fs.readdir("/tensorboard/plugins") == [".", ".."]
