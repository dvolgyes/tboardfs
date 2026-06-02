import json
from pathlib import Path

from tboardfs import TensorBoardFS, find_tensorboard_files


FIXTURE_ROOT = Path("test-logs")


def test_generated_logs_include_both_producers() -> None:
    """Generated fixtures include both requested producer directories."""
    event_files = find_tensorboard_files(FIXTURE_ROOT)

    assert (FIXTURE_ROOT / "tensorboardx") in {path.parent for path in event_files}
    assert (FIXTURE_ROOT / "tensorboard") in {path.parent for path in event_files}


def test_tensorboardx_fixture_exposes_public_api_outputs() -> None:
    """TensorBoardX fixtures expose expected virtual files."""
    fs = TensorBoardFS(FIXTURE_ROOT, step_digits=3)

    assert fs.readdir("/tensorboardx/scalars/train")[-3:] == [
        "loss.json",
        "loss.npz",
        "loss.tsv",
    ]
    loss_rows = json.loads(fs.read("/tensorboardx/scalars/loss.json", 10000, 0))
    assert [row["step"] for row in loss_rows] == [1, 2, 3]
    assert [row["epoch"] for row in loss_rows] == [0.0, 1.0, 2.0]
    assert fs.readdir("/tensorboardx/images/sample")[-3:] == [
        "001.png",
        "002.png",
        "003.png",
    ]
    assert fs.readdir("/tensorboardx/images/clip")[-3:] == [
        "001.gif",
        "002.gif",
        "003.gif",
    ]
    assert fs.readdir("/tensorboardx/audio/tone")[-3:] == [
        "001.wav",
        "002.wav",
        "003.wav",
    ]
    assert fs.readdir("/tensorboardx/text/notes/text_summary")[-3:] == [
        "001.txt",
        "002.txt",
        "003.txt",
    ]
    assert "graph.pb" in fs.readdir("/tensorboardx/graphs")
    assert "projector_config.pbtxt" in fs.readdir("/tensorboardx/projector")
    assert "triangle_1" in fs.readdir("/tensorboardx/meshes/shape")
    assert "001.json" in fs.readdir("/tensorboardx/pr_curves/quality/pr")
    assert tensorboardx_hparams_run(fs) is not None


def test_tensorboard_fixture_exposes_native_plugin_shapes() -> None:
    """Raw TensorBoard fixtures expose expected virtual files."""
    fs = TensorBoardFS(FIXTURE_ROOT, step_digits=3)

    loss_rows = json.loads(fs.read("/tensorboard/scalars/train/loss.json", 10000, 0))
    assert [row["step"] for row in loss_rows] == [1, 2, 3]
    assert [row["relative_time"] for row in loss_rows] == [0.0, 1.0, 2.0]
    assert [row["epoch"] for row in loss_rows] == [0.0, 1.0, 2.0]
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
    assert "weights.json" in fs.readdir("/tensorboard/histograms")
    assert "weights.json" in fs.readdir("/tensorboard/distributions")
    assert "graph.pb" in fs.readdir("/tensorboard/graphs")
    assert "metadata.tsv" in fs.readdir("/tensorboard/projector")
    assert "001.json" in fs.readdir("/tensorboard/meshes/shape/triangle")
    assert "001.json" in fs.readdir("/tensorboard/pr_curves/quality/pr")
    assert "001.json" in fs.readdir("/tensorboard/hparams/hparams/session")
    assert "001.json" in fs.readdir("/tensorboard/tensors/tensor/blob")


def tensorboardx_hparams_run(fs: TensorBoardFS) -> str | None:
    """Return the TensorBoardX child run that contains add_hparams output."""
    for name in fs.readdir("/tensorboardx"):
        if not name[:1].isdigit():
            continue
        if "_hparams_" in fs.readdir(f"/tensorboardx/{name}/hparams"):
            return name
    return None
