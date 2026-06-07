from io import BytesIO
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tboardfs import TensorBoardFS, find_tensorboard_files


FIXTURE_ROOT = Path("test-logs")


def test_generated_logs_include_both_producers() -> None:
    """Generated fixtures include both requested producer directories."""
    event_files = find_tensorboard_files(FIXTURE_ROOT)

    assert (FIXTURE_ROOT / "tensorboardx") in {path.parent for path in event_files}
    assert (FIXTURE_ROOT / "tensorboard") in {path.parent for path in event_files}


def test_runtime_filesystem_does_not_import_tensorboard() -> None:
    """Production import and filesystem parsing do not require forbidden stacks."""
    script = """
import builtins
from pathlib import Path
import sys

real_import = builtins.__import__
FORBIDDEN = {"tensorboard", "tensorboardX", "tensorboardx", "tensorflow"}

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", maxsplit=1)[0] in FORBIDDEN:
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from tboardfs import TensorBoardFS

fs = TensorBoardFS(Path("test-logs"), step_digits=3)
assert "layout.json" in fs.readdir("/tensorboard/custom_scalars")
assert "hparams.json" in fs.readdir("/tensorboard/hparams")
assert "001.npz" in fs.readdir("/tensorboard/meshes/box")
assert "layout.json" in fs.readdir("/tensorboardx/custom_scalars")
assert "hparams.json" in fs.readdir("/tensorboardx/hparams")
assert "001.npz" in fs.readdir("/tensorboardx/meshes/box")
assert not any(
    module == root or module.startswith(root + ".")
    for module in sys.modules
    for root in FORBIDDEN
)
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )


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
    assert fs.readdir("/tensorboardx/custom_scalars") == [".", "..", "layout.json"]
    assert "metadata.tsv" in fs.readdir("/tensorboardx/projector")
    assert "trace.json" in fs.readdir("/tensorboardx/profile")
    assert "hparams.json" in fs.readdir("/tensorboardx/hparams")
    hparams = json.loads(fs.read("/tensorboardx/hparams/hparams.json", 10000, 0))
    assert hparams["session_start"]["hparams"]["optimizer"] == "sgd"
    assert hparams["metrics"]["hparam/loss"] == 0.25
    assert not any(name[:1].isdigit() for name in fs.readdir("/tensorboardx"))
    assert fs.readdir("/tensorboardx/plugins") == [".", ".."]


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
    hparams = json.loads(fs.read("/tensorboard/hparams/hparams.json", 10000, 0))
    assert [item["name"] for item in hparams["experiment"]["hparam_infos"]] == [
        "optimizer",
        "lr",
    ]
    assert hparams["session_start"]["hparams"] == {"lr": 0.01, "optimizer": "adam"}
    assert hparams["metrics"]["hparam/f1_score"] == 0.875
    mesh_files = fs.readdir("/tensorboard/meshes/box")
    assert mesh_files == [
        ".",
        "..",
        "001.json",
        "001.npz",
        "001.obj",
        "002.json",
        "002.npz",
        "002.obj",
        "003.json",
        "003.npz",
        "003.obj",
    ]
    mesh_archive = np.load(
        BytesIO(fs.read("/tensorboard/meshes/box/001.npz", 10000, 0))
    )
    assert set(mesh_archive.files) == {"colors", "faces", "vertices"}
    assert mesh_archive["vertices"].shape == (1, 8, 3)
    assert mesh_archive["faces"].shape == (1, 12, 3)
    assert mesh_archive["colors"].shape == (1, 8, 3)
    assert "001.npz" not in fs.readdir("/tensorboard/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboard/pr_curves/quality/pr")
    assert "001.npy" in fs.readdir("/tensorboard/tensors/activations")
    assert "layout.json" in fs.readdir("/tensorboard/custom_scalars")
    assert fs.readdir("/tensorboard/custom_scalars") == [".", "..", "layout.json"]
    layout = json.loads(fs.read("/tensorboard/custom_scalars/layout.json", 10000, 0))
    assert layout["category"][0]["chart"][0]["multiline"]["tag"] == [
        "loss",
        "train/f1_score",
    ]
    assert "loss.json" in fs.readdir("/tensorboard/scalars")
    assert "trace.json" in fs.readdir("/tensorboard/profile")
    assert fs.readdir("/tensorboard/plugins") == [".", ".."]
