#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lightning==2.6.6", "pytorch-lightning==2.6.6", "torch==2.12.0+cpu", "tensorboard==2.21.0",
#   "numpy==2.2.6", "moviepy==1.0.3", "imageio==2.37.3", "pillow==11.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Use Lightning's TensorBoard logger experiment and its CPU PyTorchProfiler."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from PIL import Image
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


def _summaries(output: Path) -> None:
    """Delegate summary APIs to the actual experiment owned by TensorBoardLogger."""
    logger = TensorBoardLogger(
        save_dir=str(output), name="", version="", default_hp_metric=False
    )
    writer = logger.experiment
    writer.add_pr_curve(
        "classifier",
        np.array([1, 0, 1, 0]),
        np.array([1.0, 0.7, 0.45, 0.1]),
        num_thresholds=3,
        weights=np.array([3.0, 2.0, 4.0, 1.0]),
        global_step=4,
    )
    vertices = torch.tensor(
        [[[-1.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 2.0, 1.0], [0.0, 1.0, -1.0]]]
    )
    writer.add_mesh(
        "surface", vertices, faces=torch.tensor([[[0, 1, 2], [1, 0, 3]]]), global_step=4
    )
    writer.add_mesh(
        "scan",
        vertices,
        colors=torch.tensor(
            [[[0, 64, 255], [255, 32, 0], [0, 255, 16], [128, 0, 32]]],
            dtype=torch.uint8,
        ),
        global_step=4,
    )
    writer.add_tensor(
        "calibration",
        torch.tensor([[1.0 + 2.0**-45, -3.125]], dtype=torch.float64),
        global_step=4,
    )
    for tag, metric in {"estimate": 2.25, "low": 1.5, "high": 3.0}.items():
        logger.log_metrics({tag: metric}, step=4)
    writer.add_custom_scalars(
        {"estimates": {"interval": ["Margin", ["estimate", "low", "high"]]}}
    )
    frames = torch.zeros((3, 3, 16, 16), dtype=torch.uint8)
    frames[:, :, 12:, :] = 48
    for i in range(3):
        frames[i, i, 5:10, 2 + 4 * i : 6 + 4 * i] = 255
    writer.add_video("tracking", frames[None], global_step=4, fps=3)
    writer.add_embedding(
        torch.tensor([[-1.0, 3.0], [0.25, 1.0], [2.0, -0.5]]),
        metadata=["left", "center", "right"],
        label_img=frames.float() / 255,
        global_step=4,
        tag="tracks",
    )
    writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event = event.rename(output / "events.out.tfevents.lightning-plugins")
    (payload,) = (
        v.image.encoded_image_string
        for e in LegacyEventFileLoader(str(event)).Load()
        for v in e.summary.value
        if v.tag == "tracking"
    )
    expected = frames.permute(0, 2, 3, 1).numpy()
    with Image.open(BytesIO(payload)) as gif:
        if gif.format != "GIF" or gif.n_frames != 3:
            raise ValueError(
                "expected three GIF frames from the Lightning logger experiment"
            )
        for i, pixels in enumerate(expected):
            gif.seek(i)
            np.testing.assert_array_equal(np.asarray(gif.convert("RGB")), pixels)
    with Image.open(output / "00004/tracks/sprite.png") as sprite:
        for pixels, (x, y) in zip(expected, [(0, 0), (16, 0), (0, 16)], strict=True):
            np.testing.assert_array_equal(
                np.asarray(sprite)[y : y + 16, x : x + 16], pixels
            )


def _profile(output: Path) -> None:
    """Keep only the Lightning annotation and measured CPU operations from its Chrome export."""
    action = "calibration_training_step"
    with TemporaryDirectory() as temporary:
        profiler = PyTorchProfiler(
            dirpath=temporary,
            filename="calibration",
            schedule=None,
            activities=[torch.profiler.ProfilerActivity.CPU],
        )
        with profiler.profile(action):
            torch.tensor([1.0, 2.0]).square().sum().item()
        profiler.summary()
        (trace,) = Path(temporary).glob("*.pt.trace.json")
        data = json.loads(trace.read_bytes())
    names = {"[pl][profile]" + action, "aten::square", "aten::sum", "aten::item"}
    events = [
        e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("name") in names
    ]
    if {e["name"] for e in events} != names or len(
        {(e["pid"], e["tid"]) for e in events}
    ) != 1:
        raise ValueError(
            "expected the Lightning annotation and all three CPU operations in one thread"
        )
    start = min(e["ts"] for e in events)
    excerpt = [
        {k: e[k] for k in ("name", "cat", "dur", "ph")}
        | {"pid": 1, "tid": 1, "ts": e["ts"] - start}
        for e in events
    ]
    dest = output / "plugins/profile/capture"
    dest.mkdir(parents=True)
    (dest / "lightning.pt.trace.json").write_text(
        json.dumps({"schemaVersion": data["schemaVersion"], "traceEvents": excerpt})
        + "\n"
    )


def main() -> None:
    """Generate the Lightning plugin corpus in a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    _summaries(output)
    _profile(output)


if __name__ == "__main__":
    main()
