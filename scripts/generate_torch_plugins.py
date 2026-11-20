#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "tensorboard==2.21.0", "torch==2.12.0+cpu", "numpy==2.2.6",
#   "moviepy==1.0.3", "imageio==2.37.3", "pillow==11.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Generate native PyTorch summaries, projector assets, and a CPU Kineto trace."""

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)
from torch.utils.tensorboard import SummaryWriter


def _write_summaries(output: Path) -> None:
    """Use public SummaryWriter APIs without re-encoding any event or projector asset."""
    with SummaryWriter(str(output)) as writer:
        writer.add_pr_curve(
            "weighted",
            np.array([0, 1, 0, 1, 1, 0]),
            np.array([0.05, 0.92, 0.38, 0.71, 0.64, 0.19]),
            global_step=3,
            num_thresholds=5,
            weights=np.array([1, 2, 1, 3, 2, 1]),
        )
        writer.add_pr_curve_raw(
            "precomputed",
            np.array([4, 3, 1]),
            np.array([3, 1, 0]),
            np.array([0, 2, 3]),
            np.array([0, 1, 3]),
            np.array([4 / 7, 0.75, 1]),
            np.array([1, 0.75, 0.25]),
            global_step=3,
            num_thresholds=3,
        )
        writer.add_histogram(
            "observations",
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            bins=[0, 2, 4],
            global_step=4,
        )
        writer.add_histogram_raw(
            "interval", 0.0, 2.0, 2, 2.0, 4.0, [0.0, 2.0], [0.0, 2.0], global_step=4
        )
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        colors = torch.tensor([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        writer.add_mesh(
            "triangle",
            vertices,
            colors=colors,
            faces=torch.tensor([[[0, 1, 2]]]),
            global_step=5,
        )
        writer.add_mesh("colored-cloud", vertices, colors=colors, global_step=5)
        frames = torch.zeros((1, 3, 3, 24, 32), dtype=torch.uint8)
        for index in range(3):
            x = index * 10
            frames[0, index, index, 8:16, x : x + 8] = 255
        writer.add_video("motion", frames, global_step=6, fps=2)
        writer.add_image("still", frames[0, 1], global_step=6)
        t = torch.arange(2000, dtype=torch.float64) / 8000
        writer.add_audio(
            "tone",
            0.5 * torch.sin(2 * torch.pi * 440 * t),
            global_step=6,
            sample_rate=8000,
        )
        writer.add_text("notes", "Torch API text\nsecond line", global_step=7)
        for tag, score in {
            "score": 0.5,
            "low": 0.25,
            "high": 0.75,
            "loss/train": 1.0,
            "loss/eval": 1.5,
        }.items():
            writer.add_scalar(tag, score, global_step=7)
        writer.add_custom_scalars(
            {
                "comparison": {
                    "bounds": ["Margin", ["score", "low", "high"]],
                    "loss": ["Multiline", ["loss/train", "loss/eval"]],
                }
            }
        )
        writer.add_embedding(
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            metadata=["red", "green", "blue"],
            label_img=frames[0].float() / 255,
            global_step=8,
            tag="features",
        )
        writer.add_hparams(
            {"lr": 0.125, "optimizer": "sgd"},
            {"accuracy": 0.875},
            run_name="trial",
            global_step=9,
        )
    for directory in (output, output / "trial"):
        (event,) = directory.glob("events.out.tfevents.*")
        event.rename(directory / "events.out.tfevents.torch-plugins")
    event_file = output / "events.out.tfevents.torch-plugins"
    images = {
        v.tag: v.image.encoded_image_string
        for e in LegacyEventFileLoader(str(event_file)).Load()
        for v in e.summary.value
        if v.HasField("image")
    }
    expected = frames[0].permute(0, 2, 3, 1).numpy()
    with Image.open(BytesIO(images["motion"])) as gif:
        if gif.format != "GIF" or gif.n_frames != 3:
            raise ValueError("add_video did not produce a three-frame GIF")
        for i, rgb in enumerate(expected):
            gif.seek(i)
            np.testing.assert_array_equal(np.asarray(gif.convert("RGB")), rgb)
    with Image.open(BytesIO(images["still"])) as png:
        np.testing.assert_array_equal(np.asarray(png), expected[1])
    with Image.open(output / "00008/features/sprite.png") as sprite:
        for rgb, (x, y) in zip(expected, [(0, 0), (32, 0), (0, 24)], strict=True):
            np.testing.assert_array_equal(
                np.asarray(sprite)[y : y + 24, x : x + 32], rgb
            )


def _write_profile(output: Path) -> None:
    """Keep timed CPU operations from the handler; omit machine and process metadata."""
    dest = output / "plugins/profile/capture"
    dest.mkdir(parents=True)
    with (
        profile(
            activities=[ProfilerActivity.CPU],
            on_trace_ready=tensorboard_trace_handler(str(dest), worker_name="cpu"),
        ),
        record_function("torch_plugin_fixture_operation"),
    ):
        torch.arange(64, dtype=torch.float32).square().sum().item()
    (trace,) = dest.glob("*.pt.trace.json")
    data = json.loads(trace.read_bytes())
    events = [
        e
        for e in data["traceEvents"]
        if e.get("cat") in {"user_annotation", "cpu_op"} and e["ph"] == "X"
    ]
    if len({(e["pid"], e["tid"]) for e in events}) != 1:
        raise ValueError("expected CPU operations from exactly one profiler thread")
    start = min(e["ts"] for e in events)
    cleaned = [
        {k: e[k] for k in ("ph", "cat", "name", "dur")}
        | {"ts": e["ts"] - start, "pid": 1, "tid": 1}
        for e in events
    ]
    (dest / "cpu.pt.trace.json").write_text(
        json.dumps({"schemaVersion": data["schemaVersion"], "traceEvents": cleaned})
        + "\n"
    )
    trace.unlink()


def main() -> None:
    """Write the native Torch corpus to a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    _write_summaries(output)
    _write_profile(output)


if __name__ == "__main__":
    main()
