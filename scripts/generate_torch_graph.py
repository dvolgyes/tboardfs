#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.12.0+cpu", "tensorboard==2.21.0"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Trace a deterministic CPU classifier with PyTorch's native graph writer."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    """Trace a two-feature signed difference followed by positive-part activation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be new or empty")
    difference = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        difference.weight.copy_(torch.tensor([[1.0, -1.0]]))
    model = torch.nn.Sequential(difference, torch.nn.ReLU()).eval()
    samples = torch.tensor([[5.0, 2.0], [2.0, 5.0]])
    if not torch.equal(model(samples), torch.tensor([[3.0], [0.0]])):
        raise ValueError("classifier did not recover positive differences")
    with SummaryWriter(str(output)) as writer:
        writer.add_graph(model, samples)
    (source,) = output.glob("events.out.tfevents.*")
    data = source.read_bytes()
    for private in (
        Path.home(),
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(sys.prefix),
    ):
        if str(private).encode() in data:
            raise ValueError(f"graph unexpectedly embeds local path {private}")
    source.rename(output / "events.out.tfevents.torch-graph")


if __name__ == "__main__":
    main()
