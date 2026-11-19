#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.12.0+cpu", "tensorboard==2.21.0", "numpy==2.2.6", "pillow==11.3.0"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Generate expanded coverage from the PyTorch CPU SummaryWriter."""

import argparse
from pathlib import Path

from library_writer_cases import write_cases
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    """Write native PyTorch cases to an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    with SummaryWriter(log_dir=str(output)) as writer:
        write_cases(writer, output, colored_surface=False)
    for event in output.rglob("events.out.tfevents.*"):
        event.rename(event.with_name("events.out.tfevents.torch-matrix"))


if __name__ == "__main__":
    main()
