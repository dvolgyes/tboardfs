#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorboard==2.21.0", "torch==2.12.0+cpu"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Generate real PyTorch SummaryWriter scalar and tensor records."""

import argparse
from pathlib import Path
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter


SCALAR = 16_777_219.0


def main() -> None:
    """Write true SummaryWriter events into a new or empty output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        writer = SummaryWriter(log_dir=temporary)
        try:
            writer.add_scalar("scalar/legacy", SCALAR, global_step=1)
            writer.add_scalar("scalar/new-style", SCALAR, global_step=1, new_style=True)
            writer.add_scalar(
                "scalar/double-precision",
                SCALAR,
                global_step=1,
                new_style=True,
                double_precision=True,
            )
            tensors = {
                "float16": torch.tensor([1.5, -2.0], dtype=torch.float16),
                "bfloat16": torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
                "float32": torch.tensor([1.25, -2.5], dtype=torch.float32),
                "float64": torch.tensor([1.25, -2.5], dtype=torch.float64),
                "uint8": torch.tensor([0, 255], dtype=torch.uint8),
                "int8": torch.tensor([-128, 127], dtype=torch.int8),
                "int16": torch.tensor([-32768, 32767], dtype=torch.int16),
                "int32": torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32),
                "int64": torch.tensor([-(2**63), 2**63 - 1], dtype=torch.int64),
                "bool": torch.tensor([False, True], dtype=torch.bool),
                "complex64": torch.tensor([1 + 2j, -3.5 - 0.5j], dtype=torch.complex64),
                "complex128": torch.tensor(
                    [1 + 2j, -3.5 - 0.5j], dtype=torch.complex128
                ),
            }
            for name, tensor in tensors.items():
                writer.add_tensor(f"tensor/{name}", tensor, global_step=1)
            writer.flush()
        finally:
            writer.close()
        events = list(Path(temporary).glob("events.out.tfevents.*"))
        if len(events) != 1:
            raise RuntimeError(f"expected one event file, found {len(events)}")
        events[0].replace(output / "events.out.tfevents.torch")


if __name__ == "__main__":
    main()
