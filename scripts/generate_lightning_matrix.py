#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lightning==2.6.6", "pytorch-lightning==2.6.6", "torch==2.12.0+cpu", "tensorboard==2.21.0",
#   "numpy==2.2.6", "pillow==11.3.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Generate delegated summaries from Lightning's real TensorBoard logger experiment."""

import argparse
from pathlib import Path

import torch
from library_writer_cases import write_cases
from lightning.pytorch.loggers import TensorBoardLogger


def main() -> None:
    """Write native Lightning delegation cases to an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    logger = TensorBoardLogger(
        save_dir=str(output), name="", version="", default_hp_metric=False
    )
    with logger.experiment as writer:
        writer.add_scalar(
            "precise", 16_777_219, 4, walltime=4, new_style=True, double_precision=True
        )
        writer.add_histogram_raw(
            "buckets", 0.5, 3, 3, 5.5, 13.25, [1, 2, 4], [1, 1, 1], 4, walltime=4
        )
        writer.add_pr_curve_raw(
            "precomputed",
            [4, 3, 1],
            [4, 1, 0],
            [0, 3, 4],
            [0, 1, 3],
            [0.5, 0.75, 1],
            [1, 0.75, 0.25],
            4,
            num_thresholds=3,
            walltime=4,
        )
        logger.log_metrics({"train": 0.75, "valid": 0.625}, step=4)
        writer.add_custom_scalars_multilinechart(
            ["train", "valid"], category="accuracy", title="comparison"
        )
        writer.add_tensor(
            "complex64",
            torch.tensor([1 + 2j, -3.5 - 0.5j], dtype=torch.complex64),
            4,
            walltime=4,
        )
        writer.add_tensor(
            "float16", torch.tensor([1.5, -2], dtype=torch.float16), 4, walltime=4
        )
        write_cases(writer, output, colored_surface=True)
    for event in output.rglob("events.out.tfevents.*"):
        event.rename(event.with_name("events.out.tfevents.lightning-matrix"))


if __name__ == "__main__":
    main()
