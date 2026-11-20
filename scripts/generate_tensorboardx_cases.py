#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorboardX>=2.6.5,<3", "numpy>=2,<3"]
# ///
"""Generate a single-file TensorBoardX training and restart sequence."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tensorboardX import SummaryWriter


def main() -> None:
    """Write two training sessions in order into one restart fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    with TemporaryDirectory(dir=output) as temporary:
        work = Path(temporary)
        tag = "train/batch/loss"
        with SummaryWriter(logdir=str(work / "before")) as writer:
            writer.add_scalar(tag, 12, 0, walltime=1.0)
            writer.add_scalar("epoch", 1, 0, walltime=2.0)
            writer.add_scalar(tag, 999, 3, walltime=3.0)
        with SummaryWriter(logdir=str(work / "after"), purge_step=3) as writer:
            for walltime, (step, value, epoch) in enumerate(
                ((2**53 + 1, 7, 2**54), (2**53, 8, 2**53), (3, 9, 2), (3, 10, 2)),
                start=4,
            ):
                writer.add_scalar(tag, value, step, walltime=walltime)
                writer.add_scalar("epoch", epoch, step, walltime=walltime)
            writer.add_histogram(
                "weights", np.array([-1.0, 0.0, 1.0]), 3, bins=3, walltime=8.0
            )

        merged = work / "joined"
        with merged.open("xb") as destination:
            for session in ("before", "after"):
                (event_file,) = (work / session).glob("events.out.tfevents.*")
                with event_file.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        merged.rename(output / "events.out.tfevents.tensorboardx")


if __name__ == "__main__":
    main()
