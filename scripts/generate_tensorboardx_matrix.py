#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorboardX==2.6.5", "numpy==2.2.6"]
# ///
"""Generate TensorBoardX batched geometry and file-writer binary strings."""

import argparse
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX.proto.summary_pb2 import Summary


def main() -> None:
    """Write native TensorBoardX cases to an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    vertices = np.array(
        [[[0, 0, 0], [2, 0, 0], [0, 1, 0]], [[5, 5, 5], [7, 5, 5], [5, 6, 5]]]
    )
    faces = np.array([[[0, 1, 2]], [[2, 1, 0]]])
    colors = np.array(
        [[[255, 0, 0], [0, 64, 0], [0, 0, 128]], [[0, 0, 128], [0, 64, 0], [255, 0, 0]]]
    )
    with SummaryWriter(logdir=str(output)) as writer:
        writer.add_mesh("surface", vertices, faces=faces, global_step=4, walltime=4)
        writer.add_mesh(
            "colored-cloud", vertices, colors=colors, global_step=4, walltime=4
        )
        writer.add_pr_curve(
            "unweighted",
            np.array([1, 0, 1, 0]),
            np.array([1.0, 0.75, 0.5, 0.25]),
            4,
            num_thresholds=3,
            walltime=4,
        )
        summary = Summary()
        tensor = summary.value.add(tag="binary-strings").tensor
        tensor.dtype = "DT_STRING"
        tensor.tensor_shape.dim.add(size=3)
        tensor.string_val.extend([b"", b"A\x00B", b"\xfe\xff"])
        writer.file_writer.add_summary(summary, global_step=4, walltime=4)
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.tensorboardx-matrix")


if __name__ == "__main__":
    main()
