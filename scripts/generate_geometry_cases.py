#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorboardX==2.6.5", "numpy==2.4.6"]
# ///
"""Write compact native TensorBoardX geometry and plugin event cases."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tensorboardX import SummaryWriter


def _write(output: Path) -> Path:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        writer = SummaryWriter(logdir=temporary)
        try:
            vertices = np.array(
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32
            )
            faces = np.array(
                [[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]], dtype=np.int32
            )
            colors = np.array(
                [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]],
                dtype=np.uint8,
            )
            writer.add_mesh(
                "tetrahedron",
                vertices,
                colors,
                faces,
                config_dict={"material": {"wireframe": True}},
                global_step=4,
                walltime=40.0,
            )
            writer.add_mesh(
                "cloud",
                np.array([[[2, 3, 5], [-2, 4, 1]]], dtype=np.float32),
                global_step=4,
                walltime=40.0,
            )
            writer.add_histogram_raw(
                "interval",
                min=0.0,
                max=2.0,
                num=2,
                sum=2.0,
                sum_squares=4.0,
                bucket_limits=[2.0],
                bucket_counts=[2.0],
                global_step=4,
                walltime=40.0,
            )
            writer.add_pr_curve(
                "weighted-pr",
                np.array([1, 0, 1, 0], dtype=np.uint8),
                np.array([0.9, 0.8, 0.4, 0.1], dtype=np.float32),
                global_step=4,
                num_thresholds=5,
                weights=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                walltime=40.0,
            )
            writer.add_pr_curve_raw(
                "raw-pr",
                np.array([4, 3, 2], dtype=np.float32),
                np.array([1, 2, 3], dtype=np.float32),
                np.array([5, 4, 3], dtype=np.float32),
                np.array([0, 1, 2], dtype=np.float32),
                np.array([0.8, 0.6, 0.4], dtype=np.float32),
                np.array([1.0, 0.5, 0.0], dtype=np.float32),
                global_step=4,
                num_thresholds=3,
                walltime=40.0,
            )
            writer.flush()
        finally:
            writer.close()
        (source,) = Path(temporary).glob("events.out.tfevents.*")
        destination = output / "events.out.tfevents.geometry"
        destination.hardlink_to(source)
    return destination


def main() -> None:
    """Write an event file in a caller-selected new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    _write(parser.parse_args().output)


if __name__ == "__main__":
    main()
