#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0", "numpy==2.5.5"]
# ///
"""Write posed terrain and range scans with the TensorBoard v2 mesh API."""

import argparse
from pathlib import Path

import numpy as np
from tensorboard.plugins.mesh import summary_v2
import tensorflow as tf


def _terrain() -> None:
    """Triangulate a colored saddle and rotate its second acquisition by 90 degrees."""
    rows, cols = np.indices((9, 13))
    x, y = (cols - 6) / 4, (rows - 4) / 4
    z = x * y + y / 2
    poses = np.stack([np.stack([x, y, z], -1), np.stack([4 - y, x, z + 0.375], -1)])
    colors = np.stack([cols * 16, rows * 28, 32 + rows * 4 + cols], -1)
    colors = np.stack([colors, colors[..., ::-1]]).astype(np.uint8)
    first = (rows * 13 + cols)[:-1, :-1]
    faces = np.stack(
        [
            np.stack([first, first + 1, first + 14], -1),
            np.stack([first, first + 14, first + 13], -1),
        ],
        -2,
    ).reshape(-1, 3)
    summary_v2.mesh(
        "terrain",
        poses.reshape(2, -1, 3).astype(np.float32),
        faces=np.stack([faces, faces]).astype(np.int32),
        colors=colors.reshape(2, -1, 3),
        step=12,
    )


def _scans() -> None:
    """Measure three elevation rings at 16 bearings with distinct ranges and origins."""
    ring, beam = np.indices((3, 16))
    angle = beam * np.pi / 8
    scans = []
    for sweep in (0, 1):
        radius = 2 + ring / 4 + np.cos(2 * angle) / 2 + sweep / 8
        scans.append(
            np.stack(
                [
                    radius * np.cos(angle) + 8 * sweep,
                    radius * np.sin(angle) - sweep / 2,
                    (ring - 1) * radius / 8,
                ],
                -1,
            ).reshape(-1, 3)
        )
    summary_v2.mesh("range-scan", np.asarray(scans, dtype=np.float32), step=12)


def main() -> None:
    """Generate one event file in a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    writer = tf.summary.create_file_writer(str(output))
    with writer.as_default():
        _terrain()
        _scans()
    writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.geometry-scene")


if __name__ == "__main__":
    main()
