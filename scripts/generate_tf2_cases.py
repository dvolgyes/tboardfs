#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
# ///
"""Generate a compact TF2 event file with batched media and exact integer steps."""

import argparse
from pathlib import Path
import tempfile

import tensorflow as tf


def _write(output: Path) -> Path:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        writer = tf.summary.create_file_writer(temporary)
        try:
            with writer.as_default():
                pixels = tf.reshape(tf.range(8, dtype=tf.int32) * 25, [2, 2, 2, 1])
                pixels = tf.cast(pixels, tf.uint8)
                tf.summary.image("tiles", pixels, step=2, max_outputs=2)
                tf.summary.write(
                    "tiles", tf.stack([b"2", b"2", tf.io.encode_png(pixels[1])]), step=3
                )
                samples = tf.constant(
                    [[[-0.5], [0.25], [0.5]], [[0.0], [-0.25], [0.75]]]
                )
                tf.summary.audio("tone", samples, 8000, step=2, max_outputs=2)
                tf.summary.histogram(
                    "spread", [0.0, 0.25, 1.75, 2.0], step=2, buckets=2
                )
                big = 2**53
                tf.summary.scalar("large-step", 3.5, step=big + 1)
                tf.summary.scalar("large-step", -1.25, step=big)
                matrix = tf.constant(
                    [[big + 1, -7, 0], [12, big - 1, 4]], dtype=tf.int64
                )
                tf.summary.write("matrix", matrix, step=big + 1)
            writer.flush()
        finally:
            writer.close()
        (source,) = Path(temporary).glob("events.out.tfevents.*")
        destination = output / "events.out.tfevents.tf2"
        destination.hardlink_to(source)
    return destination


def main() -> None:
    """Write an event file in a caller-selected new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output)


if __name__ == "__main__":
    main()
