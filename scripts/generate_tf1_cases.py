#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0",
#   "tensorflow-cpu==2.21.0",
# ]
# ///
"""Generate a compact event file containing legacy TF1 summary values."""

import argparse
from itertools import pairwise
from pathlib import Path
import tempfile

import tensorflow as tf

from tensorflow.compat.v1 import Summary


def _write(output: Path) -> Path:
    """Write old-format summary values and return the final event file."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pixels = tf.constant([[[10], [240]], [[60], [180]]], dtype=tf.uint8)
    first = tf.io.encode_png(pixels).numpy()
    jpeg = tf.io.encode_jpeg(pixels).numpy()
    second = tf.io.encode_png(tf.reverse(pixels, axis=[0])).numpy()
    with tempfile.TemporaryDirectory(dir=output) as temporary, tf.Graph().as_default():
        writer = tf.compat.v1.summary.FileWriter(temporary)
        try:
            for step, tag, blobs in (
                (None, "omitted-step", [first]),
                (7, "same-tag", [first, jpeg, second]),
            ):
                event = tf.compat.v1.Event(wall_time=9.0 if step is None else 10.0)
                if step is not None:
                    event.step = step
                for blob in blobs:
                    image = event.summary.value.add(tag=tag).image
                    image.height = image.width = 2
                    image.colorspace = 1
                    image.encoded_image_string = blob
                writer.add_event(event)
            for values in ([-2.0, 0.0, 2.0], [10.0, 20.0, 30.0]):
                event = tf.compat.v1.Event(wall_time=10.0, step=7)
                event.summary.value.extend([_histogram("histogram", values)])
                writer.add_event(event)
            writer.flush()
        finally:
            writer.close()
        (source,) = Path(temporary).glob("events.out.tfevents.*")
        destination = output / "events.out.tfevents.tf1"
        destination.hardlink_to(source)
    return destination


def _histogram(tag: str, values: list[float]) -> Summary.Value:
    value = Summary.Value(tag=tag)
    value.histo.min = min(values)
    value.histo.max = max(values)
    value.histo.num = len(values)
    value.histo.sum = sum(values)
    value.histo.sum_squares = sum(item * item for item in values)
    limits = [(a + b) / 2 for a, b in pairwise(values)]
    value.histo.bucket_limit.extend([*limits, values[-1] + 1])
    value.histo.bucket.extend([1.0] * len(values))
    return value


def main() -> None:
    """Create an event file in a caller-selected new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output)


if __name__ == "__main__":
    main()
