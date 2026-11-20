#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = [
#   "tensorflow==2.0.0", "tensorboard==2.0.0", "numpy==1.18.5",
#   "protobuf==3.20.3", "setuptools==67.8.0", "werkzeug==1.0.1",
# ]
# ///
"""Generate native eager summaries using the historical TensorFlow 2.0 CPU wheel."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.mesh import summary as mesh
from tensorboard.plugins.pr_curve import summary as pr
from tensorboard.plugins.profile import trace_events_pb2


def _plugins(output: Path) -> None:
    """Write eager plugin summaries, a projector config and its assets."""
    vertices = tf.constant([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    mesh.mesh("surface", vertices, faces=tf.constant([[[0, 1, 2]]]), step=14)
    mesh.mesh(
        "cloud",
        vertices,
        colors=tf.constant([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]),
        step=14,
    )
    labels = tf.constant([True, False, True, False])
    scores = tf.constant([0.9, 0.7, 0.4, 0.1])
    classifier = pr.op(
        "classifier",
        labels,
        scores,
        num_thresholds=5,
        weights=tf.constant([2.0, 1.0, 1.0, 3.0]),
    )
    tf.summary.experimental.write_raw_pb(classifier, step=14)
    layout = layout_pb2.Layout()
    layout.category.add(title="signal").chart.add(title="power").multiline.tag.append(
        "mean_square"
    )
    tf.summary.experimental.write_raw_pb(custom_scalar.op(layout), step=14)
    rate = hp.HParam("rate", hp.Discrete([0.25, 0.5]))
    hp.hparams_config([rate], [hp.Metric("mean_square")], time_created_secs=1)
    hp.hparams({rate: 0.5}, trial_id="calibration", start_time_secs=2)

    (output / "points.tsv").write_text("0\t0\n2\t0\n0\t1\n")
    (output / "labels.tsv").write_text("origin\nright\nup\n")
    config = projector.ProjectorConfig()
    config.embeddings.add(
        tensor_name="corners", tensor_path="points.tsv", metadata_path="labels.tsv"
    )
    projector.visualize_embeddings(str(output), config)


def _profile(output: Path) -> None:
    """Capture a small CPU trace around actual eager signal operations."""
    samples = tf.constant([-3.0, -1.0, 1.0, 3.0])
    tf.summary.trace_on(graph=True, profiler=True)
    energy = tf.reduce_sum(tf.square(samples)).numpy()
    np.testing.assert_equal(energy, 20.0)
    tf.summary.trace_export("signal", step=14, profiler_outdir=str(output))
    (capture,) = (output / "plugins/profile").glob("*/local.trace")
    trace = trace_events_pb2.Trace.FromString(capture.read_bytes())
    timed = {
        event.name.strip() for event in trace.trace_events if event.duration_ps > 0
    }
    if not {"Square", "Sum"} <= timed:
        raise ValueError("the native CPU trace did not record both signal operations")
    capture.parent.rename(capture.parent.parent / "signal")
    (marker,) = output.glob("*.profile-empty")
    marker.rename(output / "events.out.tfevents.tf200-native.profile-empty")


def main() -> None:
    """Write eager summaries and their projector and profiler sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((24, 32))
    first = np.stack([255 * (x >= 16), 160 * (y >= 12), 255 * (x < 16)], -1)
    pictures = np.stack([first, first[::-1]]).astype(np.uint8)
    tone = 0.5 * np.sin(2 * np.pi * 330 * np.arange(2000) / 8000)
    writer = tf.summary.create_file_writer(str(output))
    try:
        with writer.as_default():
            samples = tf.constant([-3.0, -1.0, 1.0, 3.0])
            tf.summary.scalar(
                "mean_square", tf.reduce_mean(tf.square(samples)), step=14
            )
            tf.summary.histogram("weights", samples, step=14, buckets=2)
            tf.summary.image("quadrants", pictures, step=14, max_outputs=2)
            tf.summary.audio(
                "tone", tone.astype(np.float32).reshape(1, -1, 1), 8000, step=14
            )
            tf.summary.text("channels", tf.constant(["left", "right"]), step=14)
            tf.summary.write("energy", tf.square(samples), step=14)
            _plugins(output)
            _profile(output)
    finally:
        writer.close()
    (event,) = output.glob("events.out.tfevents.*.v2")
    values = (
        value
        for record in tf.compat.v1.train.summary_iterator(str(event))
        for value in record.summary.value
    )
    (quadrants,) = (value for value in values if value.tag == "quadrants")
    images = quadrants.tensor.string_val[2:]
    np.testing.assert_equal(len(images), len(pictures))
    for index, encoded in enumerate(images):
        np.testing.assert_array_equal(
            tf.image.decode_png(encoded).numpy(), pictures[index]
        )
    event.rename(output / "events.out.tfevents.tf200-native")


if __name__ == "__main__":
    main()
