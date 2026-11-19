#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = [
#   "tensorflow-cpu==1.15.0", "tensorboard==1.15.0", "numpy==1.18.5",
#   "protobuf==3.20.3", "setuptools==67.8.0", "werkzeug==1.0.1",
# ]
# ///
"""Generate genuine TF 1.15 Session summaries from a small signal-analysis graph."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2, summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.mesh import summary as mesh
from tensorboard.plugins.pr_curve import summary as pr


def _plugin_ops() -> tf.Tensor:
    """Add plugin graph ops directly to the v1 session's summary collection."""
    labels = tf.constant([True, False, True, False])
    scores = tf.constant([0.9, 0.7, 0.4, 0.1])
    pr.op(
        "classifier",
        labels,
        scores,
        num_thresholds=5,
        weights=tf.constant([2.0, 1.0, 1.0, 3.0]),
    )
    vertices = tf.constant([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    surface = mesh.op(
        "surface", vertices, faces=tf.constant([[[0, 1, 2]]]), collections=[]
    )
    cloud = mesh.op(
        "cloud",
        vertices,
        colors=tf.constant([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]),
        collections=[],
    )
    layout = layout_pb2.Layout()
    chart = layout.category.add(title="signal").chart.add(title="power")
    chart.multiline.tag.append("mean_square")
    custom_scalar.op(layout)
    return tf.summary.merge([tf.summary.merge_all(), surface, cloud])


def _hparams(output: Path) -> None:
    """Execute the HParams API with the TF 1.15 v2 writer in graph mode."""
    with tf.Graph().as_default(), tf.Session() as session:
        writer = tf.compat.v2.summary.create_file_writer(
            str(output), filename_suffix=".hparams"
        )
        rate = hp.HParam("rate", hp.Discrete([0.25, 0.5]))
        with writer.as_default():
            experiment = hp.hparams_config(
                [rate], [hp.Metric("mean_square")], time_created_secs=1
            )
            trial = hp.hparams({rate: 0.5}, trial_id="calibration", start_time_secs=2)
        session.run(writer.init())
        np.testing.assert_array_equal(session.run([experiment, trial]), [True, True])
        session.run(writer.close())
    (event,) = output.glob("*.hparams")
    event.rename(output / "events.out.tfevents.tf115-native-hparams")


def main() -> None:
    """Write session outputs and embedding sidecars into an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    y, x = np.indices((24, 32))
    pixels = np.stack([255 - 8 * x, 8 * x, 192 * ((x // 8 + y // 6) % 2)], -1)
    pixels = pixels.astype(np.uint8)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2000) / 8000)
    with tf.Graph().as_default(), tf.Session() as session:
        samples = tf.placeholder(tf.float32, shape=[4], name="samples")
        picture = tf.placeholder(tf.uint8, shape=[1, 24, 32, 3], name="picture")
        sound = tf.placeholder(tf.float32, shape=[1, 2000, 1], name="sound")
        energy = tf.square(samples, name="squared")
        tf.summary.scalar("mean_square", tf.reduce_mean(energy))
        tf.summary.histogram("weights", samples)
        tf.summary.image("checker", picture, max_outputs=1)
        tf.summary.audio("concert_a", sound, 8000, max_outputs=1)
        tf.summary.text("channels", tf.constant(["left", "right"]))
        tf.summary.tensor_summary("energy", energy)
        merged = _plugin_ops()
        result = session.run(
            merged,
            {
                samples: [-3.0, -1.0, 1.0, 3.0],
                picture: pixels[None],
                sound: tone.reshape(1, -1, 1),
            },
        )
        metadata = tf.RunMetadata()
        session.run(
            energy,
            {samples: [3.0, 1.0, -1.0, -3.0]},
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=metadata,
        )
        with tf.summary.FileWriter(str(output), session.graph) as writer:
            writer.add_summary(result, global_step=12)
            writer.add_run_metadata(metadata, "signal", global_step=12)
            (output / "points.tsv").write_text("0\t0\n2\t0\n0\t1\n")
            (output / "labels.tsv").write_text("origin\nright\nup\n")
            config = projector.ProjectorConfig()
            config.embeddings.add(
                tensor_name="corners",
                tensor_path="points.tsv",
                metadata_path="labels.tsv",
            )
            projector.visualize_embeddings(writer, config)
        (encoded,) = (
            v.image.encoded_image_string
            for v in tf.Summary.FromString(result).value
            if v.HasField("image")
        )
        np.testing.assert_array_equal(session.run(tf.image.decode_png(encoded)), pixels)
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.tf115-native")
    _hparams(output)


if __name__ == "__main__":
    main()
