#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorflow-cpu==2.21.0",
#   "tensorboard==2.21.0",
# ]
# ///
"""Generate high-level TF1 graph-mode writer examples in a new corpus directory."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.mesh import summary as mesh
from tensorboard.plugins.pr_curve import summary as pr_curve


def _trace(output: Path) -> tf.compat.v1.RunMetadata:
    """Trace a real CPU session; keep one native tfdbg-v1 DebugIdentity tensor."""
    v1 = tf.compat.v1
    with tf.Graph().as_default() as trace_graph:
        vector = v1.placeholder(tf.float32, [3], name="vector")
        squared = tf.square(vector, name="squared")
        total = tf.reduce_sum(squared, name="energy")
        metadata = v1.RunMetadata()
        options = v1.RunOptions(trace_level=v1.RunOptions.FULL_TRACE)
        with v1.Session(graph=trace_graph) as session:
            feed = {vector: [1.0, -2.0, 3.0]}
            if session.run(total, feed, options=options, run_metadata=metadata) != 14:
                raise ValueError("unexpected traced vector energy")
            with tempfile.TemporaryDirectory() as temporary:
                debug_options = v1.RunOptions()
                debug_options.debug_options.debug_tensor_watch_opts.add(
                    node_name="squared",
                    output_slot=0,
                    debug_ops=["DebugIdentity"],
                    debug_urls=[Path(temporary).as_uri()],
                )
                session.run(total, feed, options=debug_options)
                (dump,) = Path(temporary).rglob("squared_0_DebugIdentity_*")
                directory = output / "tfdbg"
                directory.mkdir()
                (directory / "squared.pb").write_bytes(dump.read_bytes())
        profile = v1.profiler.ProfileOptionBuilder
        options = (
            profile(profile.time_and_memory())
            .with_timeline_output(str(output / "timeline"))
            .build()
        )
        v1.profiler.profile(
            trace_graph, run_meta=metadata, cmd="scope", options=options
        )
        return metadata


def _write(output: Path) -> Path:
    """Run graph-mode compat.v1 summaries and return their native event file."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    v1 = tf.compat.v1
    graph = tf.Graph()
    with graph.as_default():
        scalar = v1.summary.scalar("loss", tf.constant(0.25))
        histogram = v1.summary.histogram("weights", tf.constant([-2.0, 0.0, 2.0, 4.0]))
        pixels = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        pixels[0, :, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
        pixels[0, :, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 8
        image = v1.summary.image("gradient", tf.constant(pixels), max_outputs=1)
        samples = (
            np.sin(np.arange(1600, dtype=np.float32) * (2 * np.pi * 440 / 8000)) * 0.6
        )
        audio = v1.summary.audio(
            "tone", tf.constant(samples.reshape(1, -1, 1)), 8000, max_outputs=1
        )
        text = v1.summary.text("note", tf.constant(["graph-mode text", "second row"]))
        numeric = v1.summary.tensor_summary(
            "numeric", tf.constant([[1.0, 2.0], [3.0, 4.0]])
        )
        strings = v1.summary.tensor_summary("words", tf.constant(["red", "blue"]))
        curve = pr_curve.op(
            "binary",
            tf.constant([True, False, True, False]),
            tf.constant([0.9, 0.7, 0.6, 0.1]),
            num_thresholds=5,
        )
        vertices = tf.constant([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        triangle = mesh.op("triangle", vertices, faces=tf.constant([[[0, 1, 2]]]))
        cloud = mesh.op(
            "cloud",
            vertices,
            colors=tf.constant([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]),
        )
        layout = layout_pb2.Layout()
        chart = layout.category.add(title="training").chart.add(title="objective")
        chart.multiline.tag.append("loss")
        dashboard = custom_scalar.op(layout)
        merged = v1.summary.merge(
            [
                scalar,
                histogram,
                image,
                audio,
                text,
                numeric,
                strings,
                curve,
                triangle,
                cloud,
                dashboard,
            ]
        )
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        with v1.Session(graph=graph) as session:
            summary = session.run(merged)
        values = v1.Summary.FromString(summary).value
        png = next(v.image.encoded_image_string for v in values if v.HasField("image"))
        np.testing.assert_array_equal(tf.io.decode_png(png).numpy(), pixels[0])
        metadata = _trace(output)
        with graph.as_default():
            writer = v1.summary.FileWriter(temporary)
            try:
                writer.add_summary(summary, global_step=3)
                writer.add_run_metadata(metadata, "energy", global_step=3)
                rate = hp.HParam("rate", hp.Discrete([0.25, 0.5]))
                experiment = hp.hparams_config_pb(
                    [rate], [hp.Metric("loss")], time_created_secs=1
                )
                trial = hp.hparams_pb({rate: 0.25}, trial_id="trial", start_time_secs=2)
                writer.add_summary(experiment.SerializeToString())
                writer.add_summary(trial.SerializeToString())
                writer.flush()
            finally:
                writer.close()
        (source,) = Path(temporary).glob("events.out.tfevents.*")
        destination = output / "events.out.tfevents.tf1-writer"
        destination.hardlink_to(source)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add(
        tensor_name="triangle", tensor_path="vectors.tsv", metadata_path="labels.tsv"
    )
    (output / embedding.tensor_path).write_text("0\t0\t0\n1\t0\t0\n0\t1\t0\n")
    (output / embedding.metadata_path).write_text("origin\nright\nup\n")
    projector.visualize_embeddings(str(output), config)
    return destination


def main() -> None:
    """Generate the event file in a caller-selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output)


if __name__ == "__main__":
    main()
