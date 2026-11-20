#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
# ///
"""Write TensorFlow plugin summaries, projector assets, and a CPU profile."""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary


def main() -> None:
    """Write plugin summaries and their sidecars into a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    layout = layout_pb2.Layout()
    category = layout.category.add(title="comparisons", closed=True)
    category.chart.add(title="range").margin.series.add(
        value="score", lower="low", upper="high"
    )
    category.chart.add(title="traces").multiline.tag.extend(["train/loss", "eval/loss"])
    domains = [
        hp.HParam("optimizer", hp.Discrete(["adam", "sgd"])),
        hp.HParam("enabled", hp.Discrete([False, True])),
        hp.HParam("rate", hp.RealInterval(0.125, 0.5)),
    ]
    writer = tf.summary.create_file_writer(str(output))
    with writer.as_default(step=2):
        hp.hparams_config(domains, [], time_created_secs=10)
        hp.hparams(
            {"optimizer": "adam", "enabled": False, "rate": 0.25},
            trial_id="trial",
            start_time_secs=11,
        )
        for summary in (
            hp_summary.session_end_pb(api_pb2.STATUS_RUNNING, end_time_secs=12),
            custom_scalar.pb(layout),
        ):
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step=2)
        cells = [["\N{GREEK SMALL LETTER ALPHA}", "x|y"], ["one\ntwo", ""]]
        tf.summary.text("labels", tf.constant(cells))
        metadata = tf.compat.v1.SummaryMetadata()
        metadata.plugin_data.plugin_name = "example_array"
        metadata.plugin_data.content = b"\x00\xffpayload"
        tf.summary.write("points", tf.constant([7, -3]), metadata=metadata)
    writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.plugins")

    assets = output / "assets"
    assets.mkdir()
    (assets / "vectors.tsv").write_text("1\t2\n3\t4\n")
    (assets / "labels.tsv").write_text("label\tid\none\t1\ntwo\t2\n")
    (assets / "views.json").write_text("[]\n")
    sprite = tf.constant([[[255, 0, 0], [0, 255, 0]]], dtype=tf.uint8)
    (assets / "sprite.png").write_bytes(tf.io.encode_png(sprite).numpy())
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add(
        tensor_name="vectors",
        tensor_path="assets/vectors.tsv",
        metadata_path="assets/labels.tsv",
        bookmarks_path="assets/views.json",
    )
    embedding.sprite.image_path = "assets/sprite.png"
    embedding.sprite.single_image_dim.extend([1, 1])
    projector.visualize_embeddings(str(output), config)

    with (
        tf.profiler.experimental.Profile(str(output)),
        tf.profiler.experimental.Trace("plugin_case"),
    ):
        tf.math.square(tf.constant(3)).numpy()
    (profile,) = (output / "plugins/profile").glob("*/*.xplane.pb")
    session = profile.parent
    profile.rename(session / "cpu.xplane.pb")
    session.rename(session.parent / "capture")


if __name__ == "__main__":
    main()
