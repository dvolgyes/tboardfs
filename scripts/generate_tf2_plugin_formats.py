#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
# ///
"""Generate native TensorBoard PR-curve and distribution summaries with TF2."""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.pr_curve import summary as pr_curve


def main() -> None:
    """Write real plugin summaries into a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(output))
    try:
        with writer.as_default():
            labels = tf.constant([True, False, True, False, True, False])
            predictions = tf.constant([0.95, 0.85, 0.65, 0.45, 0.25, 0.05])
            tf.summary.experimental.write_raw_pb(
                pr_curve.op("native_pr", labels, predictions, num_thresholds=5), step=4
            )
            weights = tf.constant([2.0, 0.5, 1.5, 3.0, 0.25, 1.25])
            tf.summary.experimental.write_raw_pb(
                pr_curve.op(
                    "weighted_pr",
                    labels,
                    predictions,
                    weights=weights,
                    num_thresholds=5,
                ),
                step=5,
            )
            raw = pr_curve.raw_data_op(
                "raw_pr",
                *map(
                    tf.constant,
                    (
                        [4.0, 3.0, 1.0],
                        [3.0, 1.0, 0.0],
                        [0.0, 2.0, 3.0],
                        [0.0, 1.0, 3.0],
                        [4 / 7, 0.75, 1.0],
                        [1.0, 0.75, 0.25],
                    ),
                ),
                num_thresholds=3,
            )
            tf.summary.experimental.write_raw_pb(raw, step=6)
            tf.summary.histogram(
                "distribution",
                tf.constant([-4.0, -2.0, 0.0, 0.5, 1.5, 4.0]),
                step=7,
                buckets=2,
            )
            tf.summary.write(
                "words", tf.constant([["café", ""], ["南", "a\nb"]]), step=8
            )
            tf.summary.write("bytes", tf.constant([b"a\x00b", b"\xff\xfe"]), step=9)
            handle = tf.raw_ops.VarHandleOp(
                dtype=tf.float32, shape=[2], shared_name="plugin_weights"
            )
            tf.summary.write("handle", handle, step=10)
            tensor_list = tf.raw_ops.TensorListFromTensor(
                tensor=tf.constant([[7, -2], [8, 11]]), element_shape=[2]
            )
            tf.summary.write("list", tensor_list, step=11)
    finally:
        writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.tf2-plugin-formats")


if __name__ == "__main__":
    main()
