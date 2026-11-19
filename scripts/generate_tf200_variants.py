#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = [
#   "tensorflow==2.0.0", "tensorboard==2.0.0", "numpy==1.18.5",
#   "protobuf==3.20.3", "setuptools==67.8.0", "werkzeug==1.0.1",
# ]
# ///
"""Generate the native TF/TB 2.0 historical variant corpus and tools."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from historical_variants import main
from tensorboard.plugins.beholder import beholder
from tensorboard.plugins.beholder.file_system_tools import (
    read_pickle,
    read_tensor_summary,
)
from tensorflow.python.ops.math_ops import square


def _trace(run: Path) -> None:
    """Export a graph captured while executing a TensorFlow function eagerly."""
    function = tf.function(square, autograph=False)
    writer = tf.summary.create_file_writer(str(run), filename_suffix=".function")
    with writer.as_default():
        tf.summary.trace_on(graph=True, profiler=False)
        np.testing.assert_array_equal(
            function(tf.constant([3.0, -2.0])).numpy(), [9, 4]
        )
        tf.summary.trace_export(name="squared_function", step=7)
    writer.close()
    (event,) = run.glob("events.out.tfevents.*.function")
    (trace,) = (
        value
        for record in tf.compat.v1.train.summary_iterator(str(event))
        for value in record.summary.value
        if value.tag == "squared_function"
    )
    metadata = tf.compat.v1.RunMetadata.FromString(tf.make_ndarray(trace.tensor).item())
    if not any(
        node.op == "Square"
        for fn in metadata.function_graphs
        for node in fn.pre_optimization_graph.node
    ):
        raise ValueError("The function trace did not record a Square op")
    event.rename(run / "events.out.tfevents.function")


def _beholder(run: Path) -> None:
    """Capture current arrays with the native graph-mode Beholder plugin."""
    with tf.Graph().as_default(), tf.compat.v1.Session() as session:
        instance = beholder.Beholder(str(run))
        config = dict(
            beholder.DEFAULT_CONFIG,
            values="arrays",
            mode="current",
            colormap="grayscale",
        )
        directory = run / "plugins" / "beholder"
        config_path = directory / "config.pkl"
        beholder.write_pickle(config, str(config_path))
        instance.previous_config = config
        instance.config_last_modified_time = config_path.stat().st_mtime
        instance.last_update_time = 0
        instance.update(
            session,
            arrays=[
                np.array([[-3, 0, 3], [1, -1, 2]], np.float32),
                np.arange(8, dtype=np.float32),
            ],
        )
        sections = read_pickle(str(directory / "section-info.pkl"))
        if [section["shape"] for section in sections] != ["(2, 3)", "(8,)"]:
            raise ValueError("Beholder did not record both arrays")
        frame = read_tensor_summary(str(directory / "frame.summary"))
        if not frame.size:
            raise ValueError("Beholder wrote an empty frame")


if __name__ == "__main__":
    directory = main()
    _trace(directory)
    _beholder(directory)
