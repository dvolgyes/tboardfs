#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
# ///
"""Generate native TensorFlow 2.21 graph, eager, and live debugger fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hp_summary
from tensorboard.plugins.mesh import summary as mesh
from tensorboard.plugins.pr_curve import summary as pr_curve
from tensorflow.python.debug.lib import debug_events_reader

TABLE = (("north", "east"), ("south", "west"))
PLANES = (TABLE, (("silver", "gold"), ("green", "blue")))
WIDE = (1 + 2**-48, -2.5)
COMPLEX = (2 - 3j, -1 + 0.5j)


def _graph(run: Path) -> None:
    """Write native Session summaries and a discoverable projector checkpoint."""
    v1 = tf.compat.v1
    checkpoint_name = "positions"
    with tf.Graph().as_default() as graph:
        weighted = pr_curve.op(
            "weighted",
            tf.constant([True, False, True, False]),
            tf.constant([1.0, 0.6, 0.4, 0.2]),
            num_thresholds=3,
            weights=tf.constant([2.0, 3.0, 5.0, 7.0]),
        )
        prepared = pr_curve.raw_data_op(
            "prepared",
            [4, 3, 1],
            [4, 1, 0],
            [0, 3, 4],
            [0, 1, 3],
            [0.5, 0.75, 1.0],
            [1.0, 0.75, 0.25],
            num_thresholds=3,
        )
        vertices = tf.constant(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 0.0, 1.0]],
            ]
        )
        faces = tf.constant([[[0, 1, 2]], [[0, 2, 1]]])
        colors = tf.constant(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
            ]
        )
        layout = layout_pb2.Layout()
        category = layout.category.add(title="measurements")
        category.chart.add(title="bounds").margin.series.add(
            value="score", lower="minimum", upper="maximum"
        )
        merged = v1.summary.merge(
            [
                weighted,
                prepared,
                mesh.op("surface-rgb", vertices, faces=faces, colors=colors),
                mesh.op("surface", vertices, faces=faces),
                mesh.op("cloud-rgb", vertices, colors=colors),
                mesh.op("cloud", vertices),
                v1.summary.text("directions", tf.constant(TABLE)),
                v1.summary.text("planes", tf.constant(PLANES)),
                v1.summary.tensor_summary("wide", tf.constant(WIDE, tf.float64)),
                v1.summary.tensor_summary(
                    "complex", tf.constant(COMPLEX, tf.complex64)
                ),
                v1.summary.tensor_summary(
                    "binary", tf.constant([b"a\0b", b"\xff\x80"])
                ),
                v1.summary.scalar("score", tf.constant(2.0)),
                v1.summary.scalar("minimum", tf.constant(1.0)),
                v1.summary.scalar("maximum", tf.constant(4.0)),
                custom_scalar.op(layout),
            ]
        )
        vectors = tf.Variable([[1.0, -2.0], [3.0, 4.5]], name="vectors")
        saver = v1.train.Saver({"vectors": vectors}, save_relative_paths=True)
        sprite = tf.io.encode_png(tf.constant([[[255, 0, 0], [0, 255, 0]]], tf.uint8))
        decoded_sprite = tf.io.decode_png(sprite)
        with v1.Session(graph=graph) as session:
            session.run(v1.global_variables_initializer())
            summary, image, pixels = session.run([merged, sprite, decoded_sprite])
            saver.save(session, str(run / checkpoint_name), write_meta_graph=False)
        checkpoint = tf.train.latest_checkpoint(str(run))
        if checkpoint is None:
            raise ValueError("projector checkpoint was not discoverable")
        np.testing.assert_array_equal(
            tf.train.load_checkpoint(checkpoint).get_tensor("vectors"),
            [[1.0, -2.0], [3.0, 4.5]],
        )
        np.testing.assert_array_equal(pixels, [[[255, 0, 0], [0, 255, 0]]])
        bookmark = {
            "label": "First point",
            "dataSetDimensions": [2, 2],
            "selectedProjection": "pca",
            "pcaComponentDimensions": [0, 1],
            "projections": [
                {"pca-0": -3.4, "pca-1": 0.0},
                {"pca-0": 3.4, "pca-1": 0.0},
            ],
            "selectedPoints": [0],
            "tSNEIteration": 0,
            "tSNEPerplexity": 0,
            "tSNELearningRate": 0,
            "tSNEis3d": True,
            "umapIs3d": True,
            "umapNeighbors": 15,
            "umapMinDist": 0.1,
        }
        (run / "bookmarks.json").write_text(
            json.dumps([bookmark], separators=(",", ":")) + "\n"
        )
        config = projector.ProjectorConfig()
        (run / "labels.tsv").write_text("red\ngreen\n")
        (run / "sprite.png").write_bytes(image)
        embedding = config.embeddings.add(
            tensor_name="vectors",
            metadata_path="labels.tsv",
            bookmarks_path="bookmarks.json",
        )
        embedding.sprite.image_path = "sprite.png"
        embedding.sprite.single_image_dim.extend([1, 1])
        domains = [
            hp.HParam("mode", hp.Discrete(["fast", "accurate"])),
            hp.HParam("enabled", hp.Discrete([False, True])),
            hp.HParam("rate", hp.RealInterval(0.125, 0.5)),
        ]
        with v1.summary.FileWriter(str(run)) as writer:
            writer.add_summary(summary, global_step=7)
            for proto in (
                hp.hparams_config_pb(
                    domains, [hp.Metric("score")], time_created_secs=30
                ),
                hp.hparams_pb(
                    {"mode": "fast", "enabled": True, "rate": 0.25},
                    trial_id="bounds",
                    start_time_secs=31,
                ),
                hp_summary.session_end_pb(api_pb2.STATUS_SUCCESS, end_time_secs=32),
            ):
                writer.add_summary(proto.SerializeToString(), global_step=7)
            projector.visualize_embeddings(writer, config)
    (event,) = run.glob("events.out.tfevents.*")
    event.rename(run / "events.out.tfevents.graph-extra")


def _eager(run: Path) -> None:
    """Write eager summaries and retain the debugger execution and device records."""
    writer = tf.summary.create_file_writer(str(run))
    with writer.as_default(step=7):
        tf.summary.text("planes", tf.constant(PLANES))
        tf.summary.write("wide", tf.constant(WIDE, tf.float64))
        tf.summary.write("complex", tf.constant(COMPLEX, tf.complex64))
    writer.close()
    (event,) = run.glob("events.out.tfevents.*")
    event.rename(run / "events.out.tfevents.eager-extra")

    dump = run / "tfdbg"
    with tf.device("/CPU:0"):
        vector = tf.constant([-3.0, 2.0])
        debug = tf.debugging.experimental.enable_dump_debug_info(
            str(dump),
            tensor_debug_mode="FULL_TENSOR",
            circular_buffer_size=-1,
            op_regex="^Square$",
        )
        try:
            squared = tf.square(vector)
        finally:
            tf.debugging.experimental.disable_dump_debug_info()
            debug.Close()
    np.testing.assert_array_equal(squared.numpy(), [9.0, 4.0])
    reader = debug_events_reader.DebugEventsReader(str(dump))
    try:
        (record,) = reader.execution_iterator()
        execution = record.debug_event.execution
        (tensor,) = execution.tensor_protos
        if execution.op_type != "Square":
            raise ValueError(f"unexpected recorded operation: {execution.op_type}")
        np.testing.assert_array_equal(tf.make_ndarray(tensor), [9.0, 4.0])
        (device,) = reader.graphs_iterator()
        if not device.debug_event.debugged_device.device_name.endswith("/device:CPU:0"):
            raise ValueError("debugger did not record the CPU device")
    finally:
        reader.close()
    for extension in ("metadata", "graphs", "execution"):
        (stream,) = dump.glob(f"*.{extension}")
        stream.rename(dump / f"tfdbg_events.capture.{extension}")
    for extension in ("source_files", "stack_frames", "graph_execution_traces"):
        (stream,) = dump.glob(f"*.{extension}")
        stream.unlink()


def main() -> None:
    """Write graph and eager outputs into a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    graph = output / "graph"
    graph.mkdir(parents=True)
    _graph(graph)
    _eager(output / "eager")


if __name__ == "__main__":
    main()
