#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0",
#   "tensorflow>=2.21.0,<3.0",
# ]
# ///
"""Generate TF2 graph summaries under test-logs/graphs, with TensorBoard's graphs.

The run holds Keras model configs (graph_keras_model) and traced functions
(graph_run_metadata_graph). tests/golden/graphs/<tag>.pb is the GraphDef that
the TensorBoard graph plugin builds for a summary, or <tag>.error names the
exception when TensorBoard cannot build one.
"""

from collections.abc import Callable, Iterator
import json
from pathlib import Path
import shutil
from typing import Any

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import config_pb2, summary_pb2
from tensorboard.plugins.graph import graph_util, keras_util
import tensorflow as tf

ROOT = Path("test-logs") / "graphs"
GOLDEN = Path("tests") / "golden" / "graphs"
keras = tf.keras
layers = keras.layers


def main() -> None:
    """Write the run and the expected graphs, replacing earlier ones."""
    for directory in (ROOT, GOLDEN):
        if directory.exists():
            shutil.rmtree(directory)
    GOLDEN.mkdir(parents=True)
    tf.keras.utils.set_random_seed(0)

    def inner(values: tf.Tensor) -> tf.Tensor:
        """Return a value of a traced function that another one calls."""
        return tf.sin(values) * 2.0

    def outer(values: tf.Tensor) -> tf.Tensor:
        """Return a value of a traced function that calls inner."""
        return traced_inner(values) + 1.0

    traced_inner, traced_outer = tf.function(inner), tf.function(outer)
    traces: dict[str, Callable[[], object]] = {
        "square": lambda: tf.function(lambda values: values * values)(
            tf.constant([1.0, 2.0])
        ),
        # Two traced functions give a RunMetadata with two function graphs.
        "calls": lambda: (
            traced_inner(tf.constant([1.0])),
            traced_outer(tf.constant([3.0])),
        ),
    }
    metadata = summary_pb2.SummaryMetadata()
    metadata.plugin_data.plugin_name = "graph_keras_model"
    metadata.plugin_data.content = b"1"
    writer = tf.summary.create_file_writer(str(ROOT))
    with writer.as_default():
        for tag, model in _models():
            tf.summary.write(
                tag,
                tf.constant([model.to_json()]),
                step=0,
                metadata=metadata.SerializeToString(),
            )
        for step, (tag, call) in enumerate(traces.items()):
            tf.summary.trace_on(graph=True)
            call()
            tf.summary.trace_export(tag, step=step)
    writer.close()
    for path in ROOT.glob("events.out.tfevents.*"):
        _write_golden(path)


def _models() -> Iterator[tuple[str, Any]]:
    yield (
        "sequential",
        keras.Sequential([keras.Input((2,)), layers.Dense(3), layers.Dense(1)]),
    )
    x = keras.Input((4,), name="x")
    yield (
        "functional",
        keras.Model(x, layers.Dense(2, name="out")(layers.Dense(3)(x)), name="func"),
    )
    a, b = keras.Input((2,), name="a"), keras.Input((3,), name="b")
    hidden = layers.Concatenate()([layers.Dense(4)(a), b])
    outputs = [layers.Dense(1, name="y1")(hidden), layers.Dense(2, name="y2")(hidden)]
    yield "multi_io", keras.Model([a, b], outputs, name="multi")
    inner = keras.Sequential([layers.Dense(3), layers.Dense(3)], name="inner")
    x = keras.Input((3,), name="x")
    yield "nested_sequential", keras.Model(x, layers.Dense(1)(inner(x)), name="outer")
    fi = keras.Input((3,), name="fi")
    finner = keras.Model(fi, layers.Dense(2, name="fd")(fi), name="finner")
    y = keras.Input((3,), name="y")
    shared = layers.Dense(3, name="shared")
    yield "nested_functional", keras.Model(y, finner(shared(shared(y))), name="fouter")
    ma, mb = keras.Input((3,), name="ma"), keras.Input((3,), name="mb")
    two_outputs = [
        layers.Add(name="sum")([ma, mb]),
        layers.Subtract(name="diff")([ma, mb]),
    ]
    two = keras.Model([ma, mb], two_outputs, name="two")
    z1, z2 = keras.Input((3,), name="z1"), keras.Input((3,), name="z2")
    total, difference = two([z1, z2])
    yield (
        "nested_multi",
        keras.Model([z1, z2], layers.Multiply()([total, difference]), name="mouter"),
    )
    sub = keras.Sequential([layers.Dense(2)], name="sub")
    yield (
        "seq_in_seq",
        keras.Sequential([keras.Input((2,)), sub, layers.Dense(1)], name="top"),
    )
    keras.mixed_precision.set_global_policy("mixed_float16")
    activation = layers.Activation("relu", dtype="float32")
    yield (
        "mixed",
        keras.Sequential([keras.Input((2,)), layers.Dense(2), activation], name="mp"),
    )
    keras.mixed_precision.set_global_policy("float32")
    # TensorBoard cannot draw a functional model inside a Sequential model.
    yield "func_in_seq", keras.Sequential([keras.Input((3,)), finner], name="fs")
    embedding = [
        keras.Input((5,), dtype="int64"),
        layers.Embedding(10, 2),
        layers.Flatten(),
    ]
    yield "int_input", keras.Sequential(embedding, name="emb")


def _write_golden(path: Path) -> None:
    """Write the graph of each summary as the TensorBoard graph plugin builds it."""
    for event in event_file_loader.EventFileLoader(str(path)).Load():
        for value in event.summary.value:
            blob = value.tensor.string_val[0]
            try:
                if value.metadata.plugin_data.plugin_name == "graph_keras_model":
                    graph = keras_util.keras_model_to_graph_def(json.loads(blob))
                else:
                    run_metadata = config_pb2.RunMetadata.FromString(blob)
                    graph = graph_util.merge_graph_defs(
                        [f.pre_optimization_graph for f in run_metadata.function_graphs]
                    )
            except Exception as error:  # noqa: BLE001 - the golden records any failure
                (GOLDEN / f"{value.tag}.error").write_text(f"{type(error).__name__}\n")
                continue
            (GOLDEN / f"{value.tag}.pb").write_bytes(
                graph.SerializeToString(deterministic=True)
            )


if __name__ == "__main__":
    main()
