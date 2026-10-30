#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0",
#   "tensorflow>=2.21.0,<3.0",
# ]
# ///
"""Generate meta graphs and run metadata under test-logs/tf1_graphs, with TensorBoard's views.

Runs:
    meta_graph: two meta graphs and two tagged run metadata (TF1 FileWriter).
    graph_first: a graph event, then a meta graph with another graph.
    no_graph: a meta graph without a graph.
    profile: a TF2 run metadata summary with profile data (graph_run_metadata).

tests/golden/tf1_graphs/<run>/graph.pb and meta_graph.pb are the graph and the
meta graph that the TensorBoard 2.21 event_accumulator keeps for a run.
<tag>.<step>.pbtxt is the profile text that the TensorBoard graph plugin
returns for a run metadata summary, after dataclass_compat migrates a tagged
run metadata to a summary, and <tag>.<step>.pb is the op graph that it draws
for a graph_run_metadata summary.
"""

from pathlib import Path
import shutil
import time

from tensorboard.backend.event_processing import (
    event_accumulator,
    event_file_loader,
)
from tensorboard import dataclass_compat
from tensorboard.compat.proto import config_pb2
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import metadata as graph_metadata
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

ROOT = Path("test-logs") / "tf1_graphs"
GOLDEN = Path("tests") / "golden" / "tf1_graphs"
v1 = tf.compat.v1


def main() -> None:
    """Write the runs and the expected views, replacing earlier ones."""
    for directory in (ROOT, GOLDEN):
        if directory.exists():
            shutil.rmtree(directory)
    small, small_meta, small_metadata = _traced_graph(1)
    large, large_meta, large_metadata = _traced_graph(3)

    # TF1 writers need a graph context.
    with tf.Graph().as_default():
        writer = v1.summary.FileWriter(str(ROOT / "meta_graph"))
        writer.add_meta_graph(small_meta, global_step=1)
        writer.add_run_metadata(small_metadata, "train", global_step=1)
        writer.add_meta_graph(large_meta, global_step=2)
        # FileWriter refuses a repeated tag, which a restarted writer can write.
        repeated = v1.Event(step=2, wall_time=time.time())
        repeated.tagged_run_metadata.tag = "train"
        repeated.tagged_run_metadata.run_metadata = large_metadata.SerializeToString()
        writer.add_event(repeated)
        writer.add_run_metadata(small_metadata, "eval", global_step=2)
        writer.close()

        writer = v1.summary.FileWriter(str(ROOT / "graph_first"), graph=small)
        writer.add_meta_graph(large_meta, global_step=1)
        writer.close()

        writer = v1.summary.FileWriter(str(ROOT / "no_graph"))
        no_graph = v1.MetaGraphDef()
        no_graph.meta_info_def.tags.append("serve")
        writer.add_meta_graph(no_graph, global_step=1)
        writer.close()

    with tf.summary.create_file_writer(str(ROOT / "profile")).as_default():
        summary_ops_v2.run_metadata("traced", large_metadata, step=4)
    del large

    for run in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        _write_golden(run)


def _traced_graph(
    layers: int,
) -> tuple[tf.Graph, tf.compat.v1.MetaGraphDef, tf.compat.v1.RunMetadata]:
    """Return a TF1 graph, its meta graph and the RunMetadata of a traced run."""
    graph = tf.Graph()
    with graph.as_default():
        values = v1.placeholder(tf.float32, [None, 2], name="x")
        for layer in range(layers):
            values = tf.nn.relu(
                values @ tf.constant([[1.0, -1.0], [0.5, 2.0]]), name=f"relu_{layer}"
            )
        meta = v1.train.export_meta_graph(graph=graph)
        options = v1.RunOptions(
            trace_level=v1.RunOptions.FULL_TRACE, output_partition_graphs=True
        )
        metadata = v1.RunMetadata()
        with v1.Session(graph=graph) as session:
            session.run(
                values,
                feed_dict={"x:0": [[1.0, 2.0]]},
                options=options,
                run_metadata=metadata,
            )
    # Keep the parts that do not depend on timing, so that the goldens are
    # stable: the partition graphs, the cost graph without times, and one
    # step statistic with fixed values.
    stable = v1.RunMetadata()
    stable.partition_graphs.extend(metadata.partition_graphs)
    stable.cost_graph.CopyFrom(metadata.cost_graph)
    for node in stable.cost_graph.node:
        node.compute_cost = node.compute_time = node.memory_time = 0
    device = stable.step_stats.dev_stats.add(device="/job:localhost/device:CPU:0")
    device.node_stats.add(node_name="x", all_start_micros=5, op_end_rel_micros=2)
    return graph, meta, stable


def _write_golden(run: Path) -> None:
    """Write the graph of a run and the profile text of its run metadata."""
    target = GOLDEN / run.name
    target.mkdir(parents=True)
    accumulator = event_accumulator.EventAccumulator(str(run))
    accumulator.Reload()
    tags = accumulator.Tags()
    if tags[event_accumulator.GRAPH]:
        (target / "graph.pb").write_bytes(accumulator._graph)
    if tags[event_accumulator.META_GRAPH]:
        (target / "meta_graph.pb").write_bytes(accumulator._meta_graph)
    profile_plugins = {
        graph_metadata.PLUGIN_NAME_TAGGED_RUN_METADATA,
        graph_metadata.PLUGIN_NAME_RUN_METADATA,
    }
    for path in sorted(run.glob("events.out.tfevents.*")):
        initial_metadata: dict[str, object] = {}
        # The legacy loader does not migrate; migrate_event does it here.
        for event in event_file_loader.LegacyEventFileLoader(str(path)).Load():
            for migrated in dataclass_compat.migrate_event(event, initial_metadata):
                for value in migrated.summary.value:
                    plugin = value.metadata.plugin_data.plugin_name
                    content = value.metadata.plugin_data.content
                    # The graph plugin ignores a graph_run_metadata tag
                    # without content b"1".
                    if plugin not in profile_plugins or (
                        plugin == graph_metadata.PLUGIN_NAME_RUN_METADATA
                        and content != b"1"
                    ):
                        continue
                    run_metadata = config_pb2.RunMetadata.FromString(
                        value.tensor.string_val[0]
                    )
                    name = f"{value.tag}.{migrated.step}"
                    (target / f"{name}.pbtxt").write_text(str(run_metadata))
                    if plugin == graph_metadata.PLUGIN_NAME_RUN_METADATA:
                        graph = graph_util.merge_graph_defs(
                            [
                                f.pre_optimization_graph
                                for f in run_metadata.function_graphs
                            ]
                        )
                        (target / f"{name}.pb").write_bytes(graph.SerializeToString())


if __name__ == "__main__":
    main()
