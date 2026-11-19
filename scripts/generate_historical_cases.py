#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "tensorboard>=1.15.0,<=2.0.0", "numpy==1.22.4",
#   "protobuf==3.20.3", "setuptools==80.9.0",
# ]
# ///
"""Write the same plugin cases using either historical TensorBoard distribution.

No TensorFlow dependency: use the release's NumPy-based summary helpers,
or its own protobuf/metadata builders for TF-dependent ops.
NumPy 1.22 retains asscalar, required by the native scalar and dtype helpers.
The CPU profile and Beholder snapshot are constructed sidecars, not captures.

Select one of the two release environments explicitly:

    uv run --with tensorboard==1.15.0 --script generate_historical_cases.py --release 1.15.0 -o history-115
    uv run --with tensorboard==2.0.0 --script generate_historical_cases.py --release 2.0.0 -o history-200
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tensorboard import version
from tensorboard.compat.proto import (
    config_pb2,
    event_pb2,
    graph_pb2,
    meta_graph_pb2,
    summary_pb2,
)
from tensorboard.plugins import projector
from tensorboard.plugins.custom_scalar import layout_pb2, metadata as layout_meta
from tensorboard.plugins.debugger.health_pill_calc import calc_health_pill
from tensorboard.plugins.histogram.summary_v2 import histogram_pb
from tensorboard.plugins.hparams import (
    api_pb2,
    metadata as hp_meta,
    plugin_data_pb2 as hp_data,
)
from tensorboard.plugins.hparams import summary_v2 as hp
from tensorboard.plugins.mesh import metadata as mesh_meta
from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData
from tensorboard.plugins.pr_curve import metadata as pr_meta
from tensorboard.plugins.profile import trace_events_pb2
from tensorboard.plugins.scalar.summary_v2 import scalar_pb
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.plugins.text.summary_v2 import text_pb
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.util.tensor_util import make_tensor_proto


def _geometry(summary: summary_pb2.Summary) -> None:
    """Build mesh components with native metadata; mesh_pb itself needs TF dtypes."""
    vertices = np.array([[[0, 0, 0], [2, 0, 0], [0, 3, 0]]], np.float32)
    faces = np.array([[[0, 1, 2]]], np.int32)
    colors = np.array([[[220, 0, 0], [0, 180, 0], [0, 0, 200]]], np.uint8)
    for tag, parts in (
        (
            "triangle",
            [
                (MeshPluginData.VERTEX, vertices),
                (MeshPluginData.FACE, faces),
                (MeshPluginData.COLOR, colors),
            ],
        ),
        (
            "cloud",
            [(MeshPluginData.VERTEX, vertices + np.array([0, 0, 1], np.float32))],
        ),
    ):
        mask = mesh_meta.get_components_bitmask([kind for kind, _ in parts])
        for kind, array in parts:
            summary.value.add(
                tag=mesh_meta.get_instance_name(tag, kind),
                metadata=mesh_meta.create_summary_metadata(
                    tag, tag, kind, mask, array.shape, json_config="{}"
                ),
                tensor=make_tensor_proto(array),
            )


def _summary() -> summary_pb2.Summary:
    summary = scalar_pb("accuracy", 0.75)
    summary.value.add(tag="classic").histo.CopyFrom(
        summary_pb2.HistogramProto(
            min=0,
            max=4,
            num=5,
            sum=10,
            sum_squares=30,
            bucket_limit=[2, 4],
            bucket=[2, 3],
        )
    )
    summary.value.add(tag="obsolete", obsolete_old_style_histogram=b"\x00\xfflegacy")
    summary.value.add(tag="metadata-only", node_name="validation/input")
    domain = hp.HParam("batch", hp.Discrete([8, 16]))
    for part in (
        histogram_pb("scores", [0, 1, 2, 3, 4], buckets=2),
        text_pb("labels", [[b"cat", b"dog"], [b"owl", b"fox"]]),
        hp.hparams_config_pb([domain], [hp.Metric("accuracy")], time_created_secs=1),
        hp.hparams_pb({domain: 16}, trial_id="baseline", start_time_secs=2),
    ):
        summary.MergeFrom(part)
    end = hp_data.HParamsPluginData()
    end.session_end_info.status = api_pb2.STATUS_SUCCESS
    end.session_end_info.end_time_secs = 10
    summary.value.add(
        tag=hp_meta.SESSION_END_INFO_TAG, metadata=hp_meta.create_summary_metadata(end)
    )
    layout = layout_pb2.Layout()
    chart = layout.category.add(title="classification").chart.add(title="validation")
    chart.multiline.tag.append("accuracy")
    summary.value.add(
        tag=layout_meta.CONFIG_SUMMARY_TAG,
        metadata=layout_meta.create_summary_metadata(),
        tensor=make_tensor_proto(layout.SerializeToString(), dtype=object),
    )
    labels = np.array([True, False, True, False])
    positive = np.array([0.9, 0.7, 0.4, 0.1]) >= np.linspace(0, 1, 5)[:, None]
    tp, fp = (np.sum(positive & mask, axis=1) for mask in (labels, ~labels))
    rows = [tp, fp, 2 - fp, 2 - tp, tp / np.maximum(tp + fp, 1), tp / 2]
    summary.value.add(
        tag="classifier/pr_curves",
        metadata=pr_meta.create_summary_metadata(
            "classifier", "validation predictions", 5
        ),
        tensor=make_tensor_proto(np.array(rows, np.float32)),
    )
    _geometry(summary)
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.complex64,
        np.complex128,
        np.bool_,
    ):
        array = np.array([[0, 1], [3, 2]], dtype)
        if np.issubdtype(dtype, np.complexfloating):
            array += 2j
        summary.value.add(tag=array.dtype.name, tensor=make_tensor_proto(array))
    summary.value.add(
        tag="words", tensor=make_tensor_proto([b"birch", b"fir"], dtype=object)
    )
    counts = calc_health_pill(np.array([-1.0, 0.0, 3.0, np.nan]))
    debug = summary.value.add(
        tag="activations",
        node_name="activations:0:DebugNumericSummary",
        tensor=make_tensor_proto(np.array(counts, dtype=np.float64)),
    )
    debug.metadata.plugin_data.plugin_name = "debugger"
    debug.metadata.plugin_data.content = b'{"device":"CPU:0","outputSlot":0}'
    return summary


def main() -> None:
    """Write fixed observations with the historical release's RecordWriter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--release", choices=("1.15.0", "2.0.0"), required=True)
    args = parser.parse_args()
    output, release = args.output, args.release
    if release != version.VERSION:
        raise RuntimeError(f"expected TensorBoard {release}, found {version.VERSION}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    graph = graph_pb2.GraphDef()
    graph.node.add(name="input", op="Placeholder").attr["dtype"].type = 1
    graph.node.add(name="energy", op="Square", input=["input"]).attr["T"].type = 1
    meta = meta_graph_pb2.MetaGraphDef(graph_def=graph)
    meta.meta_info_def.tags.append("serve")
    run = config_pb2.RunMetadata()
    run.step_stats.dev_stats.add(device="/device:CPU:0").node_stats.add(
        node_name="energy", all_start_micros=12
    )
    event = event_pb2.Event
    status = event_pb2.SessionLog
    records = (
        event(file_version="brain.Event:2", wall_time=1),
        event(wall_time=2, session_log=status(status=status.START, msg="training")),
        event(wall_time=3, graph_def=graph.SerializeToString()),
        event(wall_time=4, meta_graph_def=meta.SerializeToString()),
        event(wall_time=5, step=4, summary=_summary()),
        event(
            wall_time=6,
            step=4,
            tagged_run_metadata=event_pb2.TaggedRunMetadata(
                tag="worker", run_metadata=run.SerializeToString()
            ),
        ),
        event(
            wall_time=7,
            log_message=event_pb2.LogMessage(level=30, message="slow batch"),
        ),
        event(
            wall_time=8,
            session_log=status(status=status.CHECKPOINT, checkpoint_path="model-4"),
        ),
        event(wall_time=9, session_log=status(status=status.STOP, msg="finished")),
    )
    with (output / f"events.out.tfevents.tensorboard-{release}").open("xb") as handle:
        writer = RecordWriter(handle)
        for record in records:
            writer.write(record.SerializeToString())
    config = projector.ProjectorConfig()
    config.embeddings.add(
        tensor_name="birds", tensor_path="vectors.tsv", metadata_path="birds.tsv"
    )
    if release == "1.15.0":
        with TemporaryDirectory() as temporary:
            writer = EventFileWriter(temporary)
            try:
                projector.visualize_embeddings(writer, config)
            finally:
                writer.close()
            (output / "projector_config.pbtxt").write_bytes(
                (Path(temporary) / "projector_config.pbtxt").read_bytes()
            )
    else:
        projector.visualize_embeddings(str(output), config)
    (output / "vectors.tsv").write_bytes(b"1\t0\n0\t1\n")
    (output / "birds.tsv").write_bytes(b"owl\nfinch\n")
    trace = trace_events_pb2.Trace()
    cpu = trace.devices[1]
    cpu.name, cpu.device_id = "CPU", 1
    cpu.resources[2].name = "worker"
    trace.trace_events.add(
        device_id=1,
        resource_id=2,
        name="Square",
        timestamp_ps=8000000,
        duration_ps=2500000,
    )
    directory = output / "plugins/profile/trial"
    directory.mkdir(parents=True)
    (directory / "cpu.trace").write_bytes(trace.SerializeToString())
    y, x = np.indices((16, 16))
    frame = np.stack((x * 17, y * 17, ((x // 4 + y // 4) % 2) * 255), axis=-1)
    beholder = output / "plugins/beholder"
    beholder.mkdir()
    snapshot = summary_pb2.Summary()
    snapshot.value.add(
        tag="beholder-frame", tensor=make_tensor_proto(frame.astype(np.uint8))
    )
    (beholder / "frame.summary").write_bytes(snapshot.SerializeToString())


if __name__ == "__main__":
    main()
