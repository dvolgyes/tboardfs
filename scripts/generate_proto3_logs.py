#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0",
# ]
# ///
"""Generate proto3 events with default field values under test-logs/proto3, with TensorBoard's views.

Proto3 writers leave out every field that has its default value: step 0,
wall time 0, a histogram minimum or maximum of 0, a tensor dimension of size
0 and empty plugin content. TensorBoard reads an absent field as its
default. The events are written with TensorBoard's own proto3 classes and
fixed wall times, so the files are the same on each run.

Runs:
    step_zero: TF1 and TF2 scalars at step 0, and the value 0.0.
    wall_time_zero: scalars written without a wall time.
    histogram_zero_edges: TF1 histograms whose minimum or maximum is 0, and
        TF2 histograms with a 0 edge or no buckets.
    tensor_zero_dims: tensors with a dimension of size 0, zero values, and
        single values that fill a shape.
    empty_plugin_content: a text summary with empty plugin content.
    empty_tag: TF1 scalars without a tag.

tests/golden/proto3/<run>.json is what the TensorBoard 2.21 multiplexer
reads from the run, after data_compat and dataclass_compat: scalar rows as
[step, wall_time, value], histogram buckets as [left, right, count] per
step, and tensors as their dtype, shape and values per step.
"""

import json
import shutil
from pathlib import Path

from tensorboard import context
from tensorboard.backend.event_processing import (
    data_provider,
    plugin_event_multiplexer,
)
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.util import tensor_util

ROOT = Path("test-logs") / "proto3"
GOLDEN = Path("tests") / "golden" / "proto3"
FILE_NAME = "events.out.tfevents.1.proto3"
MAX_ITEMS = 1000


def main() -> None:
    """Write the runs and the expected views, replacing earlier ones."""
    for directory in (ROOT, GOLDEN):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    runs = {
        "step_zero": [
            _Events.event(
                0,
                10.0,
                _Events.simple("loss", 1.5),
                _Events.tf2_scalar("tf2_loss", 2.5),
            ),
            _Events.event(
                1,
                11.0,
                _Events.simple("loss", 0.0),
                _Events.tf2_scalar("tf2_loss", 0.0),
            ),
        ],
        "wall_time_zero": [
            _Events.event(1, 0.0, _Events.simple("loss", 1.0)),
            _Events.event(2, 0.0, _Events.simple("loss", 2.0)),
        ],
        "histogram_zero_edges": [
            _Events.event(
                1,
                10.0,
                _Events.histogram("zero_min", 0.0, 2.0, [1.0, 2.0], [1.0, 2.0]),
                _Events.histogram("zero_max", -2.0, 0.0, [-1.0, 0.0], [2.0, 1.0]),
                _Events.histogram("all_zero", 0.0, 0.0, [0.0, 1.0], [3.0, 0.0]),
                _Events.tf2_histogram(
                    "tf2_zero_edge", [[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]]
                ),
                _Events.tf2_histogram("tf2_empty", []),
            )
        ],
        "tensor_zero_dims": [
            _Events.event(
                1,
                10.0,
                _Events.tensor("empty_rows", types_pb2.DT_FLOAT, [3, 0]),
                _Events.tensor("empty", types_pb2.DT_FLOAT, [0]),
                _Events.tensor("zero_int", types_pb2.DT_INT32, [], int_val=[0]),
                _Events.tensor("zeros", types_pb2.DT_FLOAT, [2], float_val=[0.0, 0.0]),
                _Events.tensor(
                    "zero_filled", types_pb2.DT_FLOAT, [2, 2], float_val=[0.0]
                ),
                _Events.tensor(
                    "filled_empty", types_pb2.DT_FLOAT, [0], float_val=[1.0]
                ),
            )
        ],
        "empty_plugin_content": [
            _Events.event(1, 10.0, _Events.text("note", b"hello")),
            _Events.event(2, 11.0, _Events.text("note", b"")),
        ],
        "empty_tag": [
            _Events.event(1, 10.0, _Events.simple("", 1.0)),
            _Events.event(2, 11.0, _Events.simple("", 2.0)),
        ],
    }
    for name, events in runs.items():
        run = ROOT / name
        run.mkdir()
        header = event_pb2.Event(wall_time=1.0, file_version="brain.Event:2")
        with (run / FILE_NAME).open("wb") as handle:
            writer = RecordWriter(handle)
            for event in (header, *events):
                writer.write(event.SerializeToString())
    _Golden.write()


class _Events:
    """Build events and summary values with TensorBoard's proto3 classes."""

    @staticmethod
    def event(
        step: int, wall_time: float, *values: summary_pb2.Summary.Value
    ) -> event_pb2.Event:
        """Return one event with the given summary values."""
        event = event_pb2.Event(step=step, wall_time=wall_time)
        event.summary.value.extend(values)
        return event

    @staticmethod
    def simple(tag: str, value: float) -> summary_pb2.Summary.Value:
        """Return a TF1 scalar value."""
        return summary_pb2.Summary.Value(tag=tag, simple_value=value)

    @staticmethod
    def tf2_scalar(tag: str, value: float) -> summary_pb2.Summary.Value:
        """Return a TF2 scalar value, as tf.summary.scalar writes it."""
        summary_value = _Events.tensor(tag, types_pb2.DT_FLOAT, [], float_val=[value])
        summary_value.metadata.plugin_data.plugin_name = "scalars"
        return summary_value

    @staticmethod
    def histogram(
        tag: str, low: float, high: float, limits: list[float], counts: list[float]
    ) -> summary_pb2.Summary.Value:
        """Return a TF1 histogram value."""
        summary_value = summary_pb2.Summary.Value(tag=tag)
        histo = summary_value.histo
        histo.min, histo.max, histo.num = low, high, sum(counts)
        histo.bucket_limit.extend(limits)
        histo.bucket.extend(counts)
        return summary_value

    @staticmethod
    def tf2_histogram(
        tag: str, buckets: list[list[float]]
    ) -> summary_pb2.Summary.Value:
        """Return a TF2 histogram value: a [k, 3] tensor of left, right and count."""
        values = [number for bucket in buckets for number in bucket]
        shape = [len(buckets), 3]
        summary_value = _Events.tensor(tag, types_pb2.DT_FLOAT, shape, float_val=values)
        summary_value.metadata.plugin_data.plugin_name = "histograms"
        return summary_value

    @staticmethod
    def tensor(
        tag: str,
        dtype: int,
        shape: list[int],
        *,
        float_val: list[float] | None = None,
        int_val: list[int] | None = None,
    ) -> summary_pb2.Summary.Value:
        """Return a tensor value without a plugin."""
        summary_value = summary_pb2.Summary.Value(tag=tag)
        tensor = summary_value.tensor
        tensor.dtype = dtype
        # An empty shape is still written, as a message without dimensions.
        tensor.tensor_shape.SetInParent()
        for size in shape:
            tensor.tensor_shape.dim.add(size=size)
        tensor.float_val.extend(float_val or [])
        tensor.int_val.extend(int_val or [])
        return summary_value

    @staticmethod
    def text(tag: str, text: bytes) -> summary_pb2.Summary.Value:
        """Return a text value with empty plugin content, as tf.summary.text writes it."""
        summary_value = _Events.tensor(tag, types_pb2.DT_STRING, [])
        summary_value.tensor.string_val.append(text)
        summary_value.metadata.plugin_data.plugin_name = "text"
        return summary_value


class _Golden:
    """Read the runs with TensorBoard and write its views."""

    @staticmethod
    def write() -> None:
        """Write what TensorBoard reads from each run."""
        multiplexer = plugin_event_multiplexer.EventMultiplexer()
        multiplexer.AddRunsFromDirectory(str(ROOT))
        multiplexer.Reload()
        provider = data_provider.MultiplexerDataProvider(multiplexer, str(ROOT))
        ctx = context.RequestContext()
        scalars = provider.read_scalars(
            ctx, experiment_id="", plugin_name="scalars", downsample=MAX_ITEMS
        )
        histograms = provider.read_tensors(
            ctx, experiment_id="", plugin_name="histograms", downsample=MAX_ITEMS
        )
        for run in sorted(multiplexer.Runs()):
            view: dict[str, object] = {
                "scalars": {
                    tag: [[datum.step, datum.wall_time, datum.value] for datum in data]
                    for tag, data in sorted(scalars.get(run, {}).items())
                },
                "histograms": {
                    tag: {str(datum.step): datum.numpy.tolist() for datum in data}
                    for tag, data in sorted(histograms.get(run, {}).items())
                },
                "tensors": {
                    tag: {
                        str(event.step): _Golden.array_view(event.tensor_proto)
                        for event in multiplexer.Tensors(run, tag)
                    }
                    for tag in sorted(multiplexer.GetAccumulator(run).Tags()["tensors"])
                    if tag not in scalars.get(run, {})
                    and tag not in histograms.get(run, {})
                },
            }
            text = json.dumps(view, indent=2, sort_keys=True) + "\n"
            (GOLDEN / f"{run}.json").write_text(text)

    @staticmethod
    def array_view(tensor_proto: object) -> dict[str, object]:
        """Return the dtype, shape and values that TensorBoard decodes from a tensor."""
        array = tensor_util.make_ndarray(tensor_proto)
        values = [
            value.decode() if isinstance(value, bytes) else value
            for value in array.ravel().tolist()
        ]
        return {"dtype": str(array.dtype), "shape": list(array.shape), "values": values}


if __name__ == "__main__":
    main()
