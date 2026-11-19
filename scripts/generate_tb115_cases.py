#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "tensorboard==1.15.0",
#   "numpy==1.23.5",
#   "protobuf==3.20.3",
#   "setuptools==80.9.0",
# ]
# ///
"""Write historical event variants with TensorBoard 1.15's schema and RecordWriter."""

import argparse
from pathlib import Path

import numpy as np
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.debugger.health_pill_calc import calc_health_pill
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.util.tensor_util import make_tensor_proto


def main() -> None:
    """Generate fixed records with opaque values beside an ordinary scalar series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    event = event_pb2.Event
    status = event_pb2.SessionLog
    series = summary_pb2.Summary()
    series.value.add(tag="loss", simple_value=2.5)
    series.value.add(tag="historical", obsolete_old_style_histogram=b"\x00\xffopaque")
    series.value.add(tag="metadata-only", node_name="legacy/node")
    health = series.value.add(
        tag="activation", node_name="activation:0:DebugNumericSummary"
    )
    health.metadata.plugin_data.plugin_name = "debugger"
    health.metadata.plugin_data.content = b'{"device":"CPU:0","outputSlot":0}'
    counts = calc_health_pill(np.array([-2.0, 0.0, 2.0, np.nan]))
    health.tensor.CopyFrom(make_tensor_proto(np.array(counts, dtype=np.float64)))
    records = (
        event(wall_time=1, file_version="brain.Event:2"),
        event(wall_time=2, step=1, summary=series),
        event(
            wall_time=3,
            log_message=event_pb2.LogMessage(level=30, message="retained warning"),
        ),
        event(
            wall_time=4,
            session_log=status(
                status=status.CHECKPOINT, checkpoint_path="checkpoint-1"
            ),
        ),
        event(wall_time=5, session_log=status(status=status.STOP, msg="paused")),
        event(
            wall_time=6,
            step=1,
            summary=summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag="loss", simple_value=1.25)]
            ),
        ),
    )
    with (output / "events.out.tfevents.tb115").open("xb") as handle:
        writer = RecordWriter(handle)
        for record in records:
            writer.write(record.SerializeToString())

    frame = summary_pb2.Summary()
    pixels = np.array([[[255, 0, 128]]], dtype=np.uint8)
    frame.value.add(tag="beholder-frame", tensor=make_tensor_proto(pixels))
    beholder = output / "plugins/beholder"
    beholder.mkdir(parents=True)
    (beholder / "frame.summary").write_bytes(frame.SerializeToString())


if __name__ == "__main__":
    main()
