#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorboard>=2.20.0,<3.0",
# ]
# ///
"""Generate a compact edge-case event file with TensorBoard's native writer."""

import argparse
from pathlib import Path
import tempfile

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

FILE_NAME = "events.out.tfevents.tensorboard"


def main() -> None:
    """Write fixed summary observations; TensorBoard timestamps its own header."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        writer = EventFileWriter(temporary)
        try:
            unknown_only = event_pb2.Event.FromString(b"\xa0\x06\x07")
            writer.add_event(unknown_only)
            values = [
                summary_pb2.Summary.Value(tag="rate%", simple_value=1.25),
                summary_pb2.Summary.Value(tag="group//metric", simple_value=2.5),
                summary_pb2.Summary.Value(tag="group/./metric", simple_value=3.75),
                summary_pb2.Summary.Value(tag="group/metric", simple_value=9.0),
                summary_pb2.Summary.Value(tag="rate%.json/x", simple_value=10.0),
                summary_pb2.Summary.Value(tag="", simple_value=4.0),
                summary_pb2.Summary.Value(
                    tag="opaque",
                    simple_value=5.0,
                    metadata=summary_pb2.SummaryMetadata(
                        plugin_data=summary_pb2.SummaryMetadata.PluginData(
                            plugin_name="unrecognized",
                            content=b"\x00\xffmetadata",
                        )
                    ),
                ),
            ]
            writer.add_event(
                event_pb2.Event(
                    wall_time=10.0,
                    step=2,
                    summary=summary_pb2.Summary(value=values),
                )
            )
            for step, tag, value in (
                (2, "rate%", 6.0),
                (2**53 + 7, "rate%", 7.0),
                (1, "rate%", 8.0),
            ):
                writer.add_event(
                    event_pb2.Event(
                        wall_time=20.0 + step % 10,
                        step=step,
                        summary=summary_pb2.Summary(
                            value=[
                                summary_pb2.Summary.Value(tag=tag, simple_value=value)
                            ]
                        ),
                    )
                )
            writer.flush()
        finally:
            writer.close()
        files = list(Path(temporary).iterdir())
        if len(files) != 1:
            raise RuntimeError(f"expected one event file, found {len(files)}")
        files[0].replace(output / FILE_NAME)


if __name__ == "__main__":
    main()
