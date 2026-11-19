#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = [
#   "tensorflow-cpu==1.15.0", "tensorboard==1.15.0", "numpy==1.18.5",
#   "protobuf==3.20.3", "setuptools==67.8.0", "werkzeug==1.0.1",
# ]
# ///
"""Write native Beholder arrays and a schema-only op profile, not a TPU capture."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from google.protobuf import json_format
from tensorboard.compat.proto import event_pb2
from tensorboard.plugins.beholder import beholder
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorflow.core.profiler import op_profile_pb2


def _write_beholder(run: Path) -> None:
    """Capture two arrays with the native current-values dashboard configuration."""
    instance = beholder.Beholder(str(run))
    config = dict(
        beholder.DEFAULT_CONFIG, values="arrays", mode="current", colormap="grayscale"
    )
    config_path = run / "plugins/beholder/config.pkl"
    beholder.write_pickle(config, str(config_path))
    instance.previous_config = config
    instance.config_last_modified_time = config_path.stat().st_mtime
    instance.last_update_time = 0
    arrays = [
        np.array([[-3, 0, 3], [1, -1, 2]], np.float32),
        np.arange(8, dtype=np.float32),
    ]
    with tf.compat.v1.Session() as session:
        instance.update(session, arrays=arrays)


def _write_schema_profile(run: Path) -> None:
    """Encode illustrative XLA instruction metrics using the historical TPU schema."""
    profile = op_profile_pb2.Profile()
    root = profile.by_program
    root.name = "schema-only-program"
    root.num_children = 2
    root.metrics.time = 1.0
    root.metrics.raw_time = 2000
    for op, fraction, elapsed in (("dot", 0.6, 1200), ("add", 0.4, 800)):
        node = root.children.add(name="%" + op + ".1")
        node.xla.op = op
        node.metrics.time = fraction
        node.metrics.raw_time = elapsed
    directory = run / "plugins/profile/schema-only"
    directory.mkdir(parents=True)
    (directory / "host.op_profile.json").write_text(
        json_format.MessageToJson(profile, including_default_value_fields=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate a standalone run in an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    run = parser.parse_args().output
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        parser.error(f"output directory is not empty: {run}")
    with (run / "events.out.tfevents.legacy-tools").open("xb") as stream:
        writer = RecordWriter(stream)
        writer.write(event_pb2.Event(file_version="brain.Event:2").SerializeToString())
    _write_beholder(run)
    _write_schema_profile(run)


if __name__ == "__main__":
    main()
