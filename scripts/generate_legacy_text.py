#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.6,<3.7"
# dependencies = [
#   "tensorflow==1.2.0", "numpy==1.18.5", "protobuf==3.19.6",
#   "werkzeug==1.0.1", "html5lib==0.9999999",
# ]
# ///
"""Write native TF 1.2 text and ordinary tensor summaries; verify with TB 1.15."""

import argparse
import shutil
import subprocess
from pathlib import Path

import tensorflow as tf
from tensorflow.python.ops.summary_ops import tensor_summary
from tensorflow.python.summary.text_summary import text_summary

READER = """
import sys
from types import SimpleNamespace
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
from tensorboard.plugins.text.text_plugin import TextPlugin

mux = EventMultiplexer()
mux.AddRun(sys.argv[1], 'run')
mux.Reload()
plugin = TextPlugin(SimpleNamespace(multiplexer=mux))
if plugin.index_impl() != {'run': ['registered_text']}:
    raise ValueError('unexpected Text dashboard registry')
for tag, cell in [('registered_text', b'registered'), ('unregistered_text', b'ordinary')]:
    event, = mux.Tensors('run', tag)
    if event.tensor_proto.string_val[1] != cell:
        raise ValueError('incorrect tensor for ' + tag)
"""


def main() -> None:
    """Capture both native summaries and check the historical dashboard index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    uv = shutil.which("uv")
    if not uv:
        parser.error("uv is required for the reference reader")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tf.Graph().as_default() as graph:
        registered = text_summary(
            "registered_text", tf.constant([["legacy", "registered"], ["café", ""]])
        )
        ordinary = tensor_summary(
            "unregistered_text",
            tf.constant([["legacy", "ordinary"], ["x|y", "two\nlines"]]),
        )
        writer = tf.summary.FileWriter(str(output), graph=graph)
        try:
            with tf.Session(graph=graph) as session:
                writer.add_summary(
                    session.run(tf.summary.merge([registered, ordinary])), 4
                )
        finally:
            writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.legacy-text")
    with (output / "plugins/tensorboard_text/tensors.json").open("ab") as registry:
        registry.write(b"\n")
    subprocess.run(  # noqa: S603
        [
            uv,
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.8",
            "--with",
            "tensorboard==1.15.0,numpy==1.19.5,protobuf==3.20.3,werkzeug==1.0.1,setuptools==67.8.0",
            "python",
            "-I",
            "-c",
            READER,
            str(output),
        ],
        check=True,
        shell=False,
    )


if __name__ == "__main__":
    main()
