#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
# ///
"""Write a checkpoint-backed embedding and a synthetic Debugger V2 capture."""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.core.protobuf import debug_event_pb2 as debug
from tensorflow.python.debug.lib.debug_events_writer import DebugEventsWriter


def _debug_dump(directory: Path) -> None:
    writer = DebugEventsWriter(str(directory), "fixture-run")
    try:
        writer.WriteSourceFile(
            debug.SourceFile(file_path="fixture.py", lines=["x = 7"])
        )
        frame = debug.StackFrameWithId(id="fixture-frame")
        frame.file_line_col.line = 1
        writer.WriteStackFrameWithId(frame)
        writer.WriteDebuggedGraph(debug.DebuggedGraph(graph_id="fixture-graph"))
        value = tf.make_tensor_proto(7)
        writer.WriteExecution(
            debug.Execution(
                op_type="Const",
                num_outputs=1,
                tensor_debug_mode=debug.FULL_TENSOR,
                output_tensor_ids=[1],
                tensor_protos=[value],
            )
        )
        writer.WriteGraphExecutionTrace(
            debug.GraphExecutionTrace(
                tfdbg_context_id="fixture-graph",
                op_name="Const",
                tensor_debug_mode=debug.FULL_TENSOR,
                tensor_proto=value,
            )
        )
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
    finally:
        writer.Close()
    for part in directory.glob("tfdbg_events.*"):
        part.rename(directory / f"tfdbg_events.0000000000.fixture{part.suffix}")


def main() -> None:
    """Write a scalar alongside checkpoint and debugger files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(output))
    with writer.as_default():
        tf.summary.scalar("control", 1.25, step=3)
    writer.close()
    (event,) = output.glob("events.out.tfevents.*")
    event.rename(output / "events.out.tfevents.auxiliary")
    tf.raw_ops.SaveV2(
        prefix=str(output / "weights"),
        tensor_names=["vectors"],
        shape_and_slices=[""],
        tensors=[tf.constant([[1.0, 2.0], [3.0, 4.0]])],
    )
    config = projector.ProjectorConfig(model_checkpoint_path="weights")
    config.embeddings.add(tensor_name="vectors")
    projector.visualize_embeddings(str(output), config)
    _debug_dump(output / "debug")


if __name__ == "__main__":
    main()
