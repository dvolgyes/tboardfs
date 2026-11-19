#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorboard==2.21.0"]
# ///
"""Recreate TF 1.2 ProjectorPluginAsset TSV layout and write a TB 2.21 binary config."""

import argparse
import struct
from pathlib import Path

from tensorboard.compat.proto import event_pb2
from tensorboard.plugins import projector
from tensorboard.summary.writer.record_writer import RecordWriter

ROWS = (
    ("alder", "wetland", -0.5, 0.875),
    ("birch", "upland", 0.25, -0.5),
    ("willow", "wetland", -0.625, 1.0),
)


def main() -> None:
    """Write aligned habitat labels and two-dimensional embedding coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    for variant in ("historical", "binary"):
        run = output / variant
        assets = (
            run / "plugins/org_tensorflow_tensorboard_projector"
            if variant == "historical"
            else run
        )
        assets.mkdir(parents=True)
        with (run / "events.out.tfevents.projector-layouts").open("wb") as handle:
            RecordWriter(handle).write(
                event_pb2.Event(file_version="brain.Event:2").SerializeToString()
            )
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add(
            tensor_name="trees",
            metadata_path="trees_metadata.tsv",
            tensor_shape=[len(ROWS), 2],
        )
        (assets / embedding.metadata_path).write_text(
            "species\thabitat\n"
            + "".join(f"{name}\t{habitat}\n" for name, habitat, _, _ in ROWS)
        )
        if variant == "historical":
            embedding.tensor_path = "trees_values.tsv"
            data = "".join(f"{x:.6g}\t{y:.6g}\n" for _, _, x, y in ROWS).encode()
        else:
            embedding.tensor_path = "trees.f32"
            data = b"".join(struct.pack("<ff", x, y) for _, _, x, y in ROWS)
        (assets / embedding.tensor_path).write_bytes(data)
        projector.visualize_embeddings(str(assets), config)


if __name__ == "__main__":
    main()
