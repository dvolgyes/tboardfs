#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["tensorboardX==2.6.5", "numpy==2.2.6", "pillow==11.3.0", "soundfile==0.13.1"]
# ///
"""Write a TensorBoardX stereo recording, note embeddings, and training summaries."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from tensorboardX.proto.summary_pb2 import Summary


def _notes(writer: SummaryWriter, output: Path) -> None:
    pitches = [220, 330, 440, 660]
    labels = ["A3", "E4", "A4", "E5"]
    t = np.arange(4_000) / 16_000
    fade = np.minimum(1.0, np.minimum(t, 0.25 - t) * 100)
    stereo = np.stack(
        [np.sin(2 * np.pi * f * t) * fade / 2 for f in pitches[:2]], axis=1
    )
    writer.add_audio("interval", stereo, global_step=3, sample_rate=16_000, walltime=3)
    rows, cols = np.indices((16, 16))
    tiles = np.stack([((cols // (i + 2) + rows // 4) % 2) for i in range(4)])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    icons = tiles[:, None] * colors[:, :, None, None]
    writer.add_embedding(
        np.array([[f, 2 * f] for f in pitches]),
        metadata=list(zip(labels, pitches, strict=True)),
        metadata_header=["note", "Hz"],
        label_img=icons,
        global_step=3,
        tag="notes",
    )
    with Image.open(output / "00003/notes/sprite.png") as sprite:
        expected = np.concatenate(
            [
                np.concatenate(pair, axis=1)
                for pair in np.split(icons.transpose(0, 2, 3, 1), 2)
            ]
        )
        np.testing.assert_array_equal(np.asarray(sprite), expected * 255)


def _tensors(writer: SummaryWriter) -> None:
    """Use the public file writer for types with no SummaryWriter.add_tensor API."""
    summary = Summary()
    for dtype, field, storage, values in (
        ("DT_FLOAT", "float_val", "<f4", [0.125, -0.75]),
        ("DT_DOUBLE", "double_val", "<f8", [1 + 2**-48, -0.5]),
        ("DT_HALF", "half_val", "<u2", [0x3E00, 0xC000]),
        ("DT_BFLOAT16", "half_val", "<u2", [0x3FC0, 0xC000]),
        ("DT_COMPLEX64", "scomplex_val", "<f4", [1, 2, -1, 0.5]),
        ("DT_COMPLEX128", "dcomplex_val", "<f8", [1, 2, -1, 0.5]),
        ("DT_BOOL", "bool_val", "?", [False, True]),
        ("DT_INT8", "int_val", "i1", [-12, 97]),
        ("DT_UINT8", "int_val", "u1", [3, 253]),
        ("DT_INT16", "int_val", "<i2", [-8000, 12000]),
        ("DT_UINT16", "int_val", "<u2", [8000, 48000]),
        ("DT_INT32", "int_val", "<i4", [-123456, 654321]),
        ("DT_INT64", "int64_val", "<i8", [2**53 + 1, -7]),
        ("DT_QINT8", "int_val", "i1", [-12, 97]),
        ("DT_QUINT8", "int_val", "u1", [3, 253]),
        ("DT_QINT16", "int_val", "<i2", [-8000, 12000]),
        ("DT_QUINT16", "int_val", "<u2", [8000, 48000]),
        ("DT_QINT32", "int_val", "<i4", [-123456, 654321]),
    ):
        tensor = summary.value.add(tag=f"fields/{dtype}").tensor
        tensor.dtype = dtype
        tensor.tensor_shape.dim.add(size=2)
        getattr(tensor, field).extend(values)
        packed = summary.value.add(tag=f"packed/{dtype}").tensor
        packed.dtype = dtype
        packed.tensor_shape.CopyFrom(tensor.tensor_shape)
        packed.tensor_content = np.array(values, dtype=storage).tobytes()
    strings = summary.value.add(tag="notes").tensor
    strings.dtype = "DT_STRING"
    strings.tensor_shape.dim.add(size=2)
    strings.string_val.extend([b"A3", "ré".encode()])
    resource = summary.value.add(tag="instrument").tensor
    resource.dtype = "DT_RESOURCE"
    resource.resource_handle_val.add(container="synth", name="oscillator")
    writer.file_writer.add_summary(summary, global_step=3, walltime=3)


def main() -> None:
    """Generate a run and its HParams trial in an empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    with SummaryWriter(logdir=str(output)) as writer:
        _notes(writer, output)
        writer.add_hparams(
            {"model": "pitch", "lr": 0.125, "normalize": True},
            {"accuracy": 0.875},
            name="trial",
            global_step=3,
        )
        writer.add_custom_scalars(
            {
                "Validation": {
                    "accuracy": ["Margin", ["score", "low", "high"]],
                    "split": ["Multiline", ["train", "eval"]],
                }
            }
        )
        for tag, value in (
            ("score", 0.875),
            ("low", 0.75),
            ("high", 1.0),
            ("train", 0.9375),
            ("eval", 0.875),
        ):
            writer.add_scalar(tag, value, 3, walltime=3)
        _tensors(writer)
    for path in output.rglob("events.out.tfevents.*"):
        path.rename(path.with_name("events.out.tfevents.tensorboardx-plugins"))


if __name__ == "__main__":
    main()
