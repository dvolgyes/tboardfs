#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorboard==2.21.0", "numpy==2.4.6", "pillow==12.1.1"]
# ///
"""Generate compact embedded-format cases with TensorBoard's real event writer."""

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


class _EmbeddedRun:
    """Build one summary from native media, arrays, and archive controls."""

    summary: summary_pb2.Summary

    def __init__(self) -> None:
        self.summary = summary_pb2.Summary()

    def add_midi(self) -> None:
        """Write a C-major arpeggio as standard MIDI, without a RIFF envelope."""
        track = bytearray()
        for pitch in (60, 64, 67):
            track.extend((0, 0x90, pitch, 100, 96, 0x80, pitch, 0))
        track.extend(bytes.fromhex("00 ff 2f 00"))
        audio = self.summary.value.add(tag="midi").audio
        audio.content_type = "audio/mid"
        audio.encoded_audio_string = (
            b"MThd"
            + bytes.fromhex("00 00 00 06 00 00 00 01 00 60")
            + b"MTrk"
            + len(track).to_bytes(4, "big")
            + track
        )

    def add_tiff(self) -> None:
        """Store two distinct grayscale planes in a big-endian TIFF."""
        planes = [
            np.fromfunction(fn, (8, 16), dtype=int).astype(">u2")
            for fn in (
                lambda y, x: x + 300 * y,
                lambda y, x: 12000 + 5 * x - 700 * y,
            )
        ]
        first, second = (Image.fromarray(plane) for plane in planes)
        output = BytesIO()
        first.save(output, format="TIFF", save_all=True, append_images=[second])
        payload = output.getvalue()
        if payload[:2] != b"MM":
            raise ValueError("expected big-endian TIFF output")
        image = self.summary.value.add(tag="multipage-tiff").image
        image.width, image.height, image.colorspace = 16, 8, 1
        image.encoded_image_string = payload

    def add_arrays_and_archives(self) -> None:
        """Exercise native-file sniffing and the separate typed numeric export."""
        expected = np.arange(384, dtype=np.int32).reshape((16, 24), order="F") - 170
        encoded = BytesIO()
        np.save(encoded, expected.astype(">i4", order="F"), allow_pickle=False)
        npy = encoded.getvalue()
        npz = BytesIO()
        np.savez(npz, values=np.arange(512, dtype="<i2"))
        plain_zip = BytesIO()
        with zipfile.ZipFile(plain_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("squares.csv")
            archive.writestr(info, "".join(f"{i},{i * i}\n" for i in range(160)))
        for tag, blob in (
            ("matrix-npy", npy),
            ("plain-zip", plain_zip.getvalue()),
            ("numeric-npz", npz.getvalue()),
        ):
            if len(blob) < 1024:
                raise ValueError(f"{tag} must be at least 1024 bytes")
            tensor = self.summary.value.add(tag=tag).tensor
            tensor.dtype = types_pb2.DT_STRING
            tensor.tensor_shape.SetInParent()
            tensor.string_val.append(blob)
        wrapped = self.summary.value.add(tag="wrapped-npy").tensor
        wrapped.dtype = types_pb2.DT_UINT8
        wrapped.tensor_shape.dim.add(size=len(npy))
        wrapped.tensor_content = npy


def _write(output: Path) -> Path:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run = _EmbeddedRun()
    run.add_midi()
    run.add_tiff()
    run.add_arrays_and_archives()
    destination = output / "events.out.tfevents.embedded-formats"
    with destination.open("xb") as handle:
        writer = RecordWriter(handle)
        for event in (
            event_pb2.Event(file_version="brain.Event:2", wall_time=1),
            event_pb2.Event(step=7, wall_time=2, summary=run.summary),
        ):
            writer.write(event.SerializeToString(deterministic=True))
    return destination


def main() -> None:
    """Write the fixture into a caller-selected new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output)


if __name__ == "__main__":
    main()
