#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["av==19.0.2", "tensorboard==2.21.0"]
# ///
"""Write valid RF64 and BW64 audio cases using TensorBoard's low-level writer."""

import argparse
import math
import struct
from io import BytesIO
from pathlib import Path

import av
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.record_writer import RecordWriter

RATE = 8000
FRAMES = 800
RUN_NAME = "wave64"


def _wave64(form: bytes, pcm: bytes) -> bytes:
    """Build a mono RF64 or BW64 file with 64-bit size fields."""
    fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
    size = 12 + 36 + 24 + 8 + len(pcm)
    ds64 = struct.pack("<QQQI", size - 8, len(pcm), FRAMES, 0)
    return b"".join(
        (
            form,
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVE",
            b"ds64",
            struct.pack("<I", len(ds64)),
            ds64,
            b"fmt ",
            struct.pack("<I", len(fmt)),
            fmt,
            b"data",
            struct.pack("<I", 0xFFFFFFFF),
            pcm,
        )
    )


def _decode(blob: bytes, pcm: bytes, form: bytes) -> None:
    """Require FFmpeg to decode the complete, exact-duration PCM tone."""
    decoded = bytearray()
    frames = 0
    with av.open(BytesIO(blob)) as container:
        for frame in container.decode(audio=0):
            if frame.sample_rate != RATE or len(frame.layout.channels) != 1:
                raise ValueError(f"{form.decode()}: unexpected audio rate or channels")
            decoded.extend(frame.to_ndarray().astype("<i2", copy=False).tobytes())
            frames += frame.samples
    if frames != FRAMES:
        raise ValueError(
            f"{form.decode()}: FFmpeg decoded the wrong duration at {RATE} Hz"
        )
    if decoded != pcm:
        raise ValueError(
            f"{form.decode()}: FFmpeg did not recover the PCM samples exactly"
        )


def main() -> None:
    """Write legacy and TF2-layout RF64/BW64 summaries into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")

    pcm = struct.pack(
        f"<{FRAMES}h",
        *(round(12000 * math.sin(math.tau * 440 * i / RATE)) for i in range(FRAMES)),
    )
    summary = summary_pb2.Summary()
    for label, form in (("rf64", b"RF64"), ("bw64", b"BW64")):
        blob = _wave64(form, pcm)
        _decode(blob, pcm, form)
        legacy = summary.value.add(tag=f"legacy-{label}").audio
        legacy.sample_rate, legacy.num_channels, legacy.length_frames = RATE, 1, FRAMES
        legacy.content_type = "audio/wav"
        legacy.encoded_audio_string = blob
        tensor_value = summary.value.add(tag=f"tf2-{label}")
        tensor_value.metadata.plugin_data.plugin_name = "audio"
        tensor_value.tensor.dtype = types_pb2.DT_STRING
        tensor_value.tensor.tensor_shape.dim.add(size=1)
        tensor_value.tensor.tensor_shape.dim.add(size=2)
        tensor_value.tensor.string_val.extend(
            (blob, f"440 Hz {label.upper()}".encode())
        )

    event = event_pb2.Event(wall_time=1.0, step=7, summary=summary)
    path = output / f"events.out.tfevents.{RUN_NAME}"
    with path.open("xb") as handle:
        writer = RecordWriter(handle)
        writer.write(event_pb2.Event(file_version="brain.Event:2").SerializeToString())
        writer.write(event.SerializeToString())


if __name__ == "__main__":
    main()
