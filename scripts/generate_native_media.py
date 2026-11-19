#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "imagecodecs==2026.11.11", "numpy==2.5.4", "tensorboard==2.21.0",
#   "av==19.0.2",
# ]
# ///
"""Generate real native media payloads for TensorBoard image and audio summaries."""

import argparse
from io import BytesIO
from pathlib import Path

import av
import imagecodecs
import numpy as np
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


RUN_NAME = "native-media"


class _MediaCases:
    """Encode, independently reopen, and write compact native media examples."""

    pixels: np.ndarray
    summary: summary_pb2.Summary

    def __init__(self) -> None:
        y, x = np.mgrid[:12, :16]
        self.pixels = np.stack((x * 11, y * 19, (x + y) * 7), axis=-1).astype(np.uint8)
        self.summary = summary_pb2.Summary()

    def _add_image(self, tag: str, blob: bytes) -> None:
        image = self.summary.value.add(tag=f"legacy-{tag}").image
        image.width, image.height, image.colorspace = 16, 12, 3
        image.encoded_image_string = blob
        tensor = self.summary.value.add(tag=f"tf2-{tag}")
        tensor.metadata.plugin_data.plugin_name = "images"
        tensor.tensor.dtype = types_pb2.DT_STRING
        tensor.tensor.tensor_shape.dim.add(size=3)
        tensor.tensor.string_val.extend((b"16", b"12", blob))

    def make_images(self) -> None:
        """Encode all image formats and check the decoded sample pixels."""
        for tag in ("jp2", "j2k"):
            blob = imagecodecs.jpeg2k_encode(
                self.pixels, codecformat=tag.upper(), reversible=True
            )
            np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(blob), self.pixels)
            self._add_image(tag, blob)
        for container in (False, True):
            blob = imagecodecs.jpegxl_encode(
                self.pixels, lossless=True, usecontainer=container, numthreads=1
            )
            np.testing.assert_array_equal(imagecodecs.jpegxl_decode(blob), self.pixels)
            self._add_image("jxl-container" if container else "jxl-codestream", blob)
        hdr = self.pixels.astype(np.float32) / 128
        blob = imagecodecs.exr_encode(hdr)
        np.testing.assert_array_equal(imagecodecs.exr_decode(blob), hdr)
        self._add_image("exr", blob)

    def make_audio(self, codec: str, suffix: str) -> None:
        """Encode 0.24 seconds of A4 and verify decoded frequency and amplitude."""
        rate, frames = 48000, 11520
        pcm = np.rint(12000 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate))
        frame = av.AudioFrame.from_ndarray(pcm.astype(np.int16)[None, :], layout="mono")
        frame.sample_rate = rate
        stream = av.CodecContext.create(codec, "w")
        stream.sample_rate, stream.layout, stream.format = rate, "mono", "s16"
        stream.bit_rate = 128000
        packets = stream.encode(frame) + stream.encode(None)
        blob = b"".join(bytes(packet) for packet in packets)
        with av.open(BytesIO(blob), format="mp3") as source:
            decoded = np.concatenate(
                [f.to_ndarray().ravel() for f in source.decode(audio=0)]
            )
        middle = decoded[2400:7200].astype(np.float64)
        crossings = np.count_nonzero((middle[:-1] <= 0) & (middle[1:] > 0))
        peak = np.max(np.abs(middle))
        if decoded.dtype.kind == "i":
            peak /= np.iinfo(decoded.dtype).max
        if abs(crossings - 44) > 1 or not 0.3 < peak < 0.4:
            raise ValueError(f"{suffix}: decoded signal is not the expected A4 tone")
        legacy = self.summary.value.add(tag=f"legacy-{suffix}").audio
        legacy.sample_rate, legacy.num_channels, legacy.length_frames = rate, 1, frames
        legacy.content_type, legacy.encoded_audio_string = "audio/mpeg", blob
        tensor = self.summary.value.add(tag=f"tf2-{suffix}")
        tensor.metadata.plugin_data.plugin_name = "audio"
        tensor.tensor.dtype = types_pb2.DT_STRING
        tensor.tensor.tensor_shape.dim.add(size=1)
        tensor.tensor.tensor_shape.dim.add(size=2)
        tensor.tensor.string_val.extend((blob, b"A4"))

    def write(self, output: Path) -> None:
        """Write the summary event after encoding all image and audio samples."""
        event = event_pb2.Event(wall_time=1.0, step=1, summary=self.summary)
        with (output / f"events.out.tfevents.{RUN_NAME}").open("xb") as handle:
            writer = RecordWriter(handle)
            writer.write(
                event_pb2.Event(file_version="brain.Event:2").SerializeToString()
            )
            writer.write(event.SerializeToString())


def main() -> None:
    """Generate native media summaries into an empty output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    cases = _MediaCases()
    cases.make_images()
    cases.make_audio("mp2", "mp2")
    cases.make_audio("libmp3lame", "mp3")
    cases.write(output)


if __name__ == "__main__":
    main()
