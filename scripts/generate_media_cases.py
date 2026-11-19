#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorboard==2.21.0", "pillow==12.1.1",
#   "imageio-ffmpeg==0.6.0", "python-ffmpeg==2.0.12",
# ]
# ///
"""Write custom encoded media with TensorBoard's low-level event writer."""

import argparse
from importlib.resources import files
from io import BytesIO
from itertools import pairwise
import math
from pathlib import Path
import struct
import tempfile
import wave

from ffmpeg import FFmpeg
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class _MediaRun:
    """Build one event from real media encoded in a private working directory."""

    work: Path
    summary: summary_pb2.Summary
    encoder: str

    def __init__(self, work: Path) -> None:
        self.work = work
        self.summary = summary_pb2.Summary()
        binaries = files("imageio_ffmpeg").joinpath("binaries")
        (binary,) = (p for p in binaries.iterdir() if p.name.startswith("ffmpeg-"))
        self.encoder = str(binary)

    def images(self) -> None:
        """Add still images and compare GIF frame order after encoding."""
        red = Image.new("RGB", (16, 8), (255, 0, 0))
        blue = Image.new("RGB", red.size, (0, 0, 255))
        for tag in ("JPEG", "PNG", "GIF", "animated", "WEBP", "BMP", "TIFF", "AVIF"):
            encoded = BytesIO()
            if tag == "animated":
                red.save(
                    encoded, "GIF", save_all=True, append_images=[blue], duration=100
                )
            else:
                red.save(encoded, tag)
            image = self.summary.value.add(tag=tag.lower()).image
            image.width, image.height = red.size
            image.colorspace = 3
            image.encoded_image_string = encoded.getvalue()
            if tag in {"GIF", "animated"}:
                with Image.open(encoded) as decoded:
                    colors = []
                    for index in range(decoded.n_frames):
                        decoded.seek(index)
                        colors.append(decoded.convert("RGB").getpixel((0, 0)))
                expected = [(255, 0, 0)]
                if tag == "animated":
                    expected.append((0, 0, 255))
                if colors != expected:
                    raise ValueError(f"{tag}: unexpected frame sequence {colors}")
            if tag == "animated":
                self.image_batch("tensor-gif", [image.encoded_image_string])
        modes = []
        for mode in ("L", "LA", "RGB", "RGBA"):
            encoded = BytesIO()
            red.convert(mode).save(encoded, "PNG")
            modes.append(encoded.getvalue())
        self.image_batch("channels", modes)

    def image_batch(self, tag: str, payloads: list[bytes]) -> None:
        """Store encoded images in the image plugin's dimension-prefixed tensor."""
        batch = self.summary.value.add(tag=tag)
        batch.metadata.CopyFrom(image_metadata.create_summary_metadata(tag, ""))
        batch.tensor.dtype = types_pb2.DT_STRING
        batch.tensor.tensor_shape.dim.add(size=len(payloads) + 2)
        batch.tensor.string_val.extend((b"16", b"8", *payloads))

    @staticmethod
    def tone(channels: int, width: int, rate: int) -> bytes:
        """Encode a quarter-second A4, with an E5 on the right when stereo."""
        scale = 1 << (8 * width - 2)
        bias = 128 if width == 1 else 0
        pcm = bytearray()
        for frame in range(rate // 4):
            for hz in (440, 660)[:channels]:
                sample = round(scale * math.sin(2 * math.pi * hz * frame / rate))
                pcm.extend((sample + bias).to_bytes(width, "little", signed=width != 1))
        return bytes(pcm)

    def wav(self) -> None:
        """Write PCM clips in legacy fields and one unlabelled audio tensor."""
        batch = self.summary.value.add(tag="unlabelled")
        batch.metadata.CopyFrom(
            audio_metadata.create_summary_metadata(
                "unlabelled", "", audio_metadata.Encoding.WAV
            )
        )
        batch.tensor.dtype = types_pb2.DT_STRING
        for channels, width, rate in (
            (1, 1, 8000),
            (2, 2, 16000),
            (2, 3, 44100),
            (2, 4, 48000),
        ):
            output = BytesIO()
            frames = self.tone(channels, width, rate)
            with wave.open(output, "wb") as stream:
                stream.setparams(
                    (channels, width, rate, rate // 4, "NONE", "not compressed")
                )
                stream.writeframes(frames)
            audio = self.summary.value.add(tag=f"pcm{width * 8}").audio
            audio.sample_rate = rate
            audio.num_channels = channels
            audio.length_frames = rate // 4
            audio.content_type = "audio/wav"
            audio.encoded_audio_string = output.getvalue()
            batch.tensor.string_val.append(audio.encoded_audio_string)
        batch.tensor.tensor_shape.dim.add(size=len(batch.tensor.string_val))

    def compressed_audio(self) -> None:
        """Write compressed clips and a RIFF-wrapped MIDI note."""
        midi = bytes.fromhex(
            "4d54686400000006000000010060 4d54726b0000000c 00904560 30804500 00ff2f00"
        )
        chunk = b"RMIDdata" + len(midi).to_bytes(4, "little") + midi
        audio = self.summary.value.add(tag="riff-midi").audio
        audio.content_type = "audio/mid"
        audio.encoded_audio_string = b"RIFF" + len(chunk).to_bytes(4, "little") + chunk
        source = self.tone(1, 2, 16000)
        for suffix, codec, mime in (
            ("flac", "flac", "audio/flac"),
            ("mp3", "libmp3lame", "audio/mpeg"),
            ("ogg", "libvorbis", "audio/ogg"),
            ("webm", "libopus", "audio/webm"),
        ):
            path = self.work / f"sample.{suffix}"
            self.pipeline().input("pipe:0", f="s16le", ar=16000, ac=1).output(
                path, acodec=codec
            ).execute(source)
            self.verify_compressed_tone(path)
            audio = self.summary.value.add(tag=f"audio-{suffix}").audio
            audio.content_type = mime
            audio.sample_rate, audio.num_channels, audio.length_frames = 16000, 1, 4000
            audio.encoded_audio_string = path.read_bytes()

    def verify_compressed_tone(self, path: Path) -> None:
        """Check the duration and pitch after decoding at the original sample rate."""
        decoded = (
            self.pipeline().input(path).output("pipe:1", f="s16le", ar=16000).execute()
        )
        samples = struct.unpack(f"<{len(decoded) // 2}h", decoded)
        if not 3900 <= len(samples) <= 4600:
            raise ValueError(f"{path.name}: unexpected decoded duration")
        middle = samples[1000:3000]
        crossings = sum(left <= 0 < right for left, right in pairwise(middle))
        if abs(crossings * 8 - 440) > 8 or max(middle) < 8192:
            raise ValueError(f"{path.name}: expected a clearly audible 440-Hz tone")

    def pipeline(self) -> FFmpeg:
        """Start an encoding or decoding pipeline with the bundled FFmpeg."""
        return FFmpeg(executable=self.encoder).option("loglevel", "error")

    def videos(self) -> None:
        """Encode two frames per container and verify their decoded colors."""
        pixels = 32 * 32
        frames = (b"\xff\0\0" * pixels) + (b"\0\0\xff" * pixels)
        for suffix, codec in (
            ("mp4", "mpeg4"),
            ("mov", "mpeg4"),
            ("avi", "mpeg4"),
            ("webm", "libvpx"),
            ("mkv", "mpeg4"),
            ("ogg", "libtheora"),
        ):
            path = self.work / f"clip.{suffix}"
            self.pipeline().input(
                "pipe:0", f="rawvideo", pix_fmt="rgb24", s="32x32", r=2
            ).output(path, vcodec=codec).execute(frames)
            decoded = (
                self.pipeline()
                .input(path)
                .output("pipe:1", f="rawvideo", pix_fmt="rgb24")
                .execute()
            )
            if len(decoded) != len(frames) or any(
                abs(actual - expected) > 16
                for actual, expected in zip(decoded, frames, strict=True)
            ):
                raise ValueError(
                    f"{suffix}: expected one red frame followed by one blue frame"
                )
            value = self.summary.value.add(tag=suffix)
            value.metadata.plugin_data.plugin_name = "video"
            value.tensor.dtype = types_pb2.DT_UINT8
            value.tensor.tensor_content = path.read_bytes()
            value.tensor.tensor_shape.dim.add(size=len(value.tensor.tensor_content))

    def write(self, output: Path) -> None:
        """Write all media cases into one TensorBoard event file."""
        self.images()
        self.wav()
        self.compressed_audio()
        self.videos()
        writer = EventFileWriter(str(self.work))
        try:
            writer.add_event(event_pb2.Event(step=1, wall_time=1, summary=self.summary))
        finally:
            writer.close()
        (source,) = self.work.glob("events.out.tfevents.*")
        source.replace(output / "events.out.tfevents.media")


def main() -> None:
    """Generate one compact event file in a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        _MediaRun(Path(temporary)).write(output)


if __name__ == "__main__":
    main()
