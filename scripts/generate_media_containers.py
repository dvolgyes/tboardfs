#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0", "imageio-ffmpeg==0.6.0",
#   "python-ffmpeg==2.0.12", "pillow==12.1.1",
#   "av==19.0.2", "numpy==2.5.4",
# ]
# ///
"""Encode and reopen compact images, audio, and moving media streams."""

import argparse
import json
import struct
import tempfile
from hashlib import sha256
from importlib.resources import files
from math import pi, sin
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree

import av
from ffmpeg import FFmpeg
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

_WIDTH = 48
_HEIGHT = 32
_RATE = 48000
_CENTERS = (8, 16, 24, 32, 40)
_RED = (235, 32, 24)
_BLUE = (16, 40, 176)
_SVG = "http://www.w3.org/2000/svg"


class _MediaCases:
    """Encode files and verify decoded content before writing summary values."""

    work: Path
    summary: summary_pb2.Summary
    executable: str
    digests: dict[str, str]

    def __init__(self, work: Path) -> None:
        self.work = work
        self.summary = summary_pb2.Summary()
        self.digests = {}
        binaries = files("imageio_ffmpeg").joinpath("binaries")
        (encoder,) = (
            path for path in binaries.iterdir() if path.name.startswith("ffmpeg-")
        )
        self.executable = str(encoder)

    def ffmpeg(self) -> FFmpeg:
        """Use the pinned, bundled encoder and decoder executable."""
        return FFmpeg(executable=self.executable).option("loglevel", "error")

    def add(self, path: Path, kind: str) -> None:
        """Store original bytes in the appropriate TensorBoard summary field."""
        blob = path.read_bytes()
        self.digests[path.stem] = sha256(blob).hexdigest()
        value = self.summary.value.add(tag=path.stem)
        if kind == "images":
            value.image.encoded_image_string = blob
            value.image.width = _HEIGHT if path.suffix == ".ico" else _WIDTH
            value.image.height = _HEIGHT
        elif kind == "audio":
            value.audio.encoded_audio_string = blob
            value.audio.content_type = f"audio/{path.suffix[1:]}"
            value.audio.sample_rate = _RATE
            value.audio.num_channels = 1
            value.audio.length_frames = _RATE // 5
        else:
            value.metadata.plugin_data.plugin_name = "video"
            value.tensor.dtype = types_pb2.DT_UINT8
            value.tensor.tensor_content = blob
            value.tensor.tensor_shape.dim.add(size=len(blob))

    @staticmethod
    def frame(center: int) -> bytes:
        """Make a blue field with a red marker at a known horizontal position."""
        return b"".join(
            bytes(_RED if abs(x - center) < 5 and 8 <= y < 24 else _BLUE)
            for y in range(_HEIGHT)
            for x in range(_WIDTH)
        )

    @staticmethod
    def decoded_rgb(path: Path) -> bytes:
        """Decode original frames with PyAV's independently bundled FFmpeg."""
        with av.open(path) as source:
            return b"".join(
                frame.to_ndarray(format="rgb24").tobytes()
                for frame in source.decode(video=0)
            )

    def images(self) -> None:
        """Cross-check QOI and both icon backing formats, plus RGBE and opaque SVG."""
        frame = self.frame(_CENTERS[2])
        source = self.work / "source.rgb"
        source.write_bytes(frame)
        qoi = self.work / "qoi.qoi"
        hdr = self.work / "hdr.hdr"
        for path in (qoi, hdr):
            (
                self.ffmpeg()
                .input(source, f="rawvideo", pix_fmt="rgb24", s=f"{_WIDTH}x{_HEIGHT}")
                .output(path, vframes=1, threads=1)
                .execute()
            )
        with Image.open(qoi) as reopened:
            if reopened.format != "QOI" or reopened.tobytes() != frame:
                raise ValueError("QOI did not round-trip through Pillow")
        self.add(qoi, "images")
        self.check_motion(hdr, (_CENTERS[2],))
        self.add(hdr, "images")
        self.icons(frame)
        self.svg()

    def icons(self, frame: bytes) -> None:
        """Write and independently decode PNG-backed and DIB-backed ICO files."""
        icon = Image.frombytes("RGB", (_WIDTH, _HEIGHT), frame).crop(
            (0, 0, _HEIGHT, _HEIGHT)
        )
        for backing in ("png", "bmp"):
            path = self.work / f"ico-{backing}.ico"
            icon.save(path, sizes=[icon.size], bitmap_format=backing)
            if self.decoded_rgb(path) != icon.tobytes():
                raise ValueError(f"{path.name}: ICO did not round-trip through PyAV")
            self.add(path, "images")

    def svg(self) -> None:
        """Write a namespaced vector graphic and verify its opaque XML structure."""
        path = self.work / "svg.svg"
        root = ElementTree.Element(
            "svg", xmlns=_SVG, width=str(_WIDTH), height=str(_HEIGHT)
        )
        ElementTree.SubElement(root, "rect", width="48", height="32", fill="#1028b0")
        ElementTree.SubElement(root, "circle", cx="24", cy="16", r="6", fill="#eb2018")
        ElementTree.ElementTree(root).write(
            path, encoding="utf-8", xml_declaration=True
        )
        document = minidom.parseString(path.read_bytes())  # noqa: S318
        svg = document.documentElement
        circle = document.getElementsByTagNameNS(_SVG, "circle")[0]
        if svg is None or svg.namespaceURI != _SVG or circle.getAttribute("r") != "6":
            raise ValueError("SVG namespace or circle geometry changed")
        self.add(path, "images")

    def check_motion(self, path: Path, centers: tuple[int, ...]) -> None:
        """Find the marker in every original frame, without synthesized repeats."""
        decoded = self.decoded_rgb(path)
        size = _WIDTH * _HEIGHT * 3
        if len(decoded) != len(centers) * size:
            raise ValueError(f"{path.name}: unexpected decoded frame count")
        for index, center in enumerate(centers):
            start = index * size + (_HEIGHT // 2) * _WIDTH * 3
            row = decoded[start : start + _WIDTH * 3]
            actual = [x for x in range(_WIDTH) if row[3 * x] > row[3 * x + 2] + 50]
            if len(actual) < 7 or abs(sum(actual) / len(actual) - center) > 1:
                raise ValueError(
                    f"{path.name}: marker lost or displaced in frame {index}"
                )

    def videos(self) -> None:
        """Write both program-stream versions, TS, ASF, and an elementary-stream control."""
        source = self.work / "frames.rgb"
        source.write_bytes(b"".join(map(self.frame, _CENTERS)))
        for tag, suffix, container, codec in (
            ("program-v1", "mpg", "mpeg", "mpeg1video"),
            ("program-v2", "mpg", "vob", "mpeg2video"),
            ("transport", "ts", "mpegts", "mpeg2video"),
            ("windows-media", "asf", "asf", "wmv2"),
            ("elementary", "m2v", "mpeg2video", "mpeg2video"),
        ):
            path = self.work / f"{tag}.{suffix}"
            (
                self.ffmpeg()
                .input(
                    source, f="rawvideo", pix_fmt="rgb24", s=f"{_WIDTH}x{_HEIGHT}", r=25
                )
                .output(path, f=container, vcodec=codec, pix_fmt="yuv420p", threads=1)
                .execute()
            )
            self.check_motion(path, _CENTERS)
            self.add(path, "videos")

    def audio(self) -> None:
        """Encode and measure a playable tone in CAF, Ogg, and WebM."""
        source = self.work / "tone.pcm"
        count = _RATE // 5
        tone = (round(12000 * sin(2 * pi * 440 * i / _RATE)) for i in range(count))
        source.write_bytes(struct.pack(f"<{count}h", *tone))
        for suffix, codec in (
            ("caf", "pcm_s16le"),
            ("ogg", "libopus"),
            ("webm", "libopus"),
        ):
            path = self.work / f"{suffix}.{suffix}"
            (
                self.ffmpeg()
                .input(source, f="s16le", ar=_RATE, ac=1)
                .output(path, acodec=codec, f=suffix)
                .execute()
            )
            self.check_tone(path, count)
            self.add(path, "audio")

    def check_tone(self, path: Path, count: int) -> None:
        """Reopen the container and measure sample count, amplitude, and pitch."""
        options = {} if path.suffix == ".caf" else {"codec": "opus"}
        decoder = self.ffmpeg().input(path, **options)
        pcm = decoder.output("pipe:1", map="0:a:0", f="s16le", ac=1, ar=_RATE).execute()
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        if abs(len(samples) - count) > 312 or not 10000 < max(samples) < 15000:
            raise ValueError(f"{path.name}: unexpected duration or signal amplitude")
        rising = [
            i for i in range(480, count - 480) if samples[i - 1] <= 0 < samples[i]
        ]
        if (
            len(rising) < 2
            or abs(_RATE * (len(rising) - 1) / (rising[-1] - rising[0]) - 440) > 2
        ):
            raise ValueError(f"{path.name}: decoded tone is not 440 Hz")

    def save(self, output: Path) -> None:
        """Write the verified opaque payloads and the digest oracle."""
        writer = EventFileWriter(str(self.work))
        try:
            writer.add_event(event_pb2.Event(step=1, wall_time=1, summary=self.summary))
        finally:
            writer.close()
        (source,) = self.work.glob("events.out.tfevents.*")
        source.replace(output / "events.out.tfevents.media-containers")
        (output / "sha256.json").write_text(
            json.dumps(self.digests, indent=2, sort_keys=True) + "\n"
        )


def main() -> None:
    """Create fixtures in a caller-selected empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        cases = _MediaCases(Path(temporary))
        cases.images()
        cases.videos()
        cases.audio()
        cases.save(output)


if __name__ == "__main__":
    main()
