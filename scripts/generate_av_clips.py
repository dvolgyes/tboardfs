#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorboard==2.21.0", "imageio-ffmpeg==0.6.0", "python-ffmpeg==2.0.12"]
# ///
"""Write moving, synchronized audiovisual clips in custom video summaries."""

import argparse
import json
import math
import struct
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

from ffmpeg import FFmpeg
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

FORMATS = (
    ("mp4", "libx264", "aac"),
    ("mov", "libx264", "aac"),
    ("avi", "mpeg4", "pcm_s16le"),
    ("webm", "libvpx-vp9", "libopus"),
    ("mkv", "libx264", "flac"),
    ("ogg", "libtheora", "libvorbis"),
)
WIDTH, HEIGHT, FPS, RATE = 96, 48, 8, 48000


def _inputs(work: Path) -> None:
    """A yellow square moves along a blue rail and crosses its white midpoint at 0.5 s."""
    frames = bytearray()
    for index in range(FPS):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if 4 + index * 11 <= x < 16 + index * 11 and 16 <= y < 28:
                    color = (248, 216, 32)
                elif 44 <= x < 52 and 34 <= y < 42:
                    color = (232, 232, 232)
                elif 28 <= y < 34:
                    color = (32, 112, 208)
                else:
                    color = (16, 24, 40)
                frames.extend(color)
    (work / "frames.rgb").write_bytes(frames)
    samples = [
        round(
            14000 * math.sin(2 * math.pi * (440 if i < RATE // 2 else 880) * i / RATE)
        )
        for i in range(RATE)
    ]
    (work / "tone.pcm").write_bytes(struct.pack(f"<{RATE}h", *samples))


@dataclass(frozen=True)
class _Clip:
    """Measure decoded data, independently of the summary and its filesystem path."""

    path: Path
    encoder: str

    def decoder(self) -> FFmpeg:
        """Open this encoded file with the bundled decoder."""
        return (
            FFmpeg(executable=self.encoder).option("loglevel", "error").input(self.path)
        )

    @staticmethod
    def pixel(frame: bytes, x: int, y: int) -> bytes:
        """Read one RGB pixel from a decoded frame."""
        offset = 3 * (y * WIDTH + x)
        return frame[offset : offset + 3]

    def center(self, frame: bytes, index: int) -> float:
        """Locate the moving square and verify the stationary rail and midpoint."""

        bright = []
        for x in range(WIDTH):
            r, g, b = self.pixel(frame, x, 22)
            if r > 180 and g > 160 and b < 100:
                bright.append(x)
        if len(bright) < 10 or abs(sum(bright) / len(bright) - (9.5 + 11 * index)) > 1:
            raise ValueError(
                f"{self.path}: square displaced at frame {index}: {bright}"
            )
        if min(self.pixel(frame, 48, 38)) < 190 or self.pixel(frame, 3, 31)[2] < 160:
            raise ValueError(f"{self.path}: missing stationary midpoint or rail")
        return sum(bright) / len(bright)

    def centers(self) -> list[float]:
        """Decode and locate all eight frames without synthesizing repeats."""
        rgb: bytes = (
            self.decoder()
            .output(
                "pipe:1",
                map="0:v:0",
                f="rawvideo",
                pix_fmt="rgb24",
                fps_mode="passthrough",
            )
            .execute()
        )
        size = WIDTH * HEIGHT * 3
        if len(rgb) != FPS * size:
            raise ValueError(
                f"{self.path}: unexpected decoded frame count {len(rgb) / size}"
            )
        return [self.center(rgb[i * size : (i + 1) * size], i) for i in range(FPS)]

    def pitch(self, index: int, window: tuple[int, ...]) -> float:
        """Measure the tone inside its video frame, away from codec edge ringing."""
        rising = [i for i in range(1, len(window)) if window[i - 1] <= 0 < window[i]]
        if len(rising) < 2 or max(window) < 11000 or min(window) > -11000:
            raise ValueError(
                f"{self.path}: silent or attenuated audio at frame {index}"
            )
        pitch = RATE * (len(rising) - 1) / (rising[-1] - rising[0])
        if abs(pitch - (440 if index < 4 else 880)) > 2:
            raise ValueError(f"{self.path}: wrong pitch {pitch} at frame {index}")
        return round(pitch, 2)

    def measurements(self) -> dict[str, str | int | list[float]]:
        """Allow an AAC final access unit or Vorbis overlap at the end of the audio."""
        pcm = (
            self.decoder()
            .output("pipe:1", map="0:a:0", f="s16le", ac=1, ar=RATE)
            .execute()
        )
        count = len(pcm) // 2
        allowance = {".mp4": 1024, ".mov": 1024, ".ogg": 128}.get(self.path.suffix, 0)
        if abs(count - RATE) > allowance:
            raise ValueError(f"{self.path}: unexpected sample count {count}")
        samples = struct.unpack(f"<{count}h", pcm)
        pitches = []
        for index in range(FPS):
            start = index * RATE // FPS + RATE // 80
            pitches.append(self.pitch(index, samples[start : start + RATE // 10]))
        return {
            "sha256": sha256(self.path.read_bytes()).hexdigest(),
            "centers": self.centers(),
            "pitch_hz": pitches,
            "samples": count,
        }


def _write(output: Path, work: Path) -> None:
    """Encode each container and retain its decode oracle alongside the summary."""
    binaries = files("imageio_ffmpeg").joinpath("binaries")
    (encoder,) = (p for p in binaries.iterdir() if p.name.startswith("ffmpeg-"))
    _inputs(work)
    summary = summary_pb2.Summary()
    measurements = {}
    for suffix, video, audio in FORMATS:
        path = work / f"crossing.{suffix}"
        pipeline = FFmpeg(executable=str(encoder)).option("loglevel", "error")
        pipeline.input(
            work / "frames.rgb",
            f="rawvideo",
            pix_fmt="rgb24",
            s=f"{WIDTH}x{HEIGHT}",
            r=FPS,
        ).input(work / "tone.pcm", f="s16le", ar=RATE, ac=1).output(
            path, vcodec=video, acodec=audio, pix_fmt="yuv420p", threads=1
        ).execute()
        measurements[suffix] = _Clip(path, str(encoder)).measurements()
        value = summary.value.add(tag=suffix)
        value.metadata.plugin_data.plugin_name = "video"
        value.tensor.dtype = types_pb2.DT_UINT8
        value.tensor.tensor_content = path.read_bytes()
        value.tensor.tensor_shape.dim.add(size=len(value.tensor.tensor_content))
    writer = EventFileWriter(str(work))
    try:
        writer.add_event(event_pb2.Event(step=8, wall_time=1, summary=summary))
    finally:
        writer.close()
    (source,) = work.glob("events.out.tfevents.*")
    source.replace(output / "events.out.tfevents.av-clips")
    (output / "decoded.json").write_text(json.dumps(measurements, indent=2) + "\n")


def main() -> None:
    """Write fixtures to a caller-selected empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        _write(output, Path(temporary))


if __name__ == "__main__":
    main()
