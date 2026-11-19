#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0", "imageio-ffmpeg==0.6.0", "python-ffmpeg==2.0.12",
# ]
# ///
"""Generate compact, genuinely encoded audio format fixtures."""

import argparse
from importlib.resources import files
from itertools import pairwise
import math
from pathlib import Path
import struct
import tempfile
import wave

from ffmpeg import FFmpeg
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


FORMATS = (
    ("aac-adts", "aac", "audio/aac", "aac", None),
    ("aac-m4a", "m4a", "audio/mp4", "aac", "ipod"),
    ("alac-m4a", "m4a", "audio/mp4", "alac", "ipod"),
    ("float-wav", "wav", "audio/wav", "pcm_f32le", "wav"),
    ("pcm-aiff", "aiff", "audio/aiff", "pcm_s16be", "aiff"),
)
RATE = 24_000
FRAMES = RATE // 4
FREQUENCY = 440


def _pipeline(executable: str) -> FFmpeg:
    return FFmpeg(executable=executable).option("loglevel", "error")


def _verify_decode(executable: str, path: Path) -> None:
    """Decode output to float samples and verify its duration and 440-Hz tone."""
    decoded = (
        _pipeline(executable)
        .input(path)
        .output("pipe:1", f="f32le", ac=1, ar=RATE)
        .execute()
    )
    samples = struct.unpack(f"<{len(decoded) // 4}f", decoded)
    if not FRAMES <= len(samples) < FRAMES + 2048:
        raise ValueError(f"{path.name}: unexpected decoded length {len(samples)}")
    middle = samples[FRAMES // 3 : 2 * FRAMES // 3]
    crossings = sum(left <= 0 < right for left, right in pairwise(middle))
    measured = crossings * RATE / len(middle)
    if abs(measured - FREQUENCY) > 8:
        raise ValueError(f"{path.name}: decoded tone at {measured:.1f} Hz")


def main() -> None:
    """Build one event file with decoded-and-verified audio payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    binaries = files("imageio_ffmpeg").joinpath("binaries")
    (binary,) = (path for path in binaries.iterdir() if path.name.startswith("ffmpeg-"))
    executable = str(binary)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        work = Path(temporary)
        source = work / "tone.wav"
        pcm = b"".join(
            struct.pack(
                "<h", round(12_000 * math.sin(2 * math.pi * FREQUENCY * i / RATE))
            )
            for i in range(FRAMES)
        )
        with wave.open(str(source), "wb") as stream:
            stream.setparams((1, 2, RATE, FRAMES, "NONE", "not compressed"))
            stream.writeframes(pcm)
        summary = summary_pb2.Summary()
        for tag, suffix, mime, codec, muxer in FORMATS:
            path = work / f"tone.{tag}.{suffix}"
            output_args = {"acodec": codec}
            if muxer:
                output_args["f"] = muxer
            _pipeline(executable).input(source).output(path, **output_args).execute()
            _verify_decode(executable, path)
            audio = summary.value.add(tag=tag).audio
            audio.sample_rate = RATE
            audio.num_channels = 1
            audio.length_frames = FRAMES
            audio.content_type = mime
            audio.encoded_audio_string = path.read_bytes()
        writer = EventFileWriter(str(work))
        try:
            writer.add_event(event_pb2.Event(step=1, wall_time=1, summary=summary))
        finally:
            writer.close()
        (event_file,) = work.glob("events.out.tfevents.*")
        event_file.replace(output / "events.out.tfevents.audio-variants")


if __name__ == "__main__":
    main()
