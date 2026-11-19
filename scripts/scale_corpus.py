"""Write a deterministic, dense TFRecord workload for scale profiling."""

import struct
from hashlib import shake_256
from io import BytesIO
from math import tau
from pathlib import Path
from wave import open as open_wave
from zlib import compress, crc32

import numpy as np

from scripts.scale_breadth import captured_frames
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2

EVENT_NAME = "events.out.tfevents.scale"
IMAGES = "/images/series"
LOSS = "/scalars/loss.json"
LARGE = "/images/large/000000.png"
SIBLING = "/images/sibling/000000.png"
LARGE_WIDTH, LARGE_HEIGHT = 2048, 1536

BULK_IMAGE = "/images/bulk_image/000000.png"
BULK_AUDIO = "/audio/bulk_audio/000000.wav"
BULK_NUMERIC = "/tensors/bulk_numeric/000000.npy"
_IMAGE_SIDE = 512
_AUDIO_RATE = 48_000
_NOTE_FRAMES = 32_768
_AUDIO_FRAMES = 6 * _NOTE_FRAMES
_TENSOR_SHAPE = (384, 512)


def _frame(payload: bytes) -> bytes:
    """Frame one event with TFRecord length and payload checksums."""
    length = struct.pack("<Q", len(payload))
    return b"".join(
        (
            length,
            struct.pack("<I", masked_crc32c(length)),
            payload,
            struct.pack("<I", masked_crc32c(payload)),
        )
    )


class _ScaleCorpus:
    """Generate the original deterministic scalar and image workload."""

    @staticmethod
    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", crc32(kind + data))
        )

    @classmethod
    def _png(cls, width: int, height: int, rows: bytes) -> bytes:
        return (
            b"\x89PNG\r\n\x1a\n"
            + cls._chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + cls._chunk(b"IDAT", compress(rows, level=0))
            + cls._chunk(b"IEND", b"")
        )

    @classmethod
    def small_png(cls, pixel: int) -> bytes:
        """Encode a one-pixel RGB image."""
        return cls._png(1, 1, b"\0" + pixel.to_bytes(3, "big"))

    @classmethod
    def large_png(cls) -> bytes:
        """Encode an RGB gradient with uncompressed scanlines."""
        row = b"\0" + bytes(range(256)) * (LARGE_WIDTH * 3 // 256)
        return cls._png(LARGE_WIDTH, LARGE_HEIGHT, row * LARGE_HEIGHT)

    @classmethod
    def event_frame(
        cls, step: int, *, pixel: int | None = None, large: bytes | None = None
    ) -> bytes:
        """Write a scalar and image event as one checksummed frame."""
        event = getattr(compat_pb2, "Event")(step=step, wall_time=100.0 + step)
        event.summary.value.add(tag="loss", simple_value=step / 4)
        image = event.summary.value.add(tag="series").image
        image.width = image.height = 1
        image.colorspace = 3
        image.encoded_image_string = cls.small_png(step if pixel is None else pixel)
        if large is not None:
            image = event.summary.value.add(tag="large").image
            image.width, image.height, image.colorspace = LARGE_WIDTH, LARGE_HEIGHT, 3
            image.encoded_image_string = large
            image = event.summary.value.add(tag="sibling").image
            image.width = image.height = 1
            image.colorspace = 3
            image.encoded_image_string = cls.small_png(0xABCDEF)
        return _frame(event.SerializeToString())

    @classmethod
    def write(cls, path: Path, count: int) -> None:
        """Write one image and scalar per step, plus a large image and sibling."""
        with path.open("xb") as stream:
            stream.write(cls.event_frame(0, large=cls.large_png()))
            for step in range(1, count):
                stream.write(cls.event_frame(step))

    @staticmethod
    def image_path(step: int) -> str:
        """Return the virtual image path for a series step."""
        return f"{IMAGES}/{step:06d}.png"


class _BulkSamples:
    """Generate and frame synthetic images, audio and activation maps."""

    @staticmethod
    def _image(sample: int) -> bytes:
        """Encode a false-color field with changing bands and textured shading."""
        side = _IMAGE_SIDE
        texture = shake_256(str(sample).encode()).digest(side * side * 3)
        rgb = np.frombuffer(texture, dtype=np.uint8).reshape(side, side, 3)
        channels = np.arange(3)
        band = (
            np.arange(side)[:, None] + sample * (channels + 3) + channels * 43
        ) % 256
        level = np.minimum(band, 256 - band)
        rows = np.zeros((side, side * 3 + 1), dtype=np.uint8)
        rows[:, 1:] = (rgb // 2 + level[:, None, :]).reshape(side, side * 3)
        return _ScaleCorpus._png(side, side, rows.tobytes())

    @staticmethod
    def _audio(sample: int) -> bytes:
        """Synthesize a stereo phrase with changing pitch, panning and harmonics."""
        pcm = np.empty((6, _NOTE_FRAMES, 2), dtype=np.int16)
        positions = np.arange(_NOTE_FRAMES)
        ramp = 1_024
        edge_distance = np.minimum(positions, _NOTE_FRAMES - 1 - positions)
        envelope = np.minimum(edge_distance, ramp) / ramp
        pitches = (220, 262, 294, 330, 392, 440, 523, 587)
        for note in range(6):
            pitch = pitches[(sample + note * (2 + sample % 3)) % len(pitches)]
            phase = positions * (tau * pitch / _AUDIO_RATE)
            overtone = 500 + ((sample * 37 + note * 173) % 3_500)
            tone = (12_000 * np.sin(phase) + overtone * np.sin(2 * phase)) * envelope
            pan = (sample + note) % 5
            pcm[note] = tone[:, None] * np.array([(8 - pan) / 8, (4 + pan) / 8])
        stream = BytesIO()
        with open_wave(stream, "wb") as wave:
            wave.setparams((2, 2, _AUDIO_RATE, _AUDIO_FRAMES, "NONE", "not compressed"))
            wave.writeframes(memoryview(pcm).cast("B"))
        return stream.getvalue()

    @staticmethod
    def _numeric(sample: int) -> bytes:
        """Encode a float activation map with drifting row and column responses."""
        height, width = _TENSOR_SHAPE
        rows = np.arange(height)[:, None]
        cols = np.arange(width)
        bias = ((rows + sample * 7) % 127) / 8
        phase = (rows * (sample % 13 + 1) + sample * 11) % 256
        values: np.ndarray = (bias + ((cols + phase) % 256) / 4).astype("<f4")
        return memoryview(values).tobytes()

    @classmethod
    def first_media(cls) -> tuple[bytes, bytes]:
        """Return the PNG and WAV bytes for checking the first bulk reads."""
        return cls._image(0), cls._audio(0)

    @classmethod
    def frame(cls, step: int, sample: int, slot: int) -> bytes:
        """Frame one balanced triplet at a step without repeating its virtual paths."""
        suffix = f"/{slot}" if slot else ""
        event = getattr(compat_pb2, "Event")(step=step, wall_time=100.0 + step)
        image = event.summary.value.add(tag=f"bulk_image{suffix}").image
        image.width = image.height = _IMAGE_SIDE
        image.colorspace = 3
        image.encoded_image_string = cls._image(sample)
        audio = event.summary.value.add(tag=f"bulk_audio{suffix}").audio
        audio.sample_rate = _AUDIO_RATE
        audio.num_channels = 2
        audio.length_frames = _AUDIO_FRAMES
        audio.encoded_audio_string = cls._audio(sample)
        audio.content_type = "audio/wav"
        tensor = event.summary.value.add(tag=f"bulk_numeric{suffix}").tensor
        tensor.dtype = getattr(compat_pb2, "DT_FLOAT")
        for size in _TENSOR_SHAPE:
            tensor.tensor_shape.dim.add(size=size)
        tensor.tensor_content = cls._numeric(sample)
        return _frame(event.SerializeToString())


def write_mixed(path: Path, count: int, target_bytes: int) -> None:
    """Write the original small-entry workload plus breadth and bulk records.

    The captured file-version header precedes the original step-zero frame.
    Bulk triplets share the synthetic step range, leaving step `count` free for
    the benchmark append. Tiny runs retain the captured steps even if they
    exceed the synthetic range. A complete triplet can overshoot the target.
    """
    if not 0 < count < 1_000_000:
        raise ValueError("count must be between 1 and 999999")
    if target_bytes < 0:
        raise ValueError("target_bytes must not be negative")
    with path.open("xb") as stream:
        captures = captured_frames()
        _, header = next(captures)
        stream.write(header)
        captured = next(captures, None)
        stream.write(_ScaleCorpus.event_frame(0, large=_ScaleCorpus.large_png()))
        stream.write(_BulkSamples.frame(0, 0, 0))
        start = stream.tell()
        remaining = max(0, target_bytes - start)
        sample = 1
        for step in range(count):
            while captured is not None and (captured[0] <= step or step == count - 1):
                stream.write(captured[1])
                captured = next(captures, None)
            goal = start + remaining * (step + 1) // count
            slot = 1 if step == 0 else 0
            while stream.tell() < goal:
                stream.write(_BulkSamples.frame(step, sample, slot))
                sample += 1
                slot += 1
            if step:
                stream.write(_ScaleCorpus.event_frame(step))
