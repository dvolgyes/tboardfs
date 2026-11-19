#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorboard==2.21.0", "pillow==12.1.1", "numpy==2.4.3", "numpngw==0.1.4",
#   "imageio-ffmpeg==0.6.0", "python-ffmpeg==2.0.12",
# ]
# ///
"""Generate frozen native raster payloads from maintained image writers.

PNG format: https://www.w3.org/TR/png-3/
Pillow image writing: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
NumPNGw writing: https://github.com/WarrenWeckesser/numpngw

These are custom encoded image payloads in TensorBoard image summaries. They do
not represent tf.summary.image's high-level PNG encoder output.
"""

import argparse
from importlib.resources import files
from io import BytesIO
from pathlib import Path

from ffmpeg import FFmpeg
from numpngw import write_png
import numpy as np
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class _RasterCases:
    """Encode genuine raster variants and write them as image summaries."""

    summary: summary_pb2.Summary

    def __init__(self) -> None:
        self.summary = summary_pb2.Summary()

    def pngs(self) -> None:
        """Write all legal PNG colour type and bit depth combinations."""
        variants: list[tuple[str, int, int, list[int]]] = [
            ("png-gray-1", 1, 0, [0, 1]),
            ("png-gray-2", 2, 0, [0, 1, 2, 3]),
            ("png-gray-4", 4, 0, list(range(16))),
            ("png-gray-8", 8, 0, [0, 64, 128, 255]),
            ("png-gray-16", 16, 0, [0, 16384, 32768, 65535]),
            ("png-rgb-8", 8, 2, [255, 0, 0, 0, 255, 0, 0, 0, 255]),
            ("png-rgba-8", 8, 6, [0, 20, 40, 255, 60, 80, 100, 128]),
            ("png-gray-alpha-8", 8, 4, [0, 255, 255, 128]),
            ("png-rgb-16", 16, 2, [65535, 0, 0, 0, 65535, 0, 0, 0, 65535]),
            ("png-rgba-16", 16, 6, [0, 2000, 4000, 65535, 6000, 8000, 10000, 32768]),
            ("png-gray-alpha-16", 16, 4, [0, 65535, 65535, 32768]),
        ]
        for tag, depth, color, pixels in variants:
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
            width = len(pixels) // channels
            encoded = BytesIO()
            shape = (1, width) if channels == 1 else (1, width, channels)
            data = np.array(
                pixels, dtype=np.uint16 if depth == 16 else np.uint8
            ).reshape(shape)
            write_png(encoded, data, bitdepth=depth if depth < 8 else None)
            self._image(tag, encoded.getvalue(), width, 1, channels)

        colors = np.array(
            [[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]], dtype=np.uint8
        )
        for depth in (1, 2, 4, 8):
            encoded = BytesIO()
            width = min(2**depth, 4)
            write_png(encoded, colors[:, :width], use_palette=True, bitdepth=depth)
            self._image(f"png-palette-{depth}", encoded.getvalue(), width, 1, 3)

    def jpegs(self) -> None:
        """Write baseline/progressive grayscale, RGB, and CMYK JPEG streams."""
        variants: list[tuple[str, str, int | tuple[int, ...]]] = [
            ("gray", "L", 80),
            ("rgb", "RGB", (255, 0, 0)),
            ("cmyk", "CMYK", (0, 128, 255, 0)),
        ]
        for color, mode, pixels in variants:
            image = Image.new(mode, (16, 8), pixels)
            for scan, progressive in (("baseline", False), ("progressive", True)):
                encoded = BytesIO()
                image.save(encoded, format="JPEG", quality=88, progressive=progressive)
                self._image(
                    f"jpeg-{color}-{scan}",
                    encoded.getvalue(),
                    16,
                    8,
                    len(image.getbands()),
                )

    def animations(self) -> None:
        """Store distinct two-frame GIF, APNG, and animated WebP byte streams."""
        frames = [Image.new("RGB", (16, 8), color) for color in ("red", "blue")]
        for image_format in ("GIF", "PNG", "WEBP"):
            encoded = BytesIO()
            frames[0].save(
                encoded,
                format=image_format,
                save_all=True,
                append_images=frames[1:],
                duration=[120, 240],
                loop=0,
                lossless=True,
            )
            with Image.open(encoded) as decoded:
                actual = []
                for index in range(decoded.n_frames):
                    decoded.seek(index)
                    actual.append(
                        (decoded.convert("RGB").tobytes(), decoded.info["duration"])
                    )
            if actual != [(frames[0].tobytes(), 120), (frames[1].tobytes(), 240)]:
                raise ValueError(f"{image_format}: frames or durations changed")
            payload = encoded.getvalue()
            if image_format == "GIF":
                binaries = files("imageio_ffmpeg").joinpath("binaries")
                (binary,) = (
                    p for p in binaries.iterdir() if p.name.startswith("ffmpeg-")
                )
                decoded_rgb = (
                    FFmpeg(executable=str(binary))
                    .option("loglevel", "error")
                    .input("pipe:0")
                    .output("pipe:1", vf="fps=25", f="rawvideo", pix_fmt="rgb24")
                    .execute(payload)
                )
                if decoded_rgb != frames[0].tobytes() * 3 + frames[1].tobytes() * 6:
                    raise ValueError(
                        "GIF: independent decoder did not recover the timed RGB frames"
                    )
            tag = "apng" if image_format == "PNG" else image_format.lower()
            self._image(f"animated-{tag}", payload, 16, 8, 3)

    def _image(
        self, tag: str, payload: bytes, width: int, height: int, channels: int
    ) -> None:
        image = self.summary.value.add(tag=tag).image
        image.width = width
        image.height = height
        image.colorspace = channels
        image.encoded_image_string = payload

    def write(self, output: Path) -> None:
        """Write generated summaries into one TensorBoard event file."""
        event = event_pb2.Event(step=1, wall_time=1.0)
        event.summary.CopyFrom(self.summary)
        writer = EventFileWriter(str(output))
        writer.add_event(event)
        writer.flush()
        writer.close()
        generated = next(output.glob("events.out.tfevents.*"))
        generated.replace(output / "events.out.tfevents.rasters")


def main() -> None:
    """Generate raster summaries in a new or empty output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    cases = _RasterCases()
    cases.pngs()
    cases.jpegs()
    cases.animations()
    cases.write(output)


if __name__ == "__main__":
    main()
