#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pillow==12.1.1", "pillow-heif==1.1.1",
#   "tensorboard==2.21.0",
# ]
# ///
"""Generate small HEVC and AV1 still images in HEIF containers."""

import argparse
from io import BytesIO
from pathlib import Path

import pillow_heif
from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.summary.writer.record_writer import RecordWriter

SIZE = (32, 16)


def _ftyp_brands(blob: bytes) -> tuple[bytes, tuple[bytes, ...]]:
    """Return the major and compatible brands from a valid ftyp box."""
    size = int.from_bytes(blob[:4], "big")
    if blob[4:8] != b"ftyp" or size < 16 or size > len(blob) or size % 4:
        raise ValueError("invalid HEIF ftyp box")
    return blob[8:12], tuple(blob[index : index + 4] for index in range(16, size, 4))


def _image() -> Image.Image:
    """Create four broad color panels resilient to lossy chroma subsampling."""
    result = Image.new("RGB", SIZE)
    colors = ((240, 24, 24), (24, 220, 32), (24, 32, 240), (230, 210, 20))
    for index, color in enumerate(colors):
        x, y = index % 2 * 16, index // 2 * 8
        result.paste(color, (x, y, x + 16, y + 8))
    return result


def main() -> None:
    """Write the verified codec fixtures to the requested output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    pillow_heif.register_heif_opener()
    source = _image()
    summary = summary_pb2.Summary()
    for name, fmt in (("heic", "HEIF"), ("avif", "AVIF")):
        expected_brand = name.encode()
        buffer = BytesIO()
        source.save(buffer, format=fmt, quality=90)
        blob = buffer.getvalue()
        major, compatible = _ftyp_brands(blob)
        if major != expected_brand or expected_brand not in compatible:
            raise ValueError(f"{name}: unexpected brands {major!r} {compatible!r}")
        with Image.open(buffer, formats=[fmt]) as decoded:
            if decoded.size != SIZE:
                raise ValueError(f"{name}: unexpected decoded size {decoded.size}")
            rgb = decoded.convert("RGB")
            for xy in ((8, 4), (24, 4), (8, 12), (24, 12)):
                if (
                    max(
                        abs(a - b)
                        for a, b in zip(rgb.getpixel(xy), source.getpixel(xy))
                    )
                    > 20
                ):
                    raise ValueError(f"{name}: unexpected panel color at {xy}")
        value = summary.value.add(tag=f"{name}-image")
        value.image.width, value.image.height = SIZE
        value.image.colorspace = 3
        value.image.encoded_image_string = blob
        tensor = summary.value.add(tag=f"{name}-tensor")
        tensor.metadata.CopyFrom(image_metadata.create_summary_metadata(tensor.tag, ""))
        tensor.tensor.dtype = types_pb2.DT_STRING
        tensor.tensor.tensor_shape.dim.add(size=3)
        tensor.tensor.string_val.extend((b"32", b"16", blob))

    with (output / "events.out.tfevents.heic").open("xb") as handle:
        writer = RecordWriter(handle)
        for event in (
            event_pb2.Event(file_version="brain.Event:2"),
            event_pb2.Event(wall_time=1.0, step=1, summary=summary),
        ):
            writer.write(event.SerializeToString())


if __name__ == "__main__":
    main()
