#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["tensorboard==2.21.0", "rasterio==1.5.2", "numpy==2.4.6"]
# ///
"""Generate tiny real GeoTIFFs inside TensorBoard image summaries."""

import argparse
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from tensorboard.compat.proto import event_pb2, summary_pb2, types_pb2
from tensorboard.plugins.image import metadata
from tensorboard.summary.writer.record_writer import RecordWriter

WIDTH, HEIGHT = 3, 2
TRANSFORM = from_origin(500_000, 4_650_000, 30, 30)


class _GeoTiff:
    """Encode and validate a small georeferenced elevation grid."""

    @staticmethod
    def payload(bigtiff: bool) -> bytes:
        """Reopen the encoded raster to verify its pixels and coordinates."""
        pixels = np.array([[11, 22, 33], [44, 55, 66]], dtype=np.uint16)
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=WIDTH,
                height=HEIGHT,
                count=1,
                dtype="uint16",
                crs="EPSG:32633",
                transform=TRANSFORM,
                compress="none",
                endianness="LITTLE",
                bigtiff="YES" if bigtiff else "NO",
            ) as dataset:
                dataset.write(pixels, 1)
            payload: bytes = memory.read()
        with MemoryFile(payload) as memory, memory.open() as dataset:
            if (
                dataset.driver != "GTiff"
                or dataset.width != WIDTH
                or dataset.height != HEIGHT
            ):
                raise ValueError("Rasterio did not reopen expected GeoTIFF dimensions")
            if dataset.crs.to_epsg() != 32633 or dataset.transform != TRANSFORM:
                raise ValueError(
                    "Rasterio did not reopen expected GeoTIFF georeferencing"
                )
            if not np.array_equal(dataset.read(1), pixels):
                raise ValueError("Rasterio did not reopen expected GeoTIFF pixels")
        version = (43 if bigtiff else 42).to_bytes(2, "little")
        if payload[:4] != b"II" + version:
            raise ValueError("unexpected TIFF header version")
        return payload

    @staticmethod
    def add_images(summary: summary_pb2.Summary, flavor: str, data: bytes) -> None:
        """Store the raster in both legacy and tensor image envelopes."""
        legacy = summary.value.add(tag=f"{flavor}-legacy").image
        legacy.width, legacy.height, legacy.colorspace = WIDTH, HEIGHT, 1
        legacy.encoded_image_string = data
        tag = f"{flavor}-tensor"
        value = summary.value.add(tag=tag)
        value.metadata.CopyFrom(metadata.create_summary_metadata(tag, ""))
        value.tensor.dtype = types_pb2.DT_STRING
        value.tensor.tensor_shape.dim.add(size=3)
        value.tensor.string_val.extend(
            (str(WIDTH).encode(), str(HEIGHT).encode(), data)
        )


def main() -> None:
    """Write one reproducible event file into a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    output = parser.parse_args().output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = summary_pb2.Summary()
    for flavor, big in (("classic", False), ("bigtiff", True)):
        _GeoTiff.add_images(summary, flavor, _GeoTiff.payload(big))
    destination = output / "events.out.tfevents.geotiff"
    with destination.open("xb") as handle:
        writer = RecordWriter(handle)
        for event in (
            event_pb2.Event(file_version="brain.Event:2", wall_time=1),
            event_pb2.Event(step=1, wall_time=2, summary=summary),
        ):
            writer.write(event.SerializeToString(deterministic=True))


if __name__ == "__main__":
    main()
