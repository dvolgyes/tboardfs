"""Write preserved producer records into the mixed scale event log."""

from collections.abc import Iterator
from pathlib import Path
import struct

from google.protobuf.message import Message

from tboardfs.crc32c import masked_crc32c
from tboardfs.model import Record
from tboardfs.proto_schema import _ProtoParser
from tboardfs.tfrecord import iter_tfrecords

_CASES = Path(__file__).resolve().parents[1] / "tests" / "data" / "producer-cases"
_SOURCES: tuple[tuple[str, tuple[int, ...] | None], ...] = (
    ("tensorboard", (1, 2)),
    ("tb115", (1, 2, 3, 4)),
    ("tf115-native", None),
    ("tf200-native", (2, 3, 4, 13, 14, 15)),
    ("historical-variants/tf200/dashboard", (1,)),
    ("historical-variants/tf200/function", (1,)),
    ("plugins", (6,)),
    ("lightning-matrix", (1, 2, 3, 7, 8, 9, 12, 13)),
    ("tensorboardx-matrix", (4,)),
    ("dtypes", None),
    ("tf2-plugin-formats", None),
    ("tf1", (1,)),
    ("tf2", (1, 2, 3, 4)),
    ("geometry-scene", None),
    ("open3d", None),
    ("open3d-advanced", None),
    ("encoded-tb115", None),
    ("rasters/moving-scenes", None),
    ("media", None),
    ("rasters", None),
    ("audio-variants", None),
    ("av-clips", None),
    ("embedded-formats", None),
    ("geotiff", None),
    ("heic", None),
    ("native-media", None),
    ("media-containers", None),
    ("wave64", None),
)


def _records(source: str) -> Iterator[tuple[Record, Message]]:
    """Read only explicitly named repository fixtures and verify their framing."""
    directory, _, name = source.rpartition("/")
    path = _CASES / (directory or name) / f"events.out.tfevents.{name}"
    with path.open("rb") as handle:
        for record in iter_tfrecords(handle):
            yield record, _ProtoParser.event(record.payload)


def captured_frames() -> Iterator[tuple[int, bytes]]:
    """Yield the original writer header, then selected frames in original step order.

    Ordinal selections avoid combining incompatible charts and HParams sessions.
    The historical loss row is removed so only the scale writer owns its tag.
    """
    frames: list[tuple[int, bytes]] = []
    for source, ordinals in _SOURCES:
        for index, (record, event) in enumerate(_records(source)):
            if event.HasField("file_version"):
                if source == "tensorboard":
                    yield 0, b"".join((record.header, record.payload, record.footer))
                continue
            if ordinals is not None and index not in ordinals:
                continue
            if source == "tb115" and index == 1:
                del getattr(event, "summary").value[0]
                payload = event.SerializeToString()
                length = struct.pack("<Q", len(payload))
                frame = (
                    length
                    + struct.pack("<I", masked_crc32c(length))
                    + payload
                    + struct.pack("<I", masked_crc32c(payload))
                )
            else:
                frame = b"".join((record.header, record.payload, record.footer))
            frames.append((getattr(event, "step"), frame))
    yield from sorted(frames, key=lambda item: item[0])
