"""Check GeoTIFF image round trips without a GIS library."""

from pathlib import Path
import struct
import unittest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords


RUN = Path(__file__).parent / "data" / "producer-cases" / "geotiff"


def _numeric_tags(data: bytes, big: bool) -> dict[int, tuple]:
    """Read the numeric TIFF tags used by the two uncompressed fixtures."""
    offset_code, count_code = ("Q", "Q") if big else ("I", "H")
    slot_size = struct.calcsize("<" + offset_code)
    (ifd,) = struct.unpack_from("<" + offset_code, data, 8 if big else 4)
    (count,) = struct.unpack_from("<" + count_code, data, ifd)
    pos = ifd + struct.calcsize("<" + count_code)
    tags = {}
    for _ in range(count):
        tag, kind, n = struct.unpack_from("<HH" + offset_code, data, pos)
        slot = pos + 4 + slot_size
        if code := {3: "H", 4: "I", 12: "d", 16: "Q"}.get(kind):
            fmt = f"<{n}{code}"
            start = (
                slot
                if struct.calcsize(fmt) <= slot_size
                else struct.unpack_from("<" + offset_code, data, slot)[0]
            )
            tags[tag] = struct.unpack_from(fmt, data, start)
        pos = slot + slot_size
    return tags


class GeoTiffCasesTest(unittest.TestCase):
    def test_geotiff_image_round_trip(self) -> None:
        fs = TensorBoardFS(RUN)
        with (RUN / "events.out.tfevents.geotiff").open("rb") as handle:
            values = {
                v.tag: v
                for frame in iter_tfrecords(handle)
                for v in compat_pb2.Event.FromString(frame.payload).summary.value
            }
        for flavor in ("classic", "bigtiff"):
            legacy = values[f"{flavor}-legacy"].image
            self.assertEqual(
                (legacy.width, legacy.height, legacy.colorspace), (3, 2, 1)
            )
            plugin = values[f"{flavor}-tensor"]
            self.assertEqual(plugin.metadata.plugin_data.plugin_name, "images")
            self.assertEqual(
                plugin.tensor.string_val, [b"3", b"2", legacy.encoded_image_string]
            )
            for envelope in ("legacy", "tensor"):
                with self.subTest(flavor=flavor, envelope=envelope):
                    path = f"/images/{flavor}-{envelope}/000001.tiff"
                    data = fs.read(path, 4096, 0)
                    self.assertEqual(data, legacy.encoded_image_string)
                    big = flavor == "bigtiff"
                    self.assertTrue(
                        data.startswith(b"II+\0\x08\0\0\0" if big else b"II*\0")
                    )
                    tags = _numeric_tags(data, big)
                    self.assertEqual(
                        (tags[256], tags[257], tags[258], tags[259]),
                        ((3,), (2,), (16,), (1,)),
                    )
                    self.assertEqual(tags[33550], (30, 30, 0))
                    self.assertEqual(tags[33922], (0, 0, 0, 500000, 4650000, 0))
                    keys = tags[34735]
                    self.assertIn(
                        (3072, 0, 1, 32633),
                        [keys[i : i + 4] for i in range(4, len(keys), 4)],
                    )
                    self.assertEqual(tags[279], (12,))
                    self.assertEqual(
                        struct.unpack_from("<6H", data, tags[273][0]),
                        (11, 22, 33, 44, 55, 66),
                    )
