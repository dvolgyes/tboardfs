"""Check byte-exact exports and PCM samples in small RF64 and BW64 audio files."""

import math
import struct
import unittest
from pathlib import Path

from tboardfs import TensorBoardFS
from tboardfs.classify import detect_extension
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

RUN = Path(__file__).parent / "data/producer-cases/wave64"


class Wave64Cases(unittest.TestCase):
    def test_wave64_recordwriter_round_trip(self) -> None:
        with (RUN / "events.out.tfevents.wave64").open("rb") as handle:
            values = {
                value.tag: value
                for frame in iter_tfrecords(handle)
                for value in compat_pb2.Event.FromString(frame.payload).summary.value
            }
        fs = TensorBoardFS(RUN)
        expected_samples = tuple(
            round(12000 * math.sin(math.tau * 440 * i / 8000)) for i in range(800)
        )
        for name in ("rf64", "bw64"):
            legacy = values[f"legacy-{name}"].audio
            self.assertEqual(
                (legacy.num_channels, legacy.sample_rate, legacy.length_frames),
                (1, 8000, 800),
            )
            plugin = values[f"tf2-{name}"]
            self.assertEqual(plugin.metadata.plugin_data.plugin_name, "audio")
            self.assertEqual(
                [dim.size for dim in plugin.tensor.tensor_shape.dim], [1, 2]
            )
            encoded, label = plugin.tensor.string_val
            self.assertEqual(label, f"440 Hz {name.upper()}".encode())
            self.assertEqual(legacy.encoded_audio_string, encoded)
            for tag in (f"legacy-{name}", f"tf2-{name}"):
                with self.subTest(tag=tag):
                    path = f"/audio/{tag}/000007.wav"
                    blob = fs.read(path, len(encoded) + 1, 0)
                    self.assertEqual(blob, encoded)
                    self.assertEqual(
                        blob[:16], name.upper().encode() + b"\xff" * 4 + b"WAVEds64"
                    )
                    self.assertEqual(
                        struct.unpack_from("<IQQQI", blob, 16),
                        (28, len(blob) - 8, 1600, 800, 0),
                    )
                    self.assertEqual(
                        struct.unpack_from("<4sIHHIIHH", blob, 48),
                        (b"fmt ", 16, 1, 1, 8000, 16000, 2, 16),
                    )
                    self.assertEqual(blob[72:80], b"data" + b"\xff" * 4)
                    self.assertEqual(
                        struct.unpack("<800h", blob[80:]), expected_samples
                    )

    def test_wave64_requires_wave_form_and_complete_ds64(self) -> None:
        for form in (b"RF64", b"BW64"):
            prefix = form + b"\xff" * 4
            for tail in (b"AVI ds64", b"WAVEJUNK", b"WAVEds64\x1c\0\0\0" + b"\0" * 27):
                with self.subTest(form=form, tail=tail):
                    self.assertEqual(detect_extension(prefix + tail), "bin")
