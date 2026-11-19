"""Read embedded files from the TensorBoard corpus without the producer stack."""

from io import BytesIO
import struct
from zipfile import ZipFile

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from test_tf_producer_cases import CORPUS, _events


@pytest.fixture(scope="module")
def run():
    name = "embedded-formats"
    values = {v.tag: v for event in _events(name) for v in event.summary.value}
    return TensorBoardFS(CORPUS / name), values


def test_standard_midi_notes(run):
    fs, values = run
    blob = fs.read("/audio/midi/000007.mid", 4096, 0)
    assert blob == values["midi"].audio.encoded_audio_string
    assert struct.unpack_from(">4sIHHH4sI", blob) == (b"MThd", 6, 0, 1, 96, b"MTrk", 28)
    for i, pitch in enumerate((60, 64, 67)):
        assert tuple(blob[22 + i * 8 : 30 + i * 8]) == (
            0,
            144,
            pitch,
            100,
            96,
            128,
            pitch,
            0,
        )
    assert blob[46:] == b"\x00\xff\x2f\x00"


def test_tiff_directory_chain_and_samples(run):
    fs, values = run
    blob = fs.read("/images/multipage-tiff/000007.tiff", 4096, 0)
    assert blob == values["multipage-tiff"].image.encoded_image_string
    endian, magic, offset = struct.unpack_from(">2sHI", blob)
    assert (endian, magic) == (b"MM", 42)
    for corners in [(0, 15, 2100, 2115), (12000, 12075, 7100, 7175)]:
        count = struct.unpack_from(">H", blob, offset)[0]
        tags = {}
        for i in range(count):
            tag, kind, n, value = struct.unpack_from(
                ">HHI4s", blob, offset + 2 + 12 * i
            )
            if kind in (3, 4) and n == 1:
                tags[tag] = struct.unpack_from(">H" if kind == 3 else ">I", value)[0]
        assert (tags[256], tags[257], tags[258], tags[259]) == (16, 8, 16, 1)
        plane = struct.unpack_from(">128H", blob, tags[273])
        assert (plane[0], plane[15], plane[112], plane[127]) == corners
        offset = struct.unpack_from(">I", blob, offset + 2 + 12 * count)[0]
    assert offset == 0


@pytest.mark.parametrize(
    "tag,suffix", [("matrix-npy", "npy"), ("plain-zip", "zip"), ("numeric-npz", "npz")]
)
def test_generic_tensor_files(run, tag, suffix):
    fs, values = run
    (payload,) = values[tag].tensor.string_val
    assert len(payload) >= 1024
    blob = fs.read(f"/tensors/{tag}/000007.{suffix}", len(payload) + 1, 0)
    assert blob == payload
    if tag == "matrix-npy":
        matrix = np.load(BytesIO(blob), allow_pickle=False)
        assert matrix.dtype.str == ">i4" and matrix.flags.f_contiguous
        rows, cols = np.indices((16, 24))
        np.testing.assert_array_equal(matrix, 16 * cols + rows - 170)
    elif tag == "numeric-npz":
        with np.load(BytesIO(blob), allow_pickle=False) as archive:
            assert archive.files == ["values"]
            np.testing.assert_array_equal(archive["values"], range(512))
    else:
        with ZipFile(BytesIO(blob)) as archive:
            assert archive.namelist() == ["squares.csv"]
            rows = archive.read("squares.csv").splitlines()
            assert len(rows) == 160 and rows[159] == b"159,25281"


def test_numeric_tensor_and_embedded_file_are_distinct(run):
    fs, values = run
    payload = values["wrapped-npy"].tensor.tensor_content
    assert payload == values["matrix-npy"].tensor.string_val[0]
    directory = "/tensors/wrapped-npy"
    with BytesIO(fs.read(f"{directory}/000007.npy", 65536, 0)) as encoded:
        outer = np.load(encoded, allow_pickle=False)
    assert outer.dtype == np.uint8 and outer.shape == (len(payload),)
    assert outer.tobytes() == payload
    assert fs.read(f"{directory}/000007.embedded.npy", len(payload) + 1, 0) == payload
