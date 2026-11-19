"""Check scientific and mesh recognition against verified native fixtures."""

import json
import struct
from pathlib import Path

import pytest

from tboardfs.classify import detect_extension
from tboardfs.data_formats import data_extension

_DATA = Path(__file__).parent / "data" / "file-types" / "science-extra"


@pytest.mark.parametrize(
    "name",
    [
        "counts.mat",
        "compressed.mat",
        "depths.sqlite",
        "classic.nc",
        "offset64.nc",
        "data64.nc",
        "counts.safetensors",
        "depths.csv.gz",
        "depths.csv.bz2",
        "depths.csv.xz",
        "slope.stl",
        "slope.obj",
    ],
)
def test_producer_files(name: str) -> None:
    """Each native file reaches its own inferred suffix."""
    blob = (_DATA / name).read_bytes()
    suffix = name.rsplit(".", 1)[1]
    assert data_extension(blob) == suffix
    assert detect_extension(blob) == suffix


def test_netcdf4_is_hdf5() -> None:
    """HDF5 container evidence does not imply a netCDF-specific suffix."""
    blob = (_DATA / "netcdf4.nc").read_bytes()
    assert data_extension(blob) is None
    assert detect_extension(blob) == "h5"


@pytest.mark.parametrize(
    ("name", "size_width", "offset_width"),
    [("classic.nc", 4, 4), ("offset64.nc", 4, 8), ("data64.nc", 8, 8)],
)
def test_netcdf_fixed_extent(name: str, size_width: int, offset_width: int) -> None:
    """A fixed variable must advertise and contain its full data extent."""
    blob = (_DATA / name).read_bytes()
    offset_at = len(blob) - 12 - offset_width
    size_at = offset_at - size_width
    past_eof = (len(blob) + 4).to_bytes(offset_width, "big")
    assert detect_extension(blob[:-1]) != "nc"
    assert detect_extension(blob[:offset_at] + past_eof + blob[-12:]) != "nc"
    wrong_size = (4).to_bytes(size_width, "big")
    assert detect_extension(blob[:size_at] + wrong_size + blob[offset_at:]) != "nc"


def test_netcdf_empty_records() -> None:
    """The native writer leaves the second unallocated slab offset beyond EOF."""
    blob = (_DATA / "empty-records.nc").read_bytes()
    assert int.from_bytes(blob[-4:], "big") > len(blob)
    assert detect_extension(blob) == "nc"
    assert detect_extension(blob[:4] + (1).to_bytes(4, "big") + blob[8:]) != "nc"


@pytest.mark.parametrize(
    "blob",
    [
        b"MATLAB 5.0 MAT-file",
        b"SQLite format 3\0",
        b"CDF\x01",
        b"\x1f\x8b\x08",
        b"BZh9",
        b"\xfd7zXZ\0",
    ],
)
def test_bare_signatures(blob: bytes) -> None:
    """Short or bare signatures are not enough."""
    assert data_extension(blob) is None


@pytest.mark.parametrize(
    ("name", "offset", "replacement"),
    [
        ("counts.mat", 132, b"\xff" * 4),
        ("depths.sqlite", 16, b"\x0f\x00"),
        ("classic.nc", 12, b"\xff" * 4),
        ("depths.csv.gz", 3, b"\xe0"),
        ("depths.csv.bz2", 4, b"random"),
        ("depths.csv.xz", 8, b"\xff" * 4),
        ("slope.stl", 80, b"\x03\0\0\0"),
        ("slope.stl", 96, struct.pack("<f", float("nan"))),
    ],
)
def test_inconsistent_framing(name: str, offset: int, replacement: bytes) -> None:
    """Recognized headers do not excuse incompatible structural fields."""
    blob = (_DATA / name).read_bytes()
    assert (
        data_extension(blob[:offset] + replacement + blob[offset + len(replacement) :])
        is None
    )


@pytest.mark.parametrize(
    "blob",
    [
        b"# OBJ-like notes\nv 0 0 0\nv 1 0 0\nv 0 1 0\n",
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -2\n",
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/1 3/1\n",
        b"solid sample\nendsolid sample\n",
        b"\0" * 80 + b"\x01" + b"\0" * 53,
    ],
)
def test_mesh_ambiguity(blob: bytes) -> None:
    """Text fragments and shape-compatible empty records are not meshes."""
    assert data_extension(blob) is None


@pytest.mark.parametrize(
    "entries",
    [
        {"x": {"dtype": "unknown", "shape": [1], "data_offsets": [0, 4]}},
        {"x": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}},
        {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        {"x": {"dtype": "U8", "shape": [3], "data_offsets": [1, 4]}},
        {"x": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}},
        {
            "x": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
            "y": {"dtype": "I16", "shape": [1], "data_offsets": [2, 4]},
        },
        {
            "__metadata__": {"unit": 7},
            "x": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
        },
    ],
)
def test_safetensor_schema(entries: dict[str, object]) -> None:
    """Even plausible JSON must establish dtype, dimensions and exact buffer coverage."""
    header = json.dumps(entries).encode()
    assert (
        data_extension(len(header).to_bytes(8, "little") + header + b"\0" * 4) is None
    )


def test_safetensor_bounds() -> None:
    """Duplicate headers, excessive nesting and excessive header lengths are rejected."""
    entry = b'{"dtype":"I32","shape":[],"data_offsets":[0,4]}'
    for header in (
        b'{"x":' + entry + b',"x":' + entry + b"}",
        b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
        b'{"x":' + entry + b"}" + b" " * 65536,
    ):
        assert (
            data_extension(len(header).to_bytes(8, "little") + header + b"\0" * 4)
            is None
        )
