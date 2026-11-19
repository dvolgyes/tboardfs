"""Check scientific file inference without loading third-party producers."""

from pathlib import Path

import pytest

from tboardfs.classify import detect_extension

_DATA = Path(__file__).parent / "data" / "file-types" / "scientific"


@pytest.mark.parametrize(
    "name",
    [
        "terrain.glb",
        "ascii.ply",
        "little.ply",
        "big.ply",
        "survey.pdf",
        "profile.h5",
        "user-block.h5",
        "stations.parquet",
        "stations.arrow",
    ],
)
def test_scientific_file(name: str) -> None:
    """Real producer files keep their native suffixes."""
    assert detect_extension((_DATA / name).read_bytes()) == name.split(".")[-1]


def test_ply_ascii_header() -> None:
    """An ordinary CRLF-terminated text point cloud is still a PLY file."""
    blob = (
        b"ply\r\nformat ascii 1.0\r\nelement vertex 1\r\n"
        b"property float x\r\nend_header\r\n2.5\r\n"
    )
    assert detect_extension(blob) == "ply"


def test_non_native_fragments() -> None:
    """Headers, streams, prose and unframed data are not scientific files."""
    assert detect_extension(b'{"asset":{"version":"2.0"}}') == "json"
    for name in ("terrain.glb", "stations.arrow", "stations.parquet"):
        blob = (_DATA / name).read_bytes()
        assert detect_extension(blob[:-1]) != name.split(".")[-1]
    assert detect_extension((_DATA / "stations.stream").read_bytes()) != "arrow"
    assert detect_extension(b"ply\nplease send the report\nend_header\n") != "ply"
    assert detect_extension(b"%PDF-1.7\n") != "pdf"
    assert detect_extension(b"PAR1PAR1") != "parquet"
    for name, trailer in (("stations.parquet", 8), ("stations.arrow", 10)):
        blob = (_DATA / name).read_bytes()
        impossible = blob[:-trailer] + b"\xff" * 4 + blob[4 - trailer :]
        assert detect_extension(impossible) != name.split(".")[-1]
    hdf = (_DATA / "profile.h5").read_bytes()
    assert detect_extension(b"\0" + hdf) != "h5"
    assert detect_extension(hdf[:9]) != "h5"
