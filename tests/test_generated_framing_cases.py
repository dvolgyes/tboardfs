"""Exercise standalone framing fixtures through strict and live readers."""

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.file_cli import main
from tboardfs.tfrecord import iter_tfrecords

GENERATOR = Path(__file__).resolve().parents[1] / "scripts/generate_framing_cases.py"
EXTRACTOR = GENERATOR.with_name("extract_tfrecord_prefix.py")
PREFIX = "events.out.tfevents."
SCALAR = "/scalars/loss.json"


@pytest.fixture
def cases(tmp_path: Path) -> Path:
    output = tmp_path / "cases"
    subprocess.run(
        [sys.executable, "-I", GENERATOR, "--output", output], cwd=tmp_path, check=True
    )
    return output


def test_strict_framing(cases: Path) -> None:
    runner = CliRunner()
    valid = cases / (PREFIX + "adjacent-valid")
    result = runner.invoke(main, ["get", str(valid), SCALAR, "-o", "-"])
    assert result.exit_code == 0, result.stderr
    rows = json.loads(result.stdout_bytes)
    assert [(r["step"], r["value"]) for r in rows] == [(1, 1.25), (2, 2.5)]

    for case, error in (
        ("bad-length-crc", "length checksum mismatch"),
        ("bad-payload-crc", "payload checksum mismatch"),
        ("valid-prefix-incomplete-tail", "truncated TFRecord payload"),
        ("header-only-over-ceiling", "exceeds maximum"),
    ):
        path = cases / (PREFIX + case)
        result = runner.invoke(main, ["list", str(path)])
        assert result.exit_code == 1 and not result.stdout_bytes
        assert str(path) in result.stderr and error in result.stderr


def test_live_framing(cases: Path) -> None:
    valid = cases / (PREFIX + "adjacent-valid")
    incomplete = cases / (PREFIX + "valid-prefix-incomplete-tail")
    assert (cases / (PREFIX + "header-only-over-ceiling")).stat().st_size == 12
    fs = TensorBoardFS(cases)
    rows = json.loads(fs.read(SCALAR, 65536, 0))
    assert [(r["step"], r["value"]) for r in rows] == [(1, 1.25), (1, 1.25), (2, 2.5)]

    with incomplete.open("ab") as stream:
        stream.write(valid.read_bytes()[-2:])
    rows = json.loads(fs.read(SCALAR, 65536, 0))
    assert [(r["step"], r["value"]) for r in rows] == [
        (1, 1.25),
        (1, 1.25),
        (2, 2.5),
        (2, 2.5),
    ]


def test_generator_refuses_nonempty_output(tmp_path: Path) -> None:
    marker = tmp_path / "keep"
    marker.write_bytes(b"original")
    result = subprocess.run(
        [sys.executable, "-I", GENERATOR, "--output", tmp_path],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0 and b"not empty" in result.stderr
    assert list(tmp_path.iterdir()) == [marker] and marker.read_bytes() == b"original"


def _extract(
    source: Path, output: Path, limit: int
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            EXTRACTOR,
            "--source",
            source,
            "--output",
            output,
            "--max-bytes",
            str(limit),
        ],
        cwd=output.parent,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_extract_copies_whole_frames(
    cases: Path, tmp_path: Path, extra_bytes: int
) -> None:
    inner_frames = (cases / (PREFIX + "adjacent-valid")).read_bytes()
    length = struct.pack("<Q", len(inner_frames))
    outer_frame = (
        length
        + struct.pack("<I", masked_crc32c(length))
        + inner_frames
        + struct.pack("<I", masked_crc32c(inner_frames))
    )
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.write_bytes(outer_frame + inner_frames)
    result = _extract(source, output, len(outer_frame) + extra_bytes)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == outer_frame
    assert source.read_bytes() == outer_frame + inner_frames


@pytest.mark.parametrize(
    ("limit_base", "extra_bytes"),
    [("prefix", 1), ("prefix", 16), ("eof", 0), ("eof", 1)],
)
def test_extract_trailing_partial_frame(
    cases: Path, tmp_path: Path, limit_base: str, extra_bytes: int
) -> None:
    source = cases / (PREFIX + "valid-prefix-incomplete-tail")
    output = tmp_path / "output"
    with source.open("rb") as stream:
        prefix_end = next(iter_tfrecords(stream)).total_size
    data = source.read_bytes()
    limit = (prefix_end if limit_base == "prefix" else len(data)) + extra_bytes
    result = _extract(source, output, limit)
    if limit_base == "prefix":
        assert result.returncode == 0, result.stderr
        assert output.read_bytes() == data[:prefix_end]
    else:
        assert (
            result.returncode != 0 and b"incomplete TFRecord payload" in result.stderr
        )
        assert list(tmp_path.iterdir()) == [cases]


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    [
        ("bad-length-crc", b"invalid TFRecord length checksum"),
        ("bad-payload-crc", b"invalid TFRecord payload checksum"),
    ],
)
def test_extract_leaves_no_partial_output(
    cases: Path, tmp_path: Path, case: str, diagnostic: bytes
) -> None:
    source = cases / (PREFIX + case)
    output = tmp_path / "output"
    result = _extract(source, output, source.stat().st_size)
    assert result.returncode != 0 and diagnostic in result.stderr
    assert not output.exists()


def test_extract_refuses_existing_output(cases: Path, tmp_path: Path) -> None:
    source = cases / (PREFIX + "adjacent-valid")
    output = tmp_path / "output"
    output.write_bytes(b"keep")
    result = _extract(source, output, source.stat().st_size)
    assert result.returncode != 0
    assert output.read_bytes() == b"keep"
