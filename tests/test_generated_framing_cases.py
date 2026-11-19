"""Exercise standalone framing fixtures through strict and live readers."""

import json
from pathlib import Path
import subprocess
import sys

from click.testing import CliRunner
import pytest

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main

GENERATOR = Path(__file__).resolve().parents[1] / "scripts/generate_framing_cases.py"
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
    )
    assert result.returncode != 0 and b"not empty" in result.stderr
    assert list(tmp_path.iterdir()) == [marker] and marker.read_bytes() == b"original"
