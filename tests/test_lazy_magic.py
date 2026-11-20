"""Exercise the native classifier boundary in a fresh CLI process."""

from pathlib import Path
import subprocess
import sys

import pytest

from test_parse_file import _record

from tboardfs.proto import compat_pb2


def test_magic_loads_only_when_the_cli_needs_mime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAGIC", raising=False)
    scalar_dir = tmp_path / "scalars"
    scalar_dir.mkdir()
    scalar = compat_pb2.Event(step=1)
    scalar.summary.value.add(tag="loss", simple_value=0.5)
    (scalar_dir / "events.out.tfevents.scalar").write_bytes(
        _record(scalar.SerializeToString())
    )

    source = tmp_path / "events.out.tfevents.svg"
    event = compat_pb2.Event(step=1)
    image = event.summary.value.add(tag="drawing").image
    image.encoded_image_string = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
        b'<rect width="1" height="1"/></svg>'
    )
    source.write_bytes(_record(event.SerializeToString()))

    script = """\
import sys

import tboardfs

assert 'magic' not in sys.modules, 'package import loaded magic'
from tboardfs.file_cli import main

svg, scalars = sys.argv[1:]
main(['--framing-only', 'list', svg], standalone_mode=False)
assert 'magic' not in sys.modules, 'raw listing loaded magic'
main(['list', '--scalar-format', 'json', scalars], standalone_mode=False)
assert 'magic' not in sys.modules, 'scalar indexing loaded magic'
main(['list', '--step-digits', '3', svg], standalone_mode=False)
assert 'magic' in sys.modules, 'SVG classification did not load magic'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(scalar_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"/raw/{source.name}/00000000000000000000.tfrecord",
        "/scalars/loss.json",
        "/images/drawing/001.svg",
    ]
