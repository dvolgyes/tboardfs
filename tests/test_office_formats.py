"""Detect Office packages from their bytes without assuming a TensorBoard writer."""

from pathlib import Path

import pytest

from tboardfs.classify import detect_extension


@pytest.mark.parametrize("name", ["survey.docx", "readings.xlsx", "briefing.pptx"])
def test_standalone_office_package(name: str) -> None:
    path = Path(__file__).parent / "data" / "file-types" / name
    assert detect_extension(path.read_bytes()) == path.suffix[1:]
