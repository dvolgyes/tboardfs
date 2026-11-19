import os
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


GENERATORS = sorted((ROOT / "scripts").glob("generate_*.py"))


@pytest.mark.parametrize("path", GENERATORS, ids=lambda path: path.name)
def test_generators_are_standalone_executables(path: Path) -> None:
    """The operating system and uv can start each generator from its metadata."""
    lines = path.read_text().splitlines()
    assert os.access(path, os.X_OK)
    assert lines[:2] == ["#!/usr/bin/env -S uv run --script", "# /// script"]
    end = lines.index("# ///", 2)
    assert all(line == "#" or line.startswith("# ") for line in lines[2:end])
    metadata = tomllib.loads("\n".join(line[1:].lstrip(" ") for line in lines[2:end]))
    assert metadata["requires-python"]
    assert isinstance(metadata["dependencies"], list)


def test_project_does_not_install_summary_writers() -> None:
    """Installing the package or its tests must not install the producer stack."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    specs = [*project["project"]["dependencies"], *project["dependency-groups"]["dev"]]
    names = {
        re.split("[^a-zA-Z0-9_.-]", spec, maxsplit=1)[0].lower().replace("_", "-")
        for spec in specs
    }
    assert names.isdisjoint(
        {"tensorboard", "tensorboardx", "tensorflow", "tensorflow-cpu", "torch"}
    )
