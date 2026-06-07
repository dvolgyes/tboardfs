import tomllib
from pathlib import Path


FIXTURE_SCRIPT_DEPENDENCIES = (
    "imageio>=2.37.3,<3.0",
    "matplotlib>=3.10.9,<4.0",
    "moviepy>=2.2.1,<3.0",
    "numpy>=2.4.6,<3.0",
    "pillow>=11.3.0,<12.0",
    "soundfile>=0.13.1,<1.0",
    "tensorboard>=2.20.0,<3.0",
    "tensorboardx>=2.6.5,<3.0",
    "torch>=2.12.0,<3.0",
)


def test_fixture_generation_script_uses_uv_pep723_metadata() -> None:
    """Fixture packages are declared by uv-compatible script metadata."""
    script_lines = Path("scripts/generate_test_logs.py").read_text().splitlines()
    metadata_end = script_lines.index("# ///", 2)
    metadata = tomllib.loads(
        "\n".join(line.removeprefix("# ") for line in script_lines[2:metadata_end])
    )

    assert script_lines[0] == "#!/usr/bin/env -S uv run --script"
    assert script_lines[1] == "# /// script"
    assert metadata["requires-python"] == ">=3.12"
    assert tuple(metadata["dependencies"]) == FIXTURE_SCRIPT_DEPENDENCIES


def test_dev_dependencies_do_not_include_fixture_generation_stack() -> None:
    """The project dev group contains test tooling, not fixture tooling."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dev_dependencies = {
        dependency.split(">=", maxsplit=1)[0].lower()
        for dependency in pyproject["dependency-groups"]["dev"]
    }
    fixture_dependency_names = {
        dependency.split(">=", maxsplit=1)[0].lower()
        for dependency in FIXTURE_SCRIPT_DEPENDENCIES
    }

    assert fixture_dependency_names.isdisjoint(dev_dependencies)


def test_runtime_dependencies_do_not_include_forbidden_protobuf_stacks() -> None:
    """Runtime dependencies avoid TensorBoard and TensorFlow packages."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    runtime_dependencies = {
        dependency.split(">=", maxsplit=1)[0].lower()
        for dependency in pyproject["project"]["dependencies"]
    }

    assert {"tensorboard", "tensorboardx", "tensorflow"}.isdisjoint(
        runtime_dependencies
    )
