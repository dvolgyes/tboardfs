import tomllib
from pathlib import Path


FIXTURE_SCRIPT_DEPENDENCIES = (
    "imageio",
    "matplotlib",
    "moviepy",
    "pillow",
    "soundfile",
    "tensorboard",
    "tensorboardx",
    "torch",
)


def test_fixture_generation_dependencies_live_in_script_metadata() -> None:
    """Fixture-only packages are declared by the generator script."""
    script = Path("scripts/generate_test_logs.py").read_text()

    assert "# Script Dependencies:" in script
    for dependency in FIXTURE_SCRIPT_DEPENDENCIES:
        assert f"#   {dependency}" in script.lower()


def test_dev_dependencies_do_not_include_fixture_generation_stack() -> None:
    """The project dev group contains test tooling, not fixture tooling."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dev_dependencies = {
        dependency.split(">=", maxsplit=1)[0].lower()
        for dependency in pyproject["dependency-groups"]["dev"]
    }

    assert set(FIXTURE_SCRIPT_DEPENDENCIES).isdisjoint(dev_dependencies)
