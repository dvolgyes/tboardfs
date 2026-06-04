from pathlib import Path
from typing import Any

from click.testing import CliRunner

from tboardfs import cli


def test_mount_cli_defaults_to_background(tmp_path: Path, monkeypatch: Any) -> None:
    """The FUSE mount command runs in background mode by default."""
    source = tmp_path / "runs"
    mountpoint = tmp_path / "mnt"
    source.mkdir()
    mountpoint.mkdir()
    fuse_calls: list[dict[str, Any]] = []

    def capture_fuse(*args: Any, **kwargs: Any) -> None:
        """Capture FUSE keyword arguments without mounting."""
        del args
        fuse_calls.append(kwargs)

    monkeypatch.setattr(cli, "FUSE", capture_fuse)

    result = CliRunner().invoke(cli.main, [str(source), str(mountpoint)])

    assert result.exit_code == 0
    assert fuse_calls == [{"foreground": False, "ro": True}]
    assert result.stderr == ""


def test_mount_cli_foreground_warns_that_background_is_preferred(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Foreground mode tells users it is mainly intended for tests."""
    source = tmp_path / "runs"
    mountpoint = tmp_path / "mnt"
    source.mkdir()
    mountpoint.mkdir()
    fuse_calls: list[dict[str, Any]] = []

    def capture_fuse(*args: Any, **kwargs: Any) -> None:
        """Capture FUSE keyword arguments without mounting."""
        del args
        fuse_calls.append(kwargs)

    monkeypatch.setattr(cli, "FUSE", capture_fuse)

    result = CliRunner().invoke(
        cli.main, [str(source), str(mountpoint), "--foreground"]
    )

    assert result.exit_code == 0
    assert fuse_calls == [{"foreground": True, "ro": True}]
    assert "background mode is usually preferred" in result.stderr
    assert "foreground mode is recommended only for testing" in result.stderr
