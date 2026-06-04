from pathlib import Path
import re
from typing import Any

from click.testing import CliRunner

from tboardfs import cli


def test_mount_cli_defaults_to_background(tmp_path: Path, monkeypatch: Any) -> None:
    """The FUSE mount command runs in background mode by default."""
    source = tmp_path / "runs"
    mountpoint = tmp_path / "mnt"
    source.mkdir()
    mountpoint.mkdir()
    monkeypatch.chdir(tmp_path)
    fuse_calls: list[dict[str, Any]] = []

    def capture_fuse(*args: Any, **kwargs: Any) -> None:
        """Capture FUSE keyword arguments without mounting."""
        filesystem = args[0]
        fuse_calls.append(
            {
                "source": filesystem.source,
                "mountpoint": args[1],
                "foreground": kwargs["foreground"],
                "ro": kwargs["ro"],
            }
        )

    monkeypatch.setattr(cli, "FUSE", capture_fuse)

    result = CliRunner().invoke(cli.main, ["runs", "mnt"])

    assert result.exit_code == 0
    assert fuse_calls == [
        {
            "source": source,
            "mountpoint": str(mountpoint),
            "foreground": False,
            "ro": True,
        }
    ]
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
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| INFO \|",
        result.stderr,
    )
    assert "background mode is usually preferred" in result.stderr
    assert "foreground mode is recommended only for testing" in result.stderr


def test_mount_cli_foreground_logs_when_parsing_finishes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Foreground mode reports when event-file parsing is complete."""
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()

    def capture_fuse(*args: Any, **kwargs: Any) -> None:
        """Trigger the first FUSE lookup without mounting."""
        del kwargs
        filesystem = args[0]
        filesystem.readdir("/")

    monkeypatch.setattr(cli, "FUSE", capture_fuse)

    result = CliRunner().invoke(
        cli.main, ["test-logs", str(mountpoint), "--foreground", "--step-digits", "3"]
    )

    assert result.exit_code == 0
    assert "finished parsing TensorBoard files" in result.stderr
