import os
from pathlib import Path
import re
from typing import Any

from click.testing import CliRunner
import pytest

from tboardfs import cli
from tboardfs.mounted import MountedTensorBoardFS


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
    assert fuse_calls == [
        {"foreground": True, "ro": True, "raw_fi": True, "attr_timeout": 0}
    ]
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


@pytest.mark.parametrize("mountpoint", ["runs", "runs/mnt", "runs/a/../mnt"])
def test_mount_cli_refuses_mountpoint_inside_source(
    tmp_path: Path, monkeypatch: Any, mountpoint: str
) -> None:
    """A mountpoint inside the source would make the scan read its own mount."""
    (tmp_path / "runs" / "a").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    fuse_calls: list[object] = []
    monkeypatch.setattr(cli, "FUSE", lambda *args, **kwargs: fuse_calls.append(args))

    result = CliRunner().invoke(cli.main, ["runs", mountpoint])

    assert result.exit_code == 2
    assert "inside the source" in result.stderr
    assert fuse_calls == []


def test_mount_cli_refuses_source_inside_mountpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The mount hides the source, so each read of it would call the mount."""
    (tmp_path / "mnt" / "runs").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    fuse_calls: list[object] = []
    monkeypatch.setattr(cli, "FUSE", lambda *args, **kwargs: fuse_calls.append(args))

    result = CliRunner().invoke(cli.main, ["mnt/runs", "mnt"])

    assert result.exit_code == 2
    assert "inside the mountpoint" in result.stderr
    assert fuse_calls == []


@pytest.mark.parametrize(
    ("option", "limit_bytes"),
    [
        ([], 1024 << 20),
        (["--cache-size-mb", "0"], 0),
        (["--cache-size-mb", "3"], 3 << 20),
    ],
)
def test_mount_cli_sets_the_render_cache_size(
    tmp_path: Path, monkeypatch: Any, option: list[str], limit_bytes: int
) -> None:
    """The render cache holds 1024 MB unless --cache-size-mb says otherwise."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "mnt").mkdir()
    monkeypatch.chdir(tmp_path)
    limits: list[int] = []
    monkeypatch.setattr(
        cli, "FUSE", lambda fs, *args, **kwargs: limits.append(fs._renders.limit_bytes)
    )

    result = CliRunner().invoke(cli.main, ["runs", "mnt", *option])

    assert result.exit_code == 0
    assert limits == [limit_bytes]


def test_mount_cli_refuses_a_negative_cache_size(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A negative cache size is a usage error, not a crash."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "mnt").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "FUSE", lambda *args, **kwargs: None)

    result = CliRunner().invoke(cli.main, ["runs", "mnt", "--cache-size-mb", "-1"])

    assert result.exit_code == 2
    assert "--cache-size-mb" in result.stderr
    assert "not in the range" in result.stderr


@pytest.mark.parametrize(
    ("option", "owner"),
    [
        ([], (0, 0)),
        (["--root-owner"], (0, 0)),
        (["--user-owner"], (os.getuid(), os.getgid())),
    ],
)
def test_mount_cli_sets_the_owner(
    tmp_path: Path, monkeypatch: Any, option: list[str], owner: tuple[int, int]
) -> None:
    """Files and directories belong to root unless --user-owner gives them to the user."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "events.out.tfevents.1").write_bytes(b"")
    (tmp_path / "mnt").mkdir()
    monkeypatch.chdir(tmp_path)
    owners: list[tuple[int, int]] = []

    def capture_owners(
        fs: MountedTensorBoardFS, *args: object, **kwargs: object
    ) -> None:
        del args, kwargs
        for path in ("/", "/scalars"):
            status = fs.getattr(path, None)
            owners.append((status["st_uid"], status["st_gid"]))

    monkeypatch.setattr(cli, "FUSE", capture_owners)

    result = CliRunner().invoke(cli.main, ["runs", "mnt", *option])

    assert result.exit_code == 0, result.output
    assert owners == [owner, owner]
