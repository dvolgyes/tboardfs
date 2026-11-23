import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from test_open_handles import _append

from tboardfs import cli, filesystem
from tboardfs.mounted import MountedTensorBoardFS

ROOT = Path(__file__).resolve().parent.parent


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

    monkeypatch.setattr(cli, "mount", capture_fuse)

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

    monkeypatch.setattr(cli, "mount", capture_fuse)

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


def test_foreground_reports_only_the_accessed_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Listing one run reports its parse without parsing sibling runs."""
    mountpoint = tmp_path / "mnt"
    mountpoint.mkdir()

    def capture_fuse(*args: Any, **kwargs: Any) -> None:
        """Trigger the first FUSE lookup without mounting."""
        del kwargs
        filesystem = args[0]
        assert any(
            name == "train" for name, _, _ in filesystem.readdir("/tensorboard/scalars")
        )

    monkeypatch.setattr(cli, "mount", capture_fuse)

    result = CliRunner().invoke(
        cli.main,
        [
            str(ROOT / "test-logs"),
            str(mountpoint),
            "--foreground",
            "--step-digits",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert str(ROOT / "test-logs/tensorboard/events.out.tfevents.") in result.stderr
    assert (
        str(ROOT / "test-logs/tensorboardx/events.out.tfevents.") not in result.stderr
    )


@pytest.mark.parametrize("mountpoint", ["runs", "runs/mnt", "runs/a/../mnt"])
def test_mount_cli_refuses_mountpoint_inside_source(
    tmp_path: Path, monkeypatch: Any, mountpoint: str
) -> None:
    """A mountpoint inside the source would make the scan read its own mount."""
    (tmp_path / "runs" / "a").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    fuse_calls: list[object] = []
    monkeypatch.setattr(cli, "mount", lambda *args, **kwargs: fuse_calls.append(args))

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
    monkeypatch.setattr(cli, "mount", lambda *args, **kwargs: fuse_calls.append(args))

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
        cli,
        "mount",
        lambda fs, *args, **kwargs: limits.append(fs._renders.limit_bytes),
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
    monkeypatch.setattr(cli, "mount", lambda *args, **kwargs: None)

    result = CliRunner().invoke(cli.main, ["runs", "mnt", "--cache-size-mb", "-1"])

    assert result.exit_code == 2
    assert "--cache-size-mb" in result.stderr
    assert "not in the range" in result.stderr


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
def test_mount_cli_refuses_invalid_index_workers(
    tmp_path: Path, monkeypatch: Any, value: str
) -> None:
    """Invalid worker counts are rejected before launching FUSE."""
    source = tmp_path / "runs"
    source.mkdir()
    fuse_calls: list[object] = []
    monkeypatch.setattr(cli, "mount", lambda *args, **kwargs: fuse_calls.append(args))

    result = CliRunner().invoke(
        cli.main, [str(source), str(tmp_path / "mnt"), "--index-workers", value]
    )

    assert result.exit_code == 2
    assert "--index-workers" in result.stderr
    assert fuse_calls == []


@pytest.mark.parametrize(
    ("option", "workers"), [([], 4), (["-j", "2"], 2), (["--index-workers", "3"], 3)]
)
def test_mount_cli_sets_index_workers(
    tmp_path: Path, monkeypatch: Any, option: list[str], workers: int
) -> None:
    """The mount defaults to four workers and accepts both option spellings."""
    source = tmp_path / "runs"
    source.mkdir()
    counts: list[int] = []
    monkeypatch.setattr(
        cli, "mount", lambda fs, *args, **kwargs: counts.append(fs.index_workers)
    )

    result = CliRunner().invoke(cli.main, [str(source), str(tmp_path / "mnt"), *option])

    assert result.exit_code == 0, result.output
    assert counts == [workers]


def test_filesystem_defaults_to_four_index_workers(tmp_path: Path) -> None:
    assert filesystem.TensorBoardFS(tmp_path).index_workers == 4


@pytest.mark.parametrize(
    ("seconds", "early_steps"),
    [("2.5", [0]), ("0", [0, 1])],
)
def test_mount_cli_source_check_interval_controls_visible_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seconds: str,
    early_steps: list[int],
) -> None:
    """An append appears when the configured source check is due."""
    source = tmp_path / "runs"
    source.mkdir()
    event = source / "events.out.tfevents"
    _append(event, 0)
    now = 0.0
    monkeypatch.setattr(filesystem, "_source_clock", lambda: now)
    path = "/scalars/loss.json"

    def mount_runner(fs: MountedTensorBoardFS, *args: Any, **kwargs: Any) -> None:
        nonlocal now

        def read_steps() -> list[int]:
            info = SimpleNamespace(flags=os.O_RDONLY, fh=0, direct_io=0)
            fs.open(path, info)
            try:
                return [row["step"] for row in json.loads(fs.read(path, 4096, 0, info))]
            finally:
                fs.release(path, info)

        assert "loss.json" in {name for name, _, _ in fs.readdir("/scalars")}
        assert read_steps() == [0]

        _append(event, 1)
        now = 2.0
        assert read_steps() == early_steps

        now = 2.5
        assert read_steps() == [0, 1]

    monkeypatch.setattr(cli, "mount", mount_runner)
    result = CliRunner().invoke(
        cli.main,
        [str(source), str(tmp_path / "mnt"), "--source-check-seconds", seconds],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output


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

    monkeypatch.setattr(cli, "mount", capture_owners)

    result = CliRunner().invoke(cli.main, ["runs", "mnt", *option])

    assert result.exit_code == 0, result.output
    assert owners == [owner, owner]


def test_mount_cli_reports_a_failed_mount_without_a_traceback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """libfuse gives the reason on stderr; mfusepy raises RuntimeError(1)."""
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(1)

    monkeypatch.setattr(cli, "mount", fail)

    result = CliRunner().invoke(cli.main, ["runs", "missing"])

    assert result.exit_code == 1
    assert "could not mount" in result.stderr
    assert "Traceback" not in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_mount_cli_writes_fuse_library_errors_to_the_logfile(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """mfusepy logs a failed operation with the logging module, with a traceback."""
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path)

    def fail_an_operation(*args: object, **kwargs: object) -> None:
        try:
            raise ValueError("bad operation")
        except ValueError:
            logging.getLogger("fuse").exception("Uncaught exception from %s", "read")

    monkeypatch.setattr(cli, "mount", fail_an_operation)

    result = CliRunner().invoke(cli.main, ["runs", "mnt", "--logfile", "log.txt"])

    assert result.exit_code == 0
    log = (tmp_path / "log.txt").read_text()
    assert "| ERROR | Uncaught exception from read" in log
    assert "ValueError: bad operation" in log
