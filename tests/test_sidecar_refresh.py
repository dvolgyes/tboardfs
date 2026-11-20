"""Sidecar files that appear, change or go away show at the next refresh."""

import errno
import os
from pathlib import Path

import pytest
from test_open_handles import _append

from tboardfs import TensorBoardFS

EVENTS = "events.out.tfevents.host"


def _fs_with_projector(tmp_path: Path) -> tuple[TensorBoardFS, Path]:
    _append(tmp_path / EVENTS, 0)
    projector = tmp_path / "projector_config.pbtxt"
    projector.write_text("config")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.readdir("/projector")[2:] == ["projector_config.pbtxt"]
    return fs, projector


def test_new_sidecar_appears_without_new_events(tmp_path: Path) -> None:
    """A profile written after the last event is listed."""
    fs, _ = _fs_with_projector(tmp_path)
    profile = tmp_path / "plugins" / "profile" / "2026_10_30"
    profile.mkdir(parents=True)
    (profile / "host.xplane.pb").write_bytes(b"xplane")

    assert fs.readdir("/profile/2026_10_30")[2:] == ["host.xplane.pb"]
    assert fs.read("/profile/2026_10_30/host.xplane.pb", 100, 0) == b"xplane"


def test_removed_sidecar_is_no_longer_listed(tmp_path: Path) -> None:
    """A deleted sidecar leaves the listing instead of failing on stat."""
    fs, projector = _fs_with_projector(tmp_path)

    projector.unlink()

    assert fs.readdir("/projector")[2:] == []


def test_changed_sidecar_shows_its_new_mtime(tmp_path: Path) -> None:
    """A rewritten sidecar reports the mtime of the file on disk."""
    fs, projector = _fs_with_projector(tmp_path)
    fs.getattr("/projector/projector_config.pbtxt")

    os.utime(projector, (1_000_000.0, 1_000_000.0))

    assert fs.getattr("/projector/projector_config.pbtxt")["st_mtime"] == 1_000_000.0


def test_unchanged_sidecars_keep_the_run_version(tmp_path: Path) -> None:
    """Refreshes with the same sidecars do not make the run render again."""
    fs, _ = _fs_with_projector(tmp_path)
    version = fs._state.runs[()].version

    fs.readdir("/projector")
    fs.readdir("/scalars")

    assert fs._state.runs[()].version == version


def test_hparams_yaml_of_a_run_is_in_the_hparams_tab(tmp_path: Path) -> None:
    """PyTorch Lightning writes hparams.yaml next to the event files of a run."""
    _append(tmp_path / EVENTS, 0)
    (tmp_path / "hparams.yaml").write_text("lr: 0.1\n")
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "hparams.yaml").write_text("lr: 0.2\n")
    fs = TensorBoardFS(tmp_path)

    assert fs.readdir("/hparams")[2:] == ["hparams.yaml"]
    assert fs.read("/hparams/hparams.yaml", 100, 0) == b"lr: 0.1\n"


def test_group_sidecars_change_owner_when_nested_run_appears(tmp_path: Path) -> None:
    _append(tmp_path / EVENTS, 0)
    group = tmp_path / "group"
    profile = group / "inner" / "profile"
    profile.mkdir(parents=True)
    (profile / "trace").write_bytes(b"trace")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/group/inner")
    assert fs.read("/profile/trace", 20, 0) == b"trace"

    event = group / EVENTS
    _append(event, 0)
    assert fs.read("/group/profile/trace", 20, 0) == b"trace"
    assert "trace" not in fs.readdir("/profile")
    event.unlink()
    fs.readdir("/group")
    assert fs.read("/profile/trace", 20, 0) == b"trace"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can search any folder")
@pytest.mark.parametrize(
    "name", ["hparams.yaml", "readme.txt", "events.out.tfevents.2.host"]
)
def test_a_folder_that_cannot_be_searched_does_not_break_the_mount(
    tmp_path: Path, name: str
) -> None:
    """Its files are not shown; the files of the run beside it are."""
    for run in ("good", "other"):
        (tmp_path / run).mkdir()
    _append(tmp_path / "good" / EVENTS, 0)
    _append(tmp_path / "other" / EVENTS, 0)
    notes = tmp_path / "other" / "notes"
    notes.mkdir()
    (notes / name).write_bytes(b"x")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    notes.chmod(0o644)
    try:
        listing = fs.readdir("/")
        good = fs.readdir("/good/scalars")
    finally:
        notes.chmod(0o755)

    assert {"good", "other"} <= set(listing)
    assert good[2:]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_a_projector_config_that_cannot_be_read_does_not_break_the_mount(
    tmp_path: Path,
) -> None:
    fs, projector = _fs_with_projector(tmp_path)
    projector.chmod(0)
    try:
        listing = fs.readdir("/")
    finally:
        projector.chmod(0o644)

    assert "scalars" in listing


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_a_sidecar_that_cannot_be_read_has_a_size_but_no_bytes(
    tmp_path: Path,
) -> None:
    """As on disk, ls -l shows the file and only a read of it fails."""
    fs, projector = _fs_with_projector(tmp_path)
    projector.write_text("a longer config")
    projector.chmod(0)
    try:
        size = fs.getattr("/projector/projector_config.pbtxt")["st_size"]
        with pytest.raises(OSError) as raised:
            fs.open("/projector/projector_config.pbtxt", os.O_RDONLY)
    finally:
        projector.chmod(0o644)

    assert size == len("a longer config")
    assert raised.value.errno == errno.EACCES
