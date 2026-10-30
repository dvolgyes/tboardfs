"""Sidecar files that appear, change or go away show at the next refresh."""

import os
from pathlib import Path

from tboardfs import TensorBoardFS

from test_open_handles import _append

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
