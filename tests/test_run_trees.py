"""Each run keeps its own tree version, so one growing run rebuilds alone."""

from pathlib import Path
from typing import Any

import pytest

from tboardfs import TensorBoardFS
from tboardfs.model import RunCache
import tboardfs.filesystem

from test_open_handles import _append

EVENTS = "events.out.tfevents.host"


def _listing(fs: TensorBoardFS, path: str = "/") -> dict[str, list[str]]:
    """Return every directory listing below ``path``."""
    entries = fs.readdir(path)[2:]
    listing = {path: entries}
    for name in entries:
        child = f"{path.rstrip('/')}/{name}"
        if fs.getattr(child)["st_nlink"] == 2:
            listing.update(_listing(fs, child))
    return listing


def _two_runs(tmp_path: Path) -> list[Path]:
    paths = []
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
        paths.append(tmp_path / name / EVENTS)
        _append(paths[-1], 0)
    return paths


def test_growing_run_rebuilds_only_its_own_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New records in one run do not rebuild the trees of other runs."""
    run_a, _ = _two_runs(tmp_path)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/")
    original = tboardfs.filesystem._build_run_virtual_tree
    built: list[tuple[str, ...]] = []

    def counting_build(run: RunCache, *args: Any, **kwargs: Any) -> dict[str, Any]:
        built.append(run.run_parts)
        return original(run, *args, **kwargs)

    monkeypatch.setattr(tboardfs.filesystem, "_build_run_virtual_tree", counting_build)
    _append(run_a, 1)

    fs.readdir("/run_b/scalars")
    fs.readdir("/run_a/scalars")

    assert built == [("run_a",)]


def test_run_versions_are_unique_and_change_with_the_run(tmp_path: Path) -> None:
    """Only a changed run gets a new version; a re-created run gets an unused one."""
    run_a, run_b = _two_runs(tmp_path)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.readdir("/")
    before = {parts: run.version for parts, run in fs._state.runs.items()}
    seen = set(before.values())

    _append(run_a, 1)
    fs.readdir("/")
    after = {parts: run.version for parts, run in fs._state.runs.items()}
    run_b.unlink()
    fs.readdir("/")
    _append(run_b, 0)
    fs.readdir("/")
    recreated = fs._state.runs[("run_b",)].version

    assert len(seen) == 2
    assert after[("run_a",)] not in seen
    assert after[("run_b",)] == before[("run_b",)]
    assert recreated not in seen | set(after.values())


def test_removed_nested_run_leaves_no_entries_in_its_parent_run(
    tmp_path: Path,
) -> None:
    """A run inside a tab-named directory of another run can go away cleanly."""
    (tmp_path / "a" / "scalars").mkdir(parents=True)
    _append(tmp_path / "a" / EVENTS, 0)
    nested = tmp_path / "a" / "scalars" / EVENTS
    _append(nested, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert "images" in fs.readdir("/a/scalars~2")

    nested.unlink()

    assert _listing(fs) == _listing(TensorBoardFS(tmp_path))
