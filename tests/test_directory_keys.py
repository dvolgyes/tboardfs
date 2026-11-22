"""Directory listings follow published trees without repeating warm work."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from test_open_handles import _append

from tboardfs import TensorBoardFS, paths
from tboardfs.file_times import FileTimes

EVENTS = "events.out.tfevents.host"


def test_warm_listing_reuses_sorted_keys_across_unrelated_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        _append(tmp_path / run / EVENTS, 0)
    fs = TensorBoardFS(
        tmp_path,
        refresh_age_seconds=0,
        source_check_seconds=0,
        scalar_formats=("json", "tsv"),
    )
    fs.virtual_tree()
    sorts = []

    def counting_sorted(values: Iterable[str], **kwargs: Any) -> list[str]:
        if isinstance(values, Mapping) and "loss.json" in values:
            sorts.append(1)
        return sorted(values, **kwargs)

    monkeypatch.setattr(paths, "sorted", counting_sorted, raising=False)
    expected = [".", "..", "loss.json", "loss.tsv"]
    listing = fs.readdir("/b/scalars")
    assert listing == expected
    listing.clear()
    assert fs.readdir("/b/scalars") == expected

    _append(tmp_path / "a" / EVENTS, 1)
    fs.readdir("/a")
    assert fs.readdir("/b/scalars") == expected
    assert len(sorts) == 1


def test_warm_listing_follows_discovery_and_collision_renames(tmp_path: Path) -> None:
    (tmp_path / "scalars").mkdir()
    _append(tmp_path / "scalars" / EVENTS, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.readdir("/") == [".", "..", "scalars"]
    assert "images" in fs.readdir("/scalars")

    _append(tmp_path / EVENTS, 1)
    names = fs.readdir("/")
    assert names[2:] == sorted(names[2:])
    assert names.index("scalars") + 1 == names.index("scalars~2")
    assert "loss.json" in fs.readdir("/scalars")
    assert "images" in fs.readdir("/scalars~2")

    (tmp_path / "scalars" / EVENTS).unlink()
    (tmp_path / "scalars").rmdir()
    (tmp_path / "extra").mkdir()
    names = fs.readdir("/")
    assert "extra" in names and "scalars~2" not in names
    assert names[2:] == sorted(names[2:])


def test_stamp_reused_subtree_does_not_visit_children() -> None:
    empty: dict[str, Any] = {"type": "dir", "children": {}}
    tree: dict[str, Any] = {
        "type": "dir",
        "children": {"empty": empty, "file": {"type": "scalar", "mtime": 3}},
    }
    FileTimes.stamp_dirs(tree)
    assert empty["mtime"] == 3

    class StampedChildren(dict[str, Any]):
        def values(self) -> Any:
            pytest.fail("a stamped subtree was revisited")

    tree["children"] = StampedChildren(tree["children"])
    sibling: dict[str, Any] = {"type": "dir", "children": {}}
    combined: dict[str, Any] = {
        "type": "dir",
        "children": {
            "shared": tree,
            "empty": sibling,
            "new": {"type": "scalar", "mtime": 9},
        },
    }
    FileTimes.stamp_dirs(combined)
    assert combined["mtime"] == sibling["mtime"] == 9
    assert tree["mtime"] == empty["mtime"] == 3
