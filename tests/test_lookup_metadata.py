"""Generated sizes and resolved names must follow the current publication."""

import json
import os
from errno import ENOENT
from pathlib import Path
from unittest.mock import call, patch

import pytest
from test_open_handles import LOSS, _append, _read_in_chunks
from test_render_cache import EVENTS, _count_renders

from tboardfs import TensorBoardFS


def _read_open(fs: TensorBoardFS, path: str) -> bytes:
    handle = fs.open(path, 0)
    try:
        return _read_in_chunks(fs, path, handle, 64)
    finally:
        fs.release(path, handle)


@pytest.mark.parametrize("open_first", [False, True], ids=["stat-first", "open-first"])
def test_generated_size_without_render_byte_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_first: bool
) -> None:
    event = tmp_path / EVENTS
    _append(event, 0)
    fs = TensorBoardFS(tmp_path, cache_size_mb=0, refresh_age_seconds=0)
    renders = _count_renders(monkeypatch)

    if open_first:
        before = _read_open(fs, LOSS)
        renders.clear()
        size = fs.getattr(LOSS)["st_size"]
        assert not renders
    else:
        size = fs.getattr(LOSS)["st_size"]
        renders.clear()
        assert fs.getattr(LOSS)["st_size"] == size
        assert not renders
        before = _read_open(fs, LOSS)
    assert size == len(before)
    assert [row["step"] for row in json.loads(before)] == [0]

    _append(event, 1)
    new_size = fs.getattr(LOSS)["st_size"]
    renders.clear()
    assert fs.getattr(LOSS)["st_size"] == new_size
    assert not renders
    after = _read_open(fs, LOSS)
    assert new_size == len(after) > size
    assert [row["step"] for row in json.loads(after)] == [0, 1]


def test_stat_after_append_scans_the_run_once(tmp_path: Path) -> None:
    event = tmp_path / EVENTS
    _append(event, 0)
    fs = TensorBoardFS(tmp_path, cache_size_mb=0, refresh_age_seconds=0)
    before = fs.getattr(LOSS)["st_size"]
    _append(event, 1)

    with patch("os.scandir", wraps=os.scandir) as listing:
        size = fs.getattr(LOSS)["st_size"]
    assert size == len(_read_open(fs, LOSS)) > before
    assert listing.call_args_list.count(call(tmp_path)) == 1


@pytest.mark.parametrize("operation", ["stat", "open"])
def test_name_map_leaf_disappears_on_direct_lookup(
    tmp_path: Path, operation: str
) -> None:
    mapped = tmp_path / ("%" * 100)
    mapped.mkdir()
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    path = "/.tboardfs-names.json"
    before = _read_open(fs, path)
    assert mapped.name in json.loads(before).values()
    assert fs.getattr(path)["st_size"] == len(before)

    mapped.rmdir()
    with pytest.raises(OSError) as raised:
        if operation == "stat":
            fs.getattr(path)
        else:
            _read_open(fs, path)
    assert raised.value.errno == ENOENT


def test_warmed_neighboring_leaves_follow_append_and_discovery(tmp_path: Path) -> None:
    event = tmp_path / EVENTS
    _append(event, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    neighbor = "/scalars/loss.tsv"
    alternate = "scalars//./loss.json"
    handle = fs.open(LOSS, 0)
    try:
        old_size = fs.getattr(LOSS)["st_size"]
        assert fs.getattr(LOSS)["st_size"] == old_size
        old_neighbor_size = fs.getattr(neighbor)["st_size"]
        assert fs.getattr(alternate)["st_size"] == old_size

        _append(event, 1)
        new_size = fs.getattr(alternate)["st_size"]
        assert fs.getattr(LOSS)["st_size"] == new_size > old_size
        assert fs.getattr(neighbor)["st_size"] > old_neighbor_size
        assert [row["step"] for row in json.loads(_read_open(fs, LOSS))] == [0, 1]
        pinned = _read_in_chunks(fs, LOSS, handle, 64)
        assert fs.getattr(LOSS, handle)["st_size"] == len(pinned) == old_size
        assert [row["step"] for row in json.loads(pinned)] == [0]

        child = tmp_path / "scalars"
        child.mkdir()
        _append(child / EVENTS, 9)
        child_rows = json.loads(_read_open(fs, "/scalars~2/scalars/loss.json"))
        assert [row["step"] for row in child_rows] == [9]

        for path in (
            "/scalars/absent.tsv",
            "scalars//./absent.json",
            "/scalars//loss.json/child",
        ):
            assert fs.getattr(alternate)["st_size"] == new_size
            with pytest.raises(FileNotFoundError) as raised:
                fs.getattr(path)
            assert (raised.value.errno, raised.value.filename) == (ENOENT, path)
    finally:
        fs.release(LOSS, handle)
