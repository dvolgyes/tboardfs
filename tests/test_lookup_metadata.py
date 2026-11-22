"""Generated sizes and resolved names must follow the current publication."""

import json
import os
from errno import ENOENT
from pathlib import Path
from unittest.mock import call, patch

import pytest
from test_incremental_parse import _frame
from test_open_handles import LOSS, _append, _read_in_chunks
from test_render_cache import EVENTS, _count_renders
from test_tf_producer_cases import CORPUS

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


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


def _image_frame(png: bytes, wall_time: float) -> bytes:
    event = compat_pb2.Event(step=1, wall_time=wall_time)
    event.summary.value.add(tag="img").image.encoded_image_string = png
    return _frame(event)


@pytest.mark.parametrize("age", [3600.0, float("inf")], ids=["live", "frozen"])
@pytest.mark.parametrize("blocked", ["mount", "storage"])
def test_warmed_binary_stat_rejects_retargeted_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age: float, blocked: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    storage = cache / "tboardfs" / "indexes-v1"
    storage.mkdir(parents=True, mode=0o700)
    event = source / EVENTS
    png = (CORPUS / "plugins/assets/sprite.png").read_bytes()
    frame = _image_frame(png, 1.0)
    event.write_bytes(frame)
    _append(source / "events.out.tfevents.sibling", 0)
    excluded = (mountpoint if blocked == "mount" else storage) / EVENTS
    excluded.write_bytes(frame)
    fs = TensorBoardFS(
        source, mountpoint=mountpoint, disk_index=True, refresh_age_seconds=age
    )
    path = "/images/img/000001.png"
    assert fs.getattr(path)["st_size"] == len(png)
    assert fs.getattr(path)["st_size"] == len(png)

    handle = fs.open(path, os.O_RDONLY)
    try:
        event.unlink()
        event.symlink_to(excluded)
        with pytest.raises(OSError) as raised:
            fs.getattr(path)
        assert raised.value.errno == ENOENT
        assert fs.getattr(path, handle)["st_size"] == len(png)
        assert fs.read(path, len(png), 0, handle) == png
    finally:
        fs.release(path, handle)


def test_repeated_binary_stat_after_sibling_and_payload_changes(tmp_path: Path) -> None:
    event = tmp_path / "events.out.tfevents.café"
    png = (CORPUS / "plugins/assets/sprite.png").read_bytes()
    event.write_bytes(_image_frame(png, 1.0))
    sibling = tmp_path / "events.out.tfevents.2"
    _append(sibling, 0)
    fs = TensorBoardFS(
        tmp_path, mountpoint=tmp_path / "mount", refresh_age_seconds=3600
    )
    path = "/images/img/000001.png"
    before = fs.getattr(path)
    assert before["st_size"] == len(png)

    _append(sibling, 1)
    assert fs.getattr(path) == before
    assert fs.getattr(path) == before
    assert [row["step"] for row in json.loads(_read_open(fs, LOSS))] == [0, 1]
    assert fs.getattr(path) == before

    payload = (CORPUS / "embedding-cases/00004/finder/sprite.png").read_bytes()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(_image_frame(payload, 2.0))
    replacement.replace(event)
    after = fs.getattr(path)
    assert after["st_size"] == len(payload)
    assert after["st_mtime"] == 2.0
    assert fs.getattr(path) == after
    assert _read_open(fs, path) == payload
