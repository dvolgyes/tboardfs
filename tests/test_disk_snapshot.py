"""Persistent cache restores must stay bound to the hashed source snapshot."""

import os
from pathlib import Path

import pytest
from test_incremental_parse import _loss

from tboardfs.disk_index import DiskIndex
from tboardfs.event_source import EventSource
from tboardfs.indexer import _EventIndexer


def test_source_replaced_after_digest_is_not_a_hit(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents.a"
    path.write_bytes(_loss(2, 1.0))
    store = DiskIndex(tmp_path / "private", limit=8 << 20, mountpoint=None)
    old = _EventIndexer.parse_event_file(path)
    store.save(old, None)
    hit = store.candidates([path], 1)[path]
    replacement = tmp_path / "replacement"
    replacement.write_bytes(_loss(2, 2.0))
    assert replacement.stat().st_size == path.stat().st_size
    os.replace(replacement, path)
    assert store.load(hit, path, {}) is None
    store.save(old, None)
    current = store.candidates([path], 1)[path]
    assert store.load(current, path, {}) is None


def test_byte_identical_replacement_keeps_the_new_identity(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents.a"
    frames = _loss(3, 2.5)
    path.write_bytes(frames)
    store = DiskIndex(tmp_path / "private", limit=8 << 20, mountpoint=None)
    cache = _EventIndexer.parse_event_file(path)
    store.save(cache, None)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(frames)
    os.replace(replacement, path)
    candidate = store.candidates([path], 1)[path]
    restored = store.load(candidate, path, {})
    assert restored is not None
    assert (
        restored.progress.identity.inode
        == path.stat().st_ino
        != cache.progress.identity.inode
    )
    assert restored.scalars["loss"]["value"].tolist() == [2.5]


def test_candidate_hash_skips_changed_sizes_and_missing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DiskIndex(tmp_path / "private", limit=8 << 20, mountpoint=None)
    paths = [tmp_path / f"events.out.tfevents.{n}" for n in range(4)]
    for n, path in enumerate(paths):
        path.write_bytes(_loss(n, float(n)))
        store.save(_EventIndexer.parse_event_file(path), None)
    with paths[2].open("ab") as stream:
        stream.write(_loss(5, 5.0))
    paths[3].unlink()
    opened = []
    original = EventSource.open

    def record_open(path: Path, *, mountpoint: Path | None = None):
        opened.append(path)
        return original(path, mountpoint=mountpoint)

    monkeypatch.setattr(EventSource, "open", record_open)
    candidates = store.candidates(paths, 4)
    assert set(candidates) == set(paths[:2]) == set(opened)
    for n, path in enumerate(paths[:2]):
        restored = store.load(candidates[path], path, {})
        assert restored is not None
        assert restored.scalars["loss"]["value"].tolist() == [float(n)]


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file read permissions")
def test_cached_source_read_permission_is_required(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents.a"
    path.write_bytes(_loss(3, 1.0))
    store = DiskIndex(tmp_path / "private", limit=8 << 20, mountpoint=None)
    store.save(_EventIndexer.parse_event_file(path), None)
    hit = store.candidates([path], 1)[path]
    path.chmod(0)
    assert store.load(hit, path, {}) is None
    assert store.candidates([path], 1) == {}
