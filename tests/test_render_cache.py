"""Rendered file bytes are cached by run data version within a byte budget."""

import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_concurrency import _run_threads
from test_open_handles import _append

import tboardfs.filesystem
from tboardfs import TensorBoardFS
from tboardfs.render_cache import ENTRY_OVERHEAD_BYTES, RenderCache

EVENTS = "events.out.tfevents.host"


def _entry(size: int) -> bytes:
    """Return bytes that use ``size`` bytes of the cache budget."""
    return b"x" * (size - ENTRY_OVERHEAD_BYTES)


def _count_renders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the node type of each render of a virtual file."""
    original = tboardfs.filesystem.render_node
    renders: list[str] = []

    def counting(node: dict[str, Any]) -> bytes:
        renders.append(node["type"])
        return original(node)

    monkeypatch.setattr(tboardfs.filesystem, "render_node", counting)
    return renders


def test_cache_returns_what_was_put() -> None:
    """A put entry is returned for its key; another key misses."""
    cache = RenderCache(10_000)
    scope = cache.scope(1)
    cache.put((scope, ("a",)), b"data")

    assert cache.get((scope, ("a",))) == b"data"
    assert cache.get((scope, ("b",))) is None


def test_overflow_drops_the_oldest_until_half_the_limit() -> None:
    """Over the limit, the least recently used entries go until half is left."""
    cache = RenderCache(1000)
    scope = cache.scope(1)
    for name in "abcd":
        cache.put((scope, (name,)), _entry(250))
    assert cache.get((scope, ("a",))) is not None

    cache.put((scope, ("e",)), _entry(250))

    kept = [name for name in "abcde" if cache.get((scope, (name,))) is not None]
    assert kept == ["a", "e"]
    assert cache.size_bytes == 500
    cache.expire(1)
    assert cache.size_bytes == 0


def test_entry_larger_than_half_the_limit_is_not_kept() -> None:
    """An entry that its own overflow would drop is not stored."""
    cache = RenderCache(1000)
    scope = cache.scope(1)
    cache.put((scope, ("small",)), _entry(200))

    cache.put((scope, ("large",)), _entry(501))

    assert cache.get((scope, ("large",))) is None
    assert cache.get((scope, ("small",))) is not None


def test_zero_limit_keeps_nothing() -> None:
    """A zero budget turns the cache off."""
    cache = RenderCache(0)
    scope = cache.scope(1)
    cache.put((scope, ("a",)), b"")

    assert cache.get((scope, ("a",))) is None


def test_negative_limit_is_refused() -> None:
    """A negative budget is a configuration error."""
    with pytest.raises(ValueError, match="cache size"):
        RenderCache(-1)


def test_concurrent_use_keeps_the_size_within_the_limit() -> None:
    """Threads that put and get at once leave a consistent byte count."""
    cache = RenderCache(10_000)
    scope = cache.scope(1)
    rng_seeds = range(8)

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(2000):
            key = (scope, (str(rng.integers(50)),))
            if cache.get(key) is None:
                cache.put(key, _entry(int(rng.integers(300, 3000))))

    errors = _run_threads([lambda seed=seed: worker(seed) for seed in rng_seeds])

    assert errors == []
    assert cache.size_bytes <= 10_000
    assert cache.size_bytes == sum(
        len(data) + ENTRY_OVERHEAD_BYTES for data in cache._entries.values()
    )


def test_filesystem_renders_an_unchanged_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stat and reads of an unchanged file render it one time."""
    _append(tmp_path / EVENTS, 0, 1)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    renders = _count_renders(monkeypatch)

    size = fs.getattr("/scalars/loss.json")["st_size"]
    first = fs.read("/scalars/loss.json", size, 0)
    second = fs.read("/scalars/loss.json", size, 0)

    assert renders == ["scalar"]
    assert first == second


def test_filesystem_renders_again_when_the_run_grows(tmp_path: Path) -> None:
    """New records make the cached bytes of their run stale."""
    path = tmp_path / EVENTS
    _append(path, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    before = fs.read("/scalars/loss.json", 1 << 20, 0)

    _append(path, 1)

    assert fs.read("/scalars/loss.json", 1 << 20, 0) == TensorBoardFS(tmp_path).read(
        "/scalars/loss.json", 1 << 20, 0
    )
    assert fs.read("/scalars/loss.json", 1 << 20, 0) != before


def test_filesystem_keeps_renders_of_other_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A growing run does not drop the cached bytes of another run."""
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
        _append(tmp_path / name / EVENTS, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.read("/run_b/scalars/loss.json", 1 << 20, 0)
    renders = _count_renders(monkeypatch)

    _append(tmp_path / "run_a" / EVENTS, 1)
    fs.read("/run_a/scalars/loss.json", 1 << 20, 0)
    fs.read("/run_b/scalars/loss.json", 1 << 20, 0)

    assert renders == ["scalar"]


def test_removed_run_releases_cache_but_keeps_bytes_of_open_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
        _append(tmp_path / name / EVENTS, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    path_a = "/run_a/scalars/loss.json"
    path_b = "/run_b/scalars/loss.json"
    other = fs.read(path_b, 1 << 20, 0)
    retained_size = fs._renders.size_bytes
    handle = fs.open(path_a, 0)
    pinned = fs.read(path_a, 1 << 20, 0, handle)
    assert fs._renders.size_bytes > retained_size

    (tmp_path / "run_a" / EVENTS).unlink()
    assert "scalars" not in fs.readdir("/run_a")
    assert fs._renders.size_bytes == retained_size
    assert ("run_a",) not in fs._state.run_trees
    assert fs.read(path_a, 1 << 20, 0, handle) == pinned
    fs.release(path_a, handle)
    renders = _count_renders(monkeypatch)
    assert fs.read(path_b, 1 << 20, 0) == other
    assert not renders


def test_shrinking_event_file_releases_old_render_bytes(tmp_path: Path) -> None:
    event = tmp_path / EVENTS
    _append(event, 0)
    size = event.stat().st_size
    _append(event, 1)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    path = "/scalars/loss.json"
    before = fs.read(path, 1 << 20, 0)
    assert fs._renders.size_bytes > 0

    with event.open("r+b") as stream:
        stream.truncate(size)
    fs.readdir("/scalars")
    assert fs._renders.size_bytes == 0
    assert fs.read(path, 1 << 20, 0) != before


def test_tree_only_change_evicts_the_name_map_without_other_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _append(run / EVENTS, 0)
    mapped = tmp_path / ("%" * 100)
    mapped.mkdir()
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    path = "/run/scalars/loss.json"
    other = fs.read(path, 1 << 20, 0)
    retained_size = fs._renders.size_bytes
    assert mapped.name.encode() in fs.read("/.tboardfs-names.json", 1 << 20, 0)
    assert fs._renders.size_bytes > retained_size

    mapped.rmdir()
    assert ".tboardfs-names.json" not in fs.readdir("/")
    assert fs._renders.size_bytes == retained_size
    renders = _count_renders(monkeypatch)
    assert fs.read(path, 1 << 20, 0) == other
    assert not renders


def test_render_finishing_after_run_change_does_not_repopulate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = tmp_path / EVENTS
    _append(event, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    path = "/scalars/loss.json"
    original = tboardfs.filesystem.render_node
    rendering = threading.Event()
    resume = threading.Event()

    def delayed(node: dict[str, Any]) -> bytes:
        data = original(node)
        rendering.set()
        assert resume.wait(timeout=5)
        return data

    monkeypatch.setattr(tboardfs.filesystem, "render_node", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(fs.read, path, 1 << 20, 0)
        try:
            assert rendering.wait(timeout=5)
            _append(event, 1)
            fs.readdir("/scalars")
        finally:
            resume.set()
        old_bytes = pending.result(timeout=5)

    assert fs._renders.size_bytes == 0
    monkeypatch.setattr(tboardfs.filesystem, "render_node", original)
    assert fs.read(path, 1 << 20, 0) != old_bytes


def test_filesystem_reads_sidecars_from_disk_each_time(tmp_path: Path) -> None:
    """Sidecar files are not cached, so a changed sidecar shows at once."""
    _append(tmp_path / EVENTS, 0)
    projector = tmp_path / "projector_config.pbtxt"
    projector.write_text("first")
    fs = TensorBoardFS(tmp_path)
    assert fs.read("/projector/projector_config.pbtxt", 100, 0) == b"first"

    projector.write_text("second")

    assert fs.read("/projector/projector_config.pbtxt", 100, 0) == b"second"


def test_filesystem_with_zero_cache_renders_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cache_size_mb=0`` turns the render cache off."""
    _append(tmp_path / EVENTS, 0)
    fs = TensorBoardFS(tmp_path, cache_size_mb=0)
    renders = _count_renders(monkeypatch)

    fs.read("/scalars/loss.json", 1 << 20, 0)
    fs.read("/scalars/loss.json", 1 << 20, 0)

    assert renders == ["scalar", "scalar"]


def test_concurrent_readers_see_whole_versions_while_runs_grow(
    tmp_path: Path,
) -> None:
    """Cached and fresh renders never mix versions under concurrent reads."""
    path = tmp_path / EVENTS
    _append(path, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    stop = threading.Event()

    def writer() -> None:
        for step in range(1, 150):
            _append(path, step)
        stop.set()

    def reader() -> None:
        last = 0
        while not stop.is_set():
            data = fs.read("/scalars/loss.npz", 1 << 20, 0)
            steps = np.load(BytesIO(data))["step"]
            assert steps.tolist() == list(range(len(steps)))
            assert len(steps) >= last
            last = len(steps)

    assert _run_threads([writer, *[reader] * 6]) == []
