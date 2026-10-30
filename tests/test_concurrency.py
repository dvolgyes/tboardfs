"""Concurrent FUSE callbacks must not corrupt state or deadlock."""

from collections.abc import Callable
import faulthandler
import json
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Any

import pytest

from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2
import tboardfs.filesystem

DEADLOCK_TIMEOUT_SECONDS = 30.0


def _loss_record(step: int) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag="loss", simple_value=1.0 / (step + 1))
    payload = event.SerializeToString()
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def _append(path: Path, step: int) -> None:
    with path.open("ab") as handle:
        handle.write(_loss_record(step))


def _run_threads(workers: list[Callable[[], None]]) -> list[BaseException]:
    """Run workers to completion; fail with all thread stacks on a deadlock."""
    errors: list[BaseException] = []

    def guarded(worker: Callable[[], None]) -> None:
        try:
            worker()
        except BaseException as error:  # noqa: BLE001 - reported by the test
            errors.append(error)

    threads = [
        threading.Thread(target=guarded, args=(worker,), daemon=True)
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + DEADLOCK_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        pytest.fail(f"threads still running after {DEADLOCK_TIMEOUT_SECONDS}s")
    return errors


def test_state_updates_never_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one callback at a time refreshes the shared run state."""
    _append(tmp_path / "events.out.tfevents", 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    original = tboardfs.filesystem._RunData.refresh_event_files
    active = [0]
    peak = [0]
    counter_lock = threading.Lock()

    def slow_refresh(*args: Any, **kwargs: Any) -> bool:
        with counter_lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with counter_lock:
            active[0] -= 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        tboardfs.filesystem._RunData, "refresh_event_files", staticmethod(slow_refresh)
    )
    start = threading.Barrier(8)

    def reader() -> None:
        start.wait()
        for _ in range(5):
            fs.getattr("/scalars/loss.json")

    errors = _run_threads([reader] * 8)

    assert errors == []
    assert peak[0] == 1


def test_tree_built_during_a_change_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change that lands while the tree is built shows on the next lookup."""
    path = tmp_path / "events.out.tfevents"
    _append(path, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    original = TensorBoardFS._build_tree
    changed = []

    def build_while_training_writes(self: TensorBoardFS) -> dict[str, Any]:
        tree = original(self)
        if not changed:
            changed.append(True)
            _append(path, 1)
            self._refresh_runs({})
        return tree

    monkeypatch.setattr(TensorBoardFS, "_build_tree", build_while_training_writes)
    fs.readdir("/")

    rows = json.loads(fs.read("/scalars/loss.json", 1 << 20, 0))

    assert [row["step"] for row in rows] == [0, 1]


def test_readers_survive_concurrent_training_writes(tmp_path: Path) -> None:
    """Readers see valid, growing data while runs grow and new runs appear."""
    runs = [tmp_path / "run_a", tmp_path / "run_b"]
    for run in runs:
        run.mkdir()
        _append(run / "events.out.tfevents.host", 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    stop = threading.Event()

    def writer() -> None:
        for step in range(1, 200):
            for run in runs:
                _append(run / "events.out.tfevents.host", step)
            if step % 20 == 0:
                new_run = tmp_path / f"run_new_{step}"
                new_run.mkdir()
                _append(new_run / "events.out.tfevents.host", 0)
            time.sleep(0.002)
        stop.set()

    def reader() -> None:
        last_seen = {run.name: 0 for run in runs}
        while not stop.is_set():
            listing = fs.readdir("/")
            assert {"run_a", "run_b"} <= set(listing)
            for name in last_seen:
                path = f"/{name}/scalars/loss.json"
                size = fs.getattr(path)["st_size"]
                assert size > 0
                steps = [row["step"] for row in json.loads(fs.read(path, 1 << 24, 0))]
                assert steps == list(range(len(steps)))
                assert len(steps) >= last_seen[name]
                last_seen[name] = len(steps)
            for name in listing:
                if name.startswith("run_new_"):
                    assert "loss.json" in fs.readdir(f"/{name}/scalars")

    errors = _run_threads([writer] + [reader] * 8)

    assert errors == []
    final = json.loads(fs.read("/run_a/scalars/loss.json", 1 << 24, 0))
    assert [row["step"] for row in final] == list(range(200))
