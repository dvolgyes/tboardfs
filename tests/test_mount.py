"""Reads through a real FUSE mount stay complete while event files grow."""

from collections.abc import Iterator
import errno
import io
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import TextIO

import numpy as np
import pytest

from test_concurrency import _run_threads
from test_open_handles import _append

FUSERMOUNT = shutil.which("fusermount3") or shutil.which("fusermount")
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or FUSERMOUNT is None
    or not os.access("/dev/fuse", os.R_OK | os.W_OK),
    reason="needs Linux FUSE and fusermount",
)
MOUNT_TIMEOUT_SECONDS = 20.0


@pytest.fixture
def mounted(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Mount a source with one growing run; yield (event file, run dir)."""
    source = tmp_path / "src"
    mountpoint = tmp_path / "mnt"
    (source / "run_a").mkdir(parents=True)
    mountpoint.mkdir()
    event = source / "run_a" / "events.out.tfevents.host"
    _append(event, *range(400))
    log = tmp_path / "mount.log"
    # A file, not a pipe: an unread pipe would block the logging mount.
    with log.open("w") as log_handle:
        process = _start_mount(source, mountpoint, log_handle)
    run = mountpoint / "run_a"
    deadline = time.monotonic() + MOUNT_TIMEOUT_SECONDS
    while not (run / "scalars").exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            process.wait()
            pytest.fail(f"mount did not start: {log.read_text()}")
        time.sleep(0.05)
    try:
        yield event, run
    finally:
        unmount = [str(FUSERMOUNT), "-u", str(mountpoint)]
        if subprocess.run(unmount, check=False).returncode != 0:
            # Failed tests can keep files open through their tracebacks.
            subprocess.run([*unmount, "-z"], check=False)
        try:
            process.wait(timeout=MOUNT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _start_mount(
    source: Path, mountpoint: Path, log_handle: TextIO
) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from tboardfs import main; main()",
            str(source),
            str(mountpoint),
            "--foreground",
            "--refresh-age-seconds",
            "0",
        ],
        stderr=log_handle,
    )


def _steps(run: Path, kind: str) -> list[int]:
    if kind == "json":
        return [
            row["step"] for row in json.loads((run / "scalars/loss.json").read_bytes())
        ]
    npz = run / "scalars/loss.npz"
    # np.load on a path seeks to the end, which uses the size the kernel
    # keeps per inode; another handle of a newer version can change it.
    source = io.BytesIO(npz.read_bytes()) if kind == "npz_bytes" else npz
    with np.load(source) as archive:
        return [int(step) for step in archive["step"]]


def test_reads_right_after_a_change_are_complete(
    mounted: tuple[Path, Path],
) -> None:
    """The kernel must not cut reads to a size it cached before the change."""
    event, run = mounted
    for step in range(400, 410):
        (run / "scalars/loss.json").stat()
        (run / "scalars/loss.npz").stat()
        _append(event, step)

        assert _steps(run, "json") == list(range(step + 1))
        assert _steps(run, "npz") == list(range(step + 1))
        with (run / "scalars/loss.json").open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            handle.seek(0)
            assert end == len(handle.read())


def test_a_private_memory_map_gives_the_bytes_of_a_read(
    mounted: tuple[Path, Path],
) -> None:
    """As the README states: direct I/O allows private maps, not shared maps."""
    _, run = mounted
    path = run / "scalars/loss.npz"
    expected = path.read_bytes()

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
            assert mapped[:] == expected
        with pytest.raises(OSError) as raised:
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    assert raised.value.errno == errno.ENODEV


def test_concurrent_readers_see_whole_versions_while_a_run_grows(
    mounted: tuple[Path, Path],
) -> None:
    """Sequential readers of one file never get bytes of two versions."""
    event, run = mounted
    stop = threading.Event()

    def writer() -> None:
        for step in range(400, 900):
            _append(event, step)
            time.sleep(0.002)
        stop.set()

    def reader(kind: str) -> None:
        while not stop.is_set():
            steps = _steps(run, kind)
            assert steps == list(range(len(steps)))
            assert len(steps) >= 400

    workers = [writer] + [
        lambda kind=kind: reader(kind) for kind in ["json", "npz_bytes"] * 4
    ]
    errors = _run_threads(workers)

    assert errors == []
    assert _steps(run, "json") == list(range(900))
