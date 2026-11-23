"""Reads through a real FUSE mount stay complete while event files grow."""

import ctypes
import errno
import io
import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import numpy as np
import pytest
from test_concurrency import _run_threads
from test_incremental_parse import _frame
from test_open_handles import _append

from tboardfs.proto import compat_pb2

FUSERMOUNT = shutil.which("fusermount3") or shutil.which("fusermount")
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or FUSERMOUNT is None
    or not os.access("/dev/fuse", os.R_OK | os.W_OK),
    reason="needs Linux FUSE and fusermount",
)
MOUNT_TIMEOUT_SECONDS = 20.0
_LINUX_DIRENT_HEADER = struct.Struct("=QqHB")
_DT_DIR = 4
_DT_REG = 8


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
            "--source-check-seconds",
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


def test_mount_root_exposes_guide(mounted: tuple[Path, Path]) -> None:
    _, run = mounted
    guide = run.parent / "llms.txt"
    assert guide.is_file()
    expected = (Path(__file__).parents[1] / "tboardfs" / "llms.txt").read_bytes()
    assert guide.read_bytes() == expected


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


def _getdents(directory: Path) -> list[tuple[str, int]]:
    """Read Linux d_type directly, without a subprocess or stat fallback."""
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "getdents64"):
        pytest.skip("libc does not expose getdents64")
    getdents = libc.getdents64
    getdents.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
    getdents.restype = ctypes.c_ssize_t
    buffer = ctypes.create_string_buffer(128)
    entries = []
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        while size := getdents(fd, buffer, len(buffer)):
            assert size > 0
            offset = 0
            while offset < size:
                _, _, length, kind = _LINUX_DIRENT_HEADER.unpack_from(buffer, offset)
                assert _LINUX_DIRENT_HEADER.size < length <= size - offset
                name = buffer[offset + _LINUX_DIRENT_HEADER.size : offset + length]
                entries.append((os.fsdecode(name.split(b"\0", 1)[0]), kind))
                offset += length
    finally:
        os.close(fd)
    return entries


def test_getdents_returns_types_across_pages_and_refreshes_on_reopen(
    mounted: tuple[Path, Path],
) -> None:
    """A metadata-free walk gets the complete, typed directory entries."""
    event, run = mounted
    assert ("scalars", _DT_DIR) in _getdents(run)
    assert _getdents(run / "scalars") == [
        (".", _DT_DIR),
        ("..", _DT_DIR),
        ("loss.json", _DT_REG),
        ("loss.npz", _DT_REG),
        ("loss.tsv", _DT_REG),
    ]
    raw = run / "raw" / event.name
    before = _getdents(raw)
    assert before[:2] == [(".", _DT_DIR), ("..", _DT_DIR)]
    assert len(before) == len(set(before)) == 402
    assert before[2:] == sorted(before[2:])
    assert all(kind == _DT_REG for _, kind in before[2:])
    offset = event.stat().st_size
    _append(event, 400)
    assert _getdents(raw) == [*before, (f"{offset:020d}.tfrecord", _DT_REG)]


def test_scandir_lists_names_in_lexical_order_after_append(
    mounted: tuple[Path, Path],
) -> None:
    """The real mount sorts final names without grouping files by type."""
    event, run = mounted
    (event.parent.parent / "a_run").mkdir()
    earlier_source = event.with_name("events.out.tfevents.earlier")
    image = (
        Path(__file__).parent / "data/producer-cases/plugins/assets/sprite.png"
    ).read_bytes()

    def append(step: int, tags: tuple[str, ...]) -> None:
        record = compat_pb2.Event(wall_time=1000.0 + step, step=step)
        for tag in tags:
            record.summary.value.add(tag=tag, simple_value=1.0)
        record.summary.value.add(tag="steps").image.encoded_image_string = image
        with event.open("ab") as handle:
            _ = handle.write(_frame(record))

    def listing(directory: Path) -> list[tuple[str, bool]]:
        with os.scandir(directory) as entries:
            return [(entry.name, entry.is_dir()) for entry in entries]

    _ = earlier_source.write_bytes(_frame(compat_pb2.Event(wall_time=1.0)))
    append(999999, ("z", "middle/child"))
    scalars = run / "scalars"
    images = run / "images" / "steps"
    assert listing(run.parent) == [
        ("a_run", True),
        ("llms.txt", False),
        ("run_a", True),
    ]
    assert listing(run / "raw") == [(earlier_source.name, True), (event.name, True)]
    before = [
        ("loss.json", False),
        ("loss.npz", False),
        ("loss.tsv", False),
        ("middle", True),
        ("z.json", False),
        ("z.npz", False),
        ("z.tsv", False),
    ]
    assert listing(scalars) == before
    assert listing(images) == [("999999.png", False)]

    append(1000000, ("a",))
    assert listing(scalars) == [
        ("a.json", False),
        ("a.npz", False),
        ("a.tsv", False),
        *before,
    ]
    assert listing(images) == [("1000000.png", False), ("999999.png", False)]
