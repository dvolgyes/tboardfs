"""Binary handles read bounded ranges without losing their indexed version."""

import os
from concurrent.futures import ThreadPoolExecutor
from errno import EBADF, EIO, ENOENT
from pathlib import Path
from threading import Event
from typing import BinaryIO

import pytest
from test_incremental_parse import _frame

from scripts.scale_corpus import EVENT_NAME, SIBLING
from scripts.scale_corpus import _ScaleCorpus as ScaleCorpus
from tboardfs import TensorBoardFS, _directory
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import Record
from tboardfs.proto import compat_pb2
from tboardfs.verified_span import _VerifiedSpan

NAME = "/images/picture/000001.png"


def _image_log(
    directory: Path, *, large: bool = False, mountpoint: Path | None = None
) -> tuple[TensorBoardFS, Path, bytes]:
    blob = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * (32 if not large else 16384)
    event = compat_pb2.Event(step=1, wall_time=3.0)
    event.summary.value.add(tag="picture").image.encoded_image_string = blob
    path = directory / "events.out.tfevents.host"
    path.write_bytes(_frame(event))
    fs = TensorBoardFS(
        directory,
        refresh_age_seconds=float("inf"),
        cache_size_mb=0,
        mountpoint=mountpoint,
    )
    assert fs.readdir("/images/picture")[2:] == ["000001.png"]
    return fs, path, blob


def test_stat_open_and_repeated_late_reads_only_read_requested_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, _, blob = _image_log(tmp_path, large=True)
    pread = os.pread
    verify = IndexedFrame.verify
    reads: list[int] = []

    def count_range(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return pread(fd, size, offset)

    def count_frame(handle: BinaryIO, frame: FrameRef) -> Record:
        reads.append(frame.size)
        return verify(handle, frame)

    monkeypatch.setattr(os, "pread", count_range)
    monkeypatch.setattr(IndexedFrame, "verify", staticmethod(count_frame))
    assert fs.getattr(NAME)["st_size"] == len(blob)
    handle = fs.open(NAME, os.O_RDONLY)
    try:
        assert fs.getattr(NAME, handle)["st_size"] == len(blob)
        assert reads == []
        assert fs.read(NAME, 4096, len(blob) - 4096, handle) == blob[-4096:]
        assert fs.read(NAME, 4096, len(blob) - 10, handle) == blob[-10:]
        assert fs.read(NAME, 4096, len(blob), handle) == b""
        assert fs.read(NAME, 0, 10, handle) == b""
        assert fs.read(NAME, 2, 9) == blob[9:11]
        assert reads == [4104, 4104, 4096]
    finally:
        fs.release(NAME, handle)


def test_small_member_does_not_read_large_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / EVENT_NAME).write_bytes(
        ScaleCorpus.event_frame(0, large=ScaleCorpus.large_png())
    )
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=float("inf"), cache_size_mb=0)
    expected = ScaleCorpus.small_png(0xABCDEF)
    assert fs.getattr(SIBLING)["st_size"] == len(expected)

    def reject_full_read(handle: BinaryIO, frame: FrameRef) -> Record:
        pytest.fail(f"reading {frame.size} frame bytes for a {len(expected)}-byte PNG")

    monkeypatch.setattr(IndexedFrame, "verify", staticmethod(reject_full_read))
    handle = fs.open(SIBLING, os.O_RDONLY)
    try:
        assert fs.read(SIBLING, len(expected), 0, handle) == expected
    finally:
        fs.release(SIBLING, handle)


@pytest.mark.parametrize("change", ["append", "rename", "unlink"])
def test_handle_keeps_the_original_frame_after_source_changes(
    tmp_path: Path, change: str
) -> None:
    fs, path, blob = _image_log(tmp_path)
    handle = fs.open(NAME, os.O_RDONLY)
    try:
        if change == "append":
            with path.open("ab") as file:
                _ = file.write(b"partial appended record")
        elif change == "rename":
            _ = path.rename(path.with_suffix(".renamed"))
            _ = path.write_bytes(b"replacement")
        else:
            path.unlink()
        assert fs.getattr(NAME, handle)["st_size"] == len(blob)
        assert fs.read(NAME, 20, 8, handle) == blob[8:28]
        assert fs.read(NAME, len(blob), 28, handle) == blob[28:]
    finally:
        fs.release(NAME, handle)


def test_in_place_change_outside_requested_range_is_eio(tmp_path: Path) -> None:
    fs, path, blob = _image_log(tmp_path)
    handle = fs.open(NAME, os.O_RDONLY)
    try:
        before = path.stat()
        _ = path.write_bytes(path.read_bytes().replace(blob, b"X" + blob[1:]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(OSError) as raised:
            fs.read(NAME, 4, len(blob) - 4, handle)
        assert raised.value.errno == EIO
        with pytest.raises(OSError) as raised:
            fs.open(NAME, os.O_RDONLY)
        assert raised.value.errno == EIO
    finally:
        fs.release(NAME, handle)


def test_mutation_during_bounded_read_with_identical_stat_is_eio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, path, blob = _image_log(tmp_path)
    handle = fs.open(NAME, os.O_RDONLY)
    pread = os.pread
    fingerprint = FileIdentity.fingerprint(path.stat())
    monkeypatch.setattr(_VerifiedSpan, "_status", lambda self: fingerprint)

    def change_then_read(fd: int, size: int, offset: int) -> bytes:
        with path.open("r+b") as file:
            _ = file.seek(offset)
            _ = file.write(b"X")
        return pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", change_then_read)
    try:
        with pytest.raises(OSError) as raised:
            fs.read(NAME, 4, len(blob) - 4, handle)
        assert raised.value.errno == EIO
    finally:
        fs.release(NAME, handle)


def test_short_range_read_is_eio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, _, blob = _image_log(tmp_path)
    handle = fs.open(NAME, os.O_RDONLY)
    monkeypatch.setattr(os, "pread", lambda fd, size, offset: b"")
    try:
        with pytest.raises(OSError) as raised:
            fs.read(NAME, 4, len(blob) - 4, handle)
        assert raised.value.errno == EIO
    finally:
        fs.release(NAME, handle)


def test_release_cannot_close_a_descriptor_mid_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, _, blob = _image_log(tmp_path)
    handle = fs.open(NAME, os.O_RDONLY)
    entered, releasing = Event(), Event()
    descriptors: list[int] = []
    pread = os.pread
    close = _VerifiedSpan.close

    def read_during_release(fd: int, size: int, offset: int) -> bytes:
        descriptors.append(fd)
        entered.set()
        assert releasing.wait(timeout=5)
        return pread(fd, size, offset)

    def close_during_read(span: _VerifiedSpan) -> None:
        releasing.set()
        close(span)

    monkeypatch.setattr(os, "pread", read_during_release)
    monkeypatch.setattr(_VerifiedSpan, "close", close_during_read)
    with ThreadPoolExecutor(max_workers=2) as workers:
        read = workers.submit(fs.read, NAME, 4096, len(blob) - 4096, handle)
        assert entered.wait(timeout=5)
        release = workers.submit(fs.release, NAME, handle)
        assert read.result(timeout=5) == blob[-4096:]
        assert release.result(timeout=5) == 0
    with pytest.raises(OSError) as raised:
        _ = os.fstat(descriptors[0])
    assert raised.value.errno == EBADF


@pytest.mark.parametrize("ancestor", [False, True], ids=["file", "parent"])
def test_retargeted_source_to_mountpoint_cannot_open_a_span(
    tmp_path: Path, ancestor: bool
) -> None:
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    fs, path, _ = _image_log(source, mountpoint=mountpoint)
    assert fs.getattr(NAME)["st_size"] > 0
    link = source if ancestor else path
    target = link.rename(mountpoint / link.name)
    link.symlink_to(target, target_is_directory=ancestor)
    with pytest.raises(OSError) as raised:
        fs.open(NAME, os.O_RDONLY)
    assert raised.value.errno == ENOENT
    link.unlink()
    target.rename(link)
    assert fs.getattr(NAME)["st_size"] > 0


def test_guard_does_not_truncate_an_invalid_source_path(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    source = root / "source"
    source.mkdir()
    event = source / EVENT_NAME
    event.touch()
    fs = TensorBoardFS(source, mountpoint=root / "mount")

    assert fs._allowed(event)
    assert not fs._allowed(event.with_name(event.name + "\0suffix"))


@pytest.mark.parametrize("disable_native", [False, True], ids=["default", "fallback"])
def test_discovery_filters_fresh_links_to_protected_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disable_native: bool
) -> None:
    if disable_native:
        monkeypatch.setattr(_directory, "_OPEN_PATH", None)
    root = tmp_path.resolve()
    source = root / "source"
    source.mkdir()
    cache = root / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    mountpoint = root / "mount"
    storage = cache / "tboardfs" / "indexes-v1"
    for target in (root / "external", mountpoint, storage / "run"):
        target.mkdir(parents=True)
    _, event, blob = _image_log(root / "external")
    (mountpoint / event.name).write_bytes(event.read_bytes())
    (storage / "run" / event.name).write_bytes(event.read_bytes())

    fs = TensorBoardFS(source, mountpoint=mountpoint, disk_index=True)
    (source / "allowed").symlink_to(event.parent, target_is_directory=True)
    (source / "mount-link").symlink_to(mountpoint, target_is_directory=True)
    (source / "storage-link").symlink_to(storage / "run", target_is_directory=True)

    assert fs.readdir("/")[2:] == ["allowed"]
    assert fs.read("/allowed" + NAME, len(blob), 0) == blob


@pytest.mark.parametrize("depth", [0, 35, 105])
def test_guard_follows_earlier_link_before_denying_lexical_prefix(
    tmp_path: Path, depth: int
) -> None:
    parent = tmp_path / "parent"
    mountpoint = parent / "mount"
    mountpoint.mkdir(parents=True)
    alternate = tmp_path / "alternate"
    source = tmp_path / "source"
    source.mkdir()
    fs = TensorBoardFS(source, mountpoint=mountpoint)

    suffix = (*("directory" for _ in range(depth)), "event")
    candidate = mountpoint.joinpath(*suffix)
    redirected = (alternate / mountpoint.name).joinpath(*suffix)
    redirected.parent.mkdir(parents=True)
    redirected.touch()
    assert not fs._allowed(candidate)
    original = parent.rename(tmp_path / "original")
    parent.symlink_to(alternate.name, target_is_directory=True)
    assert fs._allowed(candidate)
    parent.unlink()
    original.rename(parent)
    assert not fs._allowed(candidate)


def test_guard_limits_symlink_resolution(tmp_path: Path) -> None:
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    fs = TensorBoardFS(source, mountpoint=mountpoint)
    target = source / "event"
    target.touch()
    for index in range(41):
        link = source / str(index)
        link.symlink_to(target.name)
        target = link

    assert fs._allowed(source / "39")
    assert not fs._allowed(source / "40")


@pytest.mark.parametrize("meta", [False, True], ids=["graph_def", "meta_graph_def"])
def test_stored_graph_stat_ranges_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meta: bool
) -> None:
    graph = compat_pb2.GraphDef()
    content = bytes(range(256)) * 8192
    graph.node.add(name="constant", op="Const").attr["value"].s = content
    if meta:
        blob = compat_pb2.MetaGraphDef(graph_def=graph).SerializeToString()
        event = compat_pb2.Event(step=1, meta_graph_def=blob)
        name = "/graphs/meta_graph.pb"
    else:
        blob = graph.SerializeToString()
        event = compat_pb2.Event(step=1, graph_def=blob)
        name = "/graphs/graph.pb"
    sibling = compat_pb2.GraphDef()
    sibling.node.add(name="sibling", op="NoOp")
    sibling_event = (
        compat_pb2.Event(step=2, graph_def=sibling.SerializeToString())
        if meta
        else compat_pb2.Event(
            step=2,
            meta_graph_def=compat_pb2.MetaGraphDef(
                graph_def=sibling
            ).SerializeToString(),
        )
    )
    path = tmp_path / "events.out.tfevents.host"
    original = _frame(sibling_event) + _frame(event)
    _ = path.write_bytes(original)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=float("inf"), cache_size_mb=0)
    assert fs.readdir("/graphs")[2:] == ["graph.pb", "meta_graph.pb"]
    pread = os.pread
    verify = IndexedFrame.verify
    reads: list[int] = []
    verified: list[int] = []

    def count_range(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return pread(fd, size, offset)

    def count_frame(handle: BinaryIO, frame: FrameRef) -> Record:
        verified.append(frame.offset)
        return verify(handle, frame)

    monkeypatch.setattr(os, "pread", count_range)
    monkeypatch.setattr(IndexedFrame, "verify", staticmethod(count_frame))
    assert fs.getattr(name)["st_size"] == len(blob)
    handle = fs.open(name, os.O_RDONLY)
    try:
        assert reads == verified == []
        offset = len(blob) - 4169
        assert fs.read(name, 4096, offset, handle) == blob[offset : offset + 4096]
        assert len(reads) == 1
        assert 4096 <= reads[0] <= 8192
        assert verified == []

        before = path.stat()
        start = original.index(content)
        _ = path.write_bytes(original[:start] + b"X" + original[start + 1 :])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(OSError) as raised:
            fs.read(name, 4096, offset, handle)
        assert raised.value.errno == EIO
        assert verified == [len(_frame(sibling_event))]
    finally:
        fs.release(name, handle)


def test_repeated_opens_reuse_full_frame_revalidation_after_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, path, blob = _image_log(tmp_path, large=True)
    pread = os.pread
    verify = IndexedFrame.verify
    reads: list[int] = []
    verified: list[int] = []

    def count_range(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return pread(fd, size, offset)

    def count_frame(handle: BinaryIO, frame: FrameRef) -> Record:
        verified.append(frame.size)
        return verify(handle, frame)

    monkeypatch.setattr(os, "pread", count_range)
    monkeypatch.setattr(IndexedFrame, "verify", staticmethod(count_frame))
    with path.open("ab") as file:
        _ = file.write(_frame(compat_pb2.Event(step=2)))
    for _ in range(2):
        handle = fs.open(NAME, os.O_RDONLY)
        try:
            assert fs.read(NAME, 4096, len(blob) - 4096, handle) == blob[-4096:]
        finally:
            fs.release(NAME, handle)
    assert len(verified) == 1
    assert reads == [4104, 4104]

    before = path.stat()
    with path.open("r+b") as file:
        _ = file.seek(13)
        _ = file.write(b"\xff")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(OSError) as raised:
        fs.open(NAME, os.O_RDONLY)
    assert raised.value.errno == EIO
    assert len(verified) == 2
