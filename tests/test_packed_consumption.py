"""Exercise mounted media from worker ranges and restored disk indexes."""

from errno import ENOENT
from pathlib import Path
from typing import Any, Never

import pytest
from test_incremental_parse import _frame, _loss
from test_payload_spans import _PNG as PNG
from test_restart_purge import _start, _version
from test_tf2_summaries import _images

from tboardfs import TensorBoardFS, filesystem, parallel_index
from tboardfs.indexer import _EventIndexer
from tboardfs.model import BinaryEntry

_IMAGES = "/images/img"
_LAST_STEP = 257


def _image_record(
    step: int, payload: bytes, mtime: float, *, tag: str = "img"
) -> bytes:
    event = _images(step, payload)
    event.wall_time = mtime
    event.summary.value[0].tag = tag
    return _frame(event)


def _unexpected_parse(*_args: object, **_kwargs: object) -> Never:
    pytest.fail("expected packed input, not a serial parse")


@pytest.fixture(params=["worker", "disk"])
def packed_mount(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, bool]:
    source = tmp_path / "runs"
    source.mkdir()
    disk_index = request.param == "disk"
    if disk_index:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    else:
        monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 1)
        monkeypatch.setattr(parallel_index, "_available_cpus", lambda: 2)
        monkeypatch.setattr(parallel_index, "_PREFIX_RECORDS", 2)
    return source, disk_index


def test_listing_does_not_materialize_binary_entries(
    packed_mount: tuple[Path, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, disk_index = packed_mount
    event = source / "events.out.tfevents.1"
    tail = PNG + b"last row"
    event.write_bytes(
        _version()
        + _loss(0, 1.0)
        + b"".join(_image_record(step, PNG, step) for step in range(1, _LAST_STEP))
        + _image_record(_LAST_STEP, tail, 900.0)
    )
    names = [".", "..", *(f"{step:06d}.png" for step in range(1, _LAST_STEP + 1))]
    if disk_index:
        assert TensorBoardFS(source, disk_index=True).readdir(_IMAGES) == names

    entry_init = BinaryEntry.__init__
    constructions = 0
    construction_limit = 0

    def guard_construction(entry: BinaryEntry, *args: Any, **kwargs: Any) -> None:
        nonlocal constructions
        constructions += 1
        assert constructions <= construction_limit, (
            "constructed full media rows in bulk"
        )
        entry_init(entry, *args, **kwargs)

    monkeypatch.setattr(BinaryEntry, "__init__", guard_construction)
    monkeypatch.setattr(_EventIndexer, "parse_event_file", _unexpected_parse)
    fs = TensorBoardFS(
        source, disk_index=disk_index, index_workers=1 if disk_index else 2
    )
    assert fs.readdir(_IMAGES) == names
    assert fs.getattr(_IMAGES)["st_mtime"] == 900.0

    construction_limit = 2
    path = f"{_IMAGES}/{_LAST_STEP:06d}.png"
    info = fs.getattr(path)
    assert info["st_size"] == len(tail)
    assert info["st_mtime"] == 900.0
    assert fs.read(path, len(tail), 7) == tail[7:]


def _assert_media(
    fs: TensorBoardFS,
    path: str,
    payload: bytes,
    mtime: float,
    *handles: int,
    offset: int = 0,
) -> None:
    info = fs.getattr(path, *handles)
    assert info["st_size"] == len(payload)
    assert info["st_mtime"] == mtime
    assert fs.read(path, len(payload), offset, *handles) == payload[offset:]


def test_late_duplicate_and_renamed_media_survive_refresh_on_open_handles(
    packed_mount: tuple[Path, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, disk_index = packed_mount
    event = source / "events.out.tfevents.1"
    winning = PNG + b"late duplicate"
    nested = PNG + b"renamed tag"
    event.write_bytes(
        _version()
        + _loss(0, 1.0)
        + _image_record(1, PNG, 10.0)
        + _image_record(2, PNG + b"early duplicate", 20.0)
        + _image_record(3, PNG + bytes(1024), 30.0)
        + _image_record(2, winning, 40.0)
        + _image_record(4, nested, 50.0, tag="img/000002.png")
    )
    primary = f"{_IMAGES}/000002.png"
    renamed = f"{_IMAGES}/000002.png~2/000004.png"
    directories = (_IMAGES, f"{_IMAGES}/000002.png~2")
    serial = TensorBoardFS(source)
    listings = {path: serial.readdir(path) for path in directories}
    if disk_index:
        assert (
            TensorBoardFS(source, disk_index=True).readdir(_IMAGES) == listings[_IMAGES]
        )

    with monkeypatch.context() as packed_only:
        packed_only.setattr(_EventIndexer, "parse_event_file", _unexpected_parse)
        fs = TensorBoardFS(
            source,
            disk_index=disk_index,
            index_workers=1 if disk_index else 2,
            refresh_age_seconds=0,
            source_check_seconds=0,
        )
        assert {path: fs.readdir(path) for path in directories} == listings
        _assert_media(fs, primary, winning, 40.0)
        _assert_media(fs, renamed, nested, 50.0)
        primary_handle = fs.open(primary, 0)
        renamed_handle = fs.open(renamed, 0)

    try:
        appended = PNG + b"ordinary append"
        with event.open("ab") as handle:
            handle.write(_image_record(6, appended, 55.0))
        assert fs.read(f"{_IMAGES}/000006.png", len(appended), 0) == appended
        assert fs.read(primary, len(winning), 0) == winning
        assert fs.read(renamed, len(nested), 0) == nested

        replacement = PNG + b"after restart"
        with event.open("ab") as handle:
            handle.write(_start(2) + _image_record(2, replacement, 60.0))
        assert fs.readdir(_IMAGES) == [".", "..", "000001.png", "000002.png"]
        assert fs.read(f"{_IMAGES}/000001.png", len(PNG), 0) == PNG
        assert fs.read(primary, len(replacement), 0) == replacement
        assert fs.getattr(primary)["st_mtime"] == 60.0
        with pytest.raises(OSError) as missing:
            fs.getattr(renamed)
        assert missing.value.errno == ENOENT
        _assert_media(fs, primary, winning, 40.0, primary_handle, offset=5)
        _assert_media(fs, renamed, nested, 50.0, renamed_handle, offset=5)
    finally:
        fs.release(renamed, renamed_handle)
        fs.release(primary, primary_handle)
