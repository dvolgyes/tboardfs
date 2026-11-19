"""The raw view creates frame nodes only when a file is requested."""

from errno import EIO
import json
from pathlib import Path
from typing import NoReturn

import pytest

from tboardfs import TensorBoardFS
from tboardfs.file_tree import SingleEventTree
from tboardfs.raw_frames import RawFrames, _RawFrameChildren

from test_open_handles import _append

_EVENTS = "events.out.tfevents.host"


def _name(offset: int) -> str:
    return f"{offset:020d}.tfrecord"


def test_raw_listing_lookup_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / _EVENTS
    _append(source, 0)
    first = source.read_bytes()
    _append(source, 11)
    original = source.read_bytes()
    names = [_name(0), _name(len(first))]
    directory = f"/raw/{_EVENTS}"
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)

    def materialized(*args: object) -> NoReturn:
        pytest.fail("building or listing a raw tree materialized a frame")

    with monkeypatch.context() as patch:
        patch.setattr(_RawFrameChildren, "_node", materialized)
        before = fs.virtual_tree()
        assert fs.readdir(directory) == [".", "..", *names]
        assert fs.readdir(directory) == [".", "..", *names]
        assert fs.getattr(directory)["st_mtime"] == 0

    before_source = before["children"]["raw"]["children"][_EVENTS]["children"]
    assert list(before_source) == names
    for alias in (
        "0.tfrecord",
        "+" + names[0][1:],
        "٠" + names[0][1:],
        "0" + names[0],
        names[0].upper(),
        _name(1),
        _name(len(original)),
        _name(10**20 - 1),
    ):
        with pytest.raises(FileNotFoundError):
            fs.getattr(f"{directory}/{alias}")
    assert fs.getattr(f"{directory}/{names[0]}")["st_size"] == len(first)
    assert (
        fs.read(f"{directory}/{names[1]}", len(original), 0) == original[len(first) :]
    )

    _append(source, 200)
    new_name = _name(len(original))
    assert fs.readdir(directory) == [".", "..", *names, new_name]
    assert list(before_source) == names and new_name not in before_source
    assert RawFrames.read(before_source[names[0]]) == first

    source.unlink()
    with pytest.raises(FileNotFoundError):
        fs.getattr(f"{directory}/{names[0]}")
    with pytest.raises(OSError) as removed:
        RawFrames.read(before_source[names[0]])
    assert removed.value.errno == EIO


def test_raw_export_preserves_long_names_and_validates_original_frame(
    tmp_path: Path,
) -> None:
    source = tmp_path / (_EVENTS + "%" * 100)
    _append(source, 0)
    first = source.read_bytes()
    _append(source, 33)
    all_frames = source.read_bytes()
    tree = SingleEventTree(source)
    first_path, second_path = tree.list_file_paths(view="raw")
    prefix, basename = first_path.rsplit("/", 1)[0].rsplit("/", 1)
    assert prefix == "/raw"
    assert json.loads(tree.read_file("/.tboardfs-names.json")) == {
        basename: source.name
    }
    assert tree.list_file_paths(prefix=f"/raw/{basename}") == [first_path, second_path]
    assert all(not path.startswith("/raw/") for path in tree.list_file_paths())
    assert tree.read_file(first_path) == first
    assert tree.read_file(second_path) == all_frames[len(first) :]
    destination = tmp_path / "output"
    assert tree.copy_all(destination, view="raw") == (2, [])
    assert (destination / first_path.lstrip("/")).read_bytes() == first
    assert (destination / second_path.lstrip("/")).read_bytes() == all_frames[
        len(first) :
    ]

    with source.open("r+b") as handle:
        handle.seek(12)
        handle.write(bytes([first[12] ^ 1]))
    with pytest.raises(OSError) as corrupt:
        tree.read_file(first_path)
    assert corrupt.value.errno == EIO
    assert tree.read_file(second_path) == all_frames[len(first) :]


def test_empty_source_and_raw_named_nested_run(tmp_path: Path) -> None:
    _append(tmp_path / _EVENTS, 0)
    (tmp_path / (_EVENTS + ".empty")).touch()
    (tmp_path / "raw").mkdir()
    _append(tmp_path / "raw" / _EVENTS, 100)
    fs = TensorBoardFS(tmp_path)
    assert fs.readdir("/raw") == [".", "..", _EVENTS]
    assert fs.readdir(f"/raw/{_EVENTS}") == [".", "..", _name(0)]
    assert fs.readdir(f"/raw~2/raw/{_EVENTS}") == [".", "..", _name(0)]
