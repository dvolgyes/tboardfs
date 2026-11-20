"""Name-only operations do not expand file nodes or change the resolved namespace."""

from pathlib import Path
from typing import Any

import pytest
from test_incremental_parse import _frame

from scripts.scale_corpus import _ScaleCorpus
from tboardfs import TensorBoardFS
from tboardfs.file_tree import SingleEventTree
from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.proto import compat_pb2
from tboardfs.render_node import render_node


def _image(tag: str, step: int, pixel: int, timestamp: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=timestamp)
    image = event.summary.value.add(tag=tag).image
    image.width = image.height = 1
    image.colorspace = 3
    image.encoded_image_string = _ScaleCorpus.small_png(pixel)
    return _frame(event)


def test_name_only_operations_leave_typed_files_unmaterialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "events.out.tfevents.fixture"
    source.write_bytes(b"".join(_ScaleCorpus.event_frame(i) for i in range(20)))
    observed = []
    node = LeafSpec.node

    def trace(value: LeafValue) -> dict[str, Any]:
        if not isinstance(value, dict):
            observed.append(value)
        return node(value)

    monkeypatch.setattr(LeafSpec, "node", staticmethod(trace))
    fs = TensorBoardFS(tmp_path)
    names = [f"{i:06d}.png" for i in range(20)]
    assert fs.readdir("/images/series") == [".", "..", *names]
    assert fs.getattr("/images/series")["st_mtime"] == 119.0
    cli = SingleEventTree(source)
    assert cli.list_file_paths(prefix="/images") == [
        f"/images/series/{n}" for n in names
    ]
    assert not observed
    assert fs.read("/images/series/000019.png", 100, 0) == _ScaleCorpus.small_png(19)
    assert {value.step for value in observed if isinstance(value, BinaryEntry)} == {19}


def test_new_step_file_displaces_a_binary_directory_on_append(tmp_path: Path) -> None:
    source = tmp_path / "events.out.tfevents.images"
    tag = "series/000004.png/sub"
    source.write_bytes(_image("series", 1, 1, 10) + _image(tag, 3, 3, 30))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    prior = fs.virtual_tree()["children"]["images"]["children"]["series"]["children"]
    nested = "sub/000003.png"
    assert fs.read(
        f"/images/series/000004.png/{nested}", 100, 0
    ) == _ScaleCorpus.small_png(3)
    with source.open("ab") as stream:
        stream.write(_image("series", 4, 4, 40))
    expected = [".", "..", "000001.png", "000004.png", "000004.png~2"]
    for view in (fs, TensorBoardFS(tmp_path)):
        assert view.readdir("/images/series") == expected
        assert view.read("/images/series/000004.png", 100, 0) == _ScaleCorpus.small_png(
            4
        )
        assert view.read(
            f"/images/series/000004.png~2/{nested}", 100, 0
        ) == _ScaleCorpus.small_png(3)
        assert view.getattr("/images/series/000004.png~2")["st_mtime"] == 30
    old = prior["000004.png"]["children"]["sub"]["children"]["000003.png"]
    assert render_node(old) == _ScaleCorpus.small_png(3)


def test_deep_tag_can_be_listed_and_extended(tmp_path: Path) -> None:
    tag = "/".join(["x"] * 1300)
    source = tmp_path / "events.out.tfevents.deep"
    source.write_bytes(_image(tag, 5, 2, 17))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert fs.getattr(f"/images/{tag}")["st_mtime"] == 17
    assert fs.read(f"/images/{tag}/000005.png", 100, 0) == _ScaleCorpus.small_png(2)
    with source.open("ab") as stream:
        stream.write(_image(tag, 6, 3, 19))
    assert fs.getattr(f"/images/{tag}")["st_mtime"] == 19
    assert fs.read(f"/images/{tag}/000006.png", 100, 0) == _ScaleCorpus.small_png(3)
    assert SingleEventTree(source).list_file_paths() == [
        f"/images/{tag}/000005.png",
        f"/images/{tag}/000006.png",
    ]
