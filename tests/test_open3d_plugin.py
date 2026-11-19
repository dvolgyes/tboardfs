"""Verify Open3D metadata, range indexes, and lossless sidecar exports."""

import json
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS
from tboardfs.crc32c import masked_crc32c
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords
from tboardfs.render_node import render_node

from test_incremental_parse import _files, _frame
from test_proto3_reference import _fields, _values

RUN = Path(__file__).parent / "data" / "producer-cases" / "open3d"
ASSETS = "/plugins/Open3D/~assets"


def _event(
    run: Path,
    name: bytes = b"sample.msgpack",
    plugin: str = "Open3D",
    *,
    version: int = 14,
    start: bytes = b"",
) -> None:
    event = compat_pb2.Event(step=1)
    value = event.summary.value.add(tag="sample")
    value.metadata.plugin_data.plugin_name = plugin
    value.tensor.dtype = 7
    value.tensor.tensor_shape.SetInParent()
    size = start + b"\x10\x03\x20\x03\x28\x02"
    index = b"\x0a" + bytes((len(name),)) + name + b"\x12" + bytes((len(size),)) + size
    value.tensor.string_val.append(bytes((8, version, 26, len(index))) + index)
    (run / "events.out.tfevents.test").write_bytes(_frame(event))


def test_open3d_missing_then_appearing_asset(tmp_path: Path) -> None:
    _event(tmp_path)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    manifest_path = "/plugins/Open3D/sample/000001.manifest.json"
    assert json.loads(fs.read(manifest_path, 4096, 0))["asset"] is None
    directory = tmp_path / "plugins" / "Open3D"
    directory.mkdir(parents=True)
    asset = directory / "sample.msgpack"
    asset.write_bytes(b"tiny")
    assert json.loads(fs.read(manifest_path, 4096, 0))["asset"] is None
    asset.write_bytes(b"abcde")
    manifest = json.loads(fs.read(manifest_path, 4096, 0))
    assert manifest == {
        "asset": ASSETS.lstrip("/") + "/sample.msgpack",
        "filename": "sample.msgpack",
        "version": 14,
        "property_references": {},
        "start_size": [
            {
                "start": 0,
                "size": 3,
                "masked_crc32c": 0,
                "aux_start": 3,
                "aux_size": 2,
                "aux_masked_crc32c": 0,
            }
        ],
    }
    assert fs.read(f"{ASSETS}/sample.msgpack", 2, 3) == b"de"
    assert fs.getattr(f"{ASSETS}/sample.msgpack")["st_size"] == 5
    asset.unlink()
    assert json.loads(fs.read(manifest_path, 4096, 0))["asset"] is None


@pytest.mark.parametrize(
    ("name", "plugin", "version"),
    [
        (b"../sample.msgpack", "Open3D", 14),
        (b"/sample.msgpack", "Open3D", 14),
        (b"sample.msgpack", "open3d", 14),
        (b"sample.msgpack", "Open3D", 15),
    ],
)
def test_open3d_rejects_ambiguous_descriptors(
    tmp_path: Path, name: bytes, plugin: str, version: int
) -> None:
    _event(tmp_path, name, plugin, version=version)
    fs = TensorBoardFS(tmp_path)
    assert fs.readdir(f"/plugins/{plugin}/sample")[2:] == ["000001.json"]


def test_open3d_retains_uint64_offset_without_resolving_out_of_bounds(
    tmp_path: Path,
) -> None:
    _event(tmp_path, start=b"\x08" + b"\xff" * 9 + b"\x01")
    directory = tmp_path / "plugins" / "Open3D"
    directory.mkdir(parents=True)
    (directory / "sample.msgpack").write_bytes(b"abcde")
    fs = TensorBoardFS(tmp_path)
    path = "/plugins/Open3D/sample/000001.manifest.json"
    manifest = json.loads(fs.read(path, 4096, 0))
    assert manifest["start_size"][0]["start"] == 2**64 - 1
    assert manifest["asset"] is None


def test_open3d_only_scans_real_files_in_writer_directory(tmp_path: Path) -> None:
    _event(tmp_path)
    run = tmp_path / "plugins" / "Open3D"
    run.mkdir(parents=True)
    outside = tmp_path / "outside.msgpack"
    outside.write_bytes(b"abcde")
    (run / "sample.msgpack").symlink_to(outside)
    (run / "readme.txt").write_text("not a sidecar")
    wrong_case = run.with_name("open3d")
    wrong_case.mkdir()
    (wrong_case / "sample.msgpack").write_bytes(b"abcde")
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert "~assets" not in fs.readdir("/plugins/Open3D")
    path = "/plugins/Open3D/sample/000001.manifest.json"
    assert json.loads(fs.read(path, 4096, 0))["asset"] is None
    (run / "sample.msgpack").unlink()
    (run / "readme.txt").unlink()
    run.rmdir()
    run.symlink_to(wrong_case, target_is_directory=True)
    assert "~assets" not in fs.readdir("/plugins/Open3D")


def test_open3d_stale_node_does_not_follow_replacement(tmp_path: Path) -> None:
    _event(tmp_path)
    directory = tmp_path / "plugins" / "Open3D"
    directory.mkdir(parents=True)
    asset = directory / "sample.msgpack"
    asset.write_bytes(b"abcde")
    outside = tmp_path / "outside.msgpack"
    outside.write_bytes(b"private")
    fs = TensorBoardFS(tmp_path)
    node, key = fs._locate(f"{ASSETS}/sample.msgpack")
    asset.unlink()
    asset.symlink_to(outside)
    with pytest.raises(OSError):
        render_node(node)
    with pytest.raises(FileNotFoundError):
        fs._node_size(node, key)


def test_native_open3d_writer_ranges_match_original_bytes() -> None:
    fs = TensorBoardFS(RUN)
    events = RUN / "events.out.tfevents.open3d"
    with events.open("rb") as stream:
        summaries = [r.payload for r in iter_tfrecords(stream) if _fields(r.payload, 5)]
    assert len(summaries) == 4
    assert b"".join(fs.read(path, 65536, 0) for path in sorted(_files(fs, "/raw"))) == (
        events.read_bytes()
    )
    for summary in summaries:
        _assert_native_summary(fs, summary)


def _assert_native_summary(fs: TensorBoardFS, summary: bytes) -> None:
    (step,) = _fields(summary, 2)
    ((tag, value),) = _values(summary).items()
    (tensor,) = _fields(value, 8)
    (descriptor,) = _fields(tensor, 8)
    (batch,) = _fields(descriptor, 3)
    filename = _fields(batch, 1)[0].decode()
    expected = (RUN / "plugins" / "Open3D" / filename).read_bytes()
    path = f"/plugins/Open3D/{tag}/{step:06d}.manifest.json"
    manifest = json.loads(fs.read(path, 4096, 0))
    asset_path = f"{ASSETS}/{filename}"
    assert manifest["asset"] == asset_path.lstrip("/")
    assert expected == fs.read(asset_path, len(expected) + 1, 0)
    assert manifest["filename"] == filename
    assert manifest["version"] == _fields(descriptor, 1)[0] == 14
    refs = _fields(descriptor, 2)
    assert (
        manifest["property_references"]
        == ({"vertex_positions": _fields(refs[0], 2)[0]} if refs else {})
        == ({"vertex_positions": 1} if step == 2 else {})
    )
    names = (
        "start",
        "size",
        "masked_crc32c",
        "aux_start",
        "aux_size",
        "aux_masked_crc32c",
    )
    for record, bounds in zip(_fields(batch, 2), manifest["start_size"], strict=True):
        assert bounds == {
            name: (_fields(record, i) or [0])[0] for i, name in enumerate(names, 1)
        }
        data = fs.read(asset_path, bounds["size"], bounds["start"])
        assert len(data) == bounds["size"] > 0
        assert masked_crc32c(data) == bounds["masked_crc32c"]
