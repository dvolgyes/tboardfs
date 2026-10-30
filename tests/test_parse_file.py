import base64
from collections.abc import Iterator
import errno
import json
import os
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np
import pytest

from tboardfs import FIXED_TABS, TensorBoardFS, find_tensorboard_files, parse_file
from tboardfs.classify import detect_extension
from tboardfs.file_tree import SingleEventTree
from tboardfs.proto import compat_pb2
from tboardfs.summary import parse_event
from tboardfs.tables import export_table
from tboardfs.crc32c import masked_crc32c
from tboardfs.model import Record, TFRecordError
import tboardfs.indexer


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, wire_type: int, payload: bytes) -> bytes:
    return _varint((number << 3) | wire_type) + payload


def _bytes_field(number: int, value: bytes) -> bytes:
    return _field(number, 2, _varint(len(value)) + value)


def _string_field(number: int, value: str) -> bytes:
    return _bytes_field(number, value.encode())


def _double_field(number: int, value: float) -> bytes:
    return _field(number, 1, struct.pack("<d", value))


def _float_field(number: int, value: float) -> bytes:
    return _field(number, 5, struct.pack("<f", value))


def _int_field(number: int, value: int) -> bytes:
    return _field(number, 0, _varint(value))


def _summary_value(tag: str, *fields: bytes) -> bytes:
    return _bytes_field(1, _string_field(1, tag) + b"".join(fields))


def _simple_value(tag: str, value: float) -> bytes:
    return _summary_value(tag, _float_field(2, value))


def _image_value(tag: str, encoded: bytes) -> bytes:
    image = _int_field(1, 1) + _int_field(2, 2) + _int_field(3, 3)
    image += _bytes_field(4, encoded)
    return _summary_value(tag, _bytes_field(4, image))


def _tensor_scalar_value(
    tag: str, dtype: int, field_number: int, value_payload: bytes
) -> bytes:
    tensor = _int_field(1, dtype) + _field(field_number, 0, value_payload)
    return _summary_value(tag, _bytes_field(8, tensor))


def _tensor_content_scalar_value(tag: str, dtype: int, content: bytes) -> bytes:
    tensor = _int_field(1, dtype) + _bytes_field(4, content)
    return _summary_value(tag, _bytes_field(8, tensor))


def _tensor_string_value(
    tag: str, value: bytes, *, plugin_name: str | None = None
) -> bytes:
    tensor = _int_field(1, 7) + _bytes_field(8, value)
    fields = [_bytes_field(8, tensor)]
    if plugin_name is not None:
        plugin_data = _string_field(1, plugin_name)
        metadata = _bytes_field(1, plugin_data)
        fields.append(_bytes_field(9, metadata))
    return _summary_value(tag, *fields)


def _event(wall_time: float, step: int, values: list[bytes]) -> bytes:
    summary = b"".join(values)
    return _double_field(1, wall_time) + _int_field(2, step) + _bytes_field(5, summary)


def _header(length: int) -> bytes:
    length_bytes = struct.pack("<Q", length)
    return length_bytes + struct.pack("<I", masked_crc32c(length_bytes))


def _record(payload: bytes) -> bytes:
    return _header(len(payload)) + payload + struct.pack("<I", masked_crc32c(payload))


def _event_record(
    wall_time: float,
    step: int,
    values: list[object],
) -> bytes:
    Event = getattr(compat_pb2, "Event")
    Summary = getattr(compat_pb2, "Summary")
    return _record(
        Event(
            wall_time=wall_time,
            step=step,
            summary=Summary(value=values),
        ).SerializeToString()
    )


def _tensor_value(tag: str, dtype: int, field_name: str, value: int) -> object:
    SummaryValue = getattr(compat_pb2, "Summary").Value
    TensorProto = getattr(compat_pb2, "TensorProto")
    tensor = TensorProto(dtype=dtype)
    getattr(tensor, field_name).append(value)
    return SummaryValue(tag=tag, tensor=tensor)


def test_find_tensorboard_files_recurses_and_returns_paths(tmp_path: Path) -> None:
    """Event-file discovery recurses and ignores non-event paths."""
    root_event = tmp_path / "events.out.tfevents.1"
    nested_dir = tmp_path / "run" / "nested"
    nested_dir.mkdir(parents=True)
    nested_event = nested_dir / "events.out.tfevents.2"
    ignored = nested_dir / "notes.txt"
    event_named_dir = tmp_path / "events.out.tfevents.dir"

    root_event.touch()
    nested_event.touch()
    ignored.touch()
    event_named_dir.mkdir()

    assert find_tensorboard_files(tmp_path) == sorted([root_event, nested_event])


def test_parse_file_collects_all_scalar_occurrences(tmp_path: Path) -> None:
    """Scalar parsing keeps every occurrence and carries epoch values."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(
            _event(1000.0, 1, [_simple_value("epoch", 7.0), _simple_value("loss", 2.5)])
        )
        + _record(
            _event(1002.5, 2, [_simple_value("loss", 2.0), _simple_value("loss", 1.5)])
        )
    )

    result = parse_file(path)
    loss = result["scalars"]["loss"]

    np.testing.assert_array_equal(loss["epoch"], np.array([7.0, 7.0, 7.0]))
    np.testing.assert_array_equal(loss["step"], np.array([1, 2, 2]))
    np.testing.assert_array_equal(loss["wall_time"], np.array([1000.0, 1002.5, 1002.5]))
    np.testing.assert_array_equal(loss["relative_time"], np.array([0.0, 2.5, 2.5]))
    np.testing.assert_array_almost_equal(
        loss["value"], np.array([2.5, 2.0, 1.5], dtype=np.float32)
    )


def test_parse_file_indexes_binary_record_payload_with_last_wins(
    tmp_path: Path,
) -> None:
    """Binary index compatibility keeps the latest offset by tag."""
    first = _event(10.0, 1, [_image_value("image", b"first-image")])
    second = _event(12.0, 2, [_image_value("image", b"second-image")])
    data = _record(first) + _record(second)
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(data)

    result = parse_file(path)

    offset, size = result["binaries"]["images"]["image"]
    expected_offset = 12 + len(first) + 4 + 12
    assert offset == expected_offset
    assert size == len(second)
    assert path.read_bytes()[offset : offset + size] == second


def test_detect_extension_uses_magic_for_native_formats() -> None:
    """Magic detection returns native extensions and leaves unknown blobs binary."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAF"
        "AQH/KS2zWQAAAABJRU5ErkJggg=="
    )
    json_blob = b'{"plugin": "hparams", "value": 3}\n'
    unknown = b"\x01\x02not-a-native-format\x00\xff"

    assert detect_extension(png, "images") == "png"
    assert detect_extension(json_blob, "tensors") == "json"
    assert detect_extension(unknown, "tensors") == "bin"


def test_export_table_json_spells_infinities_and_nulls_nan() -> None:
    """Strict JSON keeps open bucket edges; NaN, a missing cell, becomes null."""
    series = {
        "bucket_left": np.asarray([float("-inf"), 0.0, np.nan], dtype=np.float64),
        "bucket_right": np.asarray([0.0, 1.0, float("inf")], dtype=np.float64),
    }

    rows = json.loads(export_table(series, "json"))

    assert rows == [
        {"bucket_left": "-Infinity", "bucket_right": 0.0},
        {"bucket_left": 0.0, "bucket_right": 1.0},
        {"bucket_left": None, "bucket_right": "Infinity"},
    ]


def test_filesystem_exposes_magic_detected_tensor_blob_extension(
    tmp_path: Path,
) -> None:
    """Filesystem paths use detected native extensions for tensor blobs."""
    path = tmp_path / "events.out.tfevents"
    json_blob = b'{"plugin": "mesh", "vertices": [1, 2, 3]}\n' * 40
    path.write_bytes(
        _record(_event(100.0, 7, [_tensor_string_value("tensor/blob", json_blob)]))
    )
    fs = TensorBoardFS(tmp_path, step_digits=3)

    assert "007.json" in fs.readdir("/tensors/tensor/blob")
    assert fs.read("/tensors/tensor/blob/007.json", 10000, 0) == json_blob


def test_parse_file_reads_summary_metadata_plugin_name(tmp_path: Path) -> None:
    """Summary metadata plugin names route JSON entries to plugin tabs."""
    path = tmp_path / "events.out.tfevents"
    value = _tensor_string_value(
        "shape/triangle", b'{"vertices": []}', plugin_name="mesh"
    )
    path.write_bytes(_record(_event(100.0, 2, [value])))

    result = parse_file(path)

    assert result["json_entries"][0].tab == "meshes"
    assert result["json_entries"][0].payload["plugin_name"] == "mesh"


def test_parse_file_skips_plugin_binary_without_blob(tmp_path: Path) -> None:
    """Plugin metadata without tensor bytes is not listed as a mesh file."""
    path = tmp_path / "events.out.tfevents"
    plugin_data = _string_field(1, "mesh")
    metadata = _bytes_field(1, plugin_data)
    value = _summary_value("shape/config", _bytes_field(9, metadata))
    path.write_bytes(_record(_event(100.0, 1, [value])))

    result = parse_file(path)
    fs = TensorBoardFS(tmp_path, step_digits=3)

    assert result["binary_entries"] == []
    assert fs.readdir("/meshes") == [".", ".."]


def test_parse_file_reads_integer_tensor_scalars(tmp_path: Path) -> None:
    """Integer tensor values parse as scalar series."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(_event(5.0, 9, [_tensor_scalar_value("count", 3, 7, _varint(42))]))
    )

    result = parse_file(path)
    count = result["scalars"]["count"]

    np.testing.assert_array_equal(count["step"], np.array([9]))
    np.testing.assert_array_equal(count["value"], np.array([42], dtype=np.int32))


def test_parse_event_reads_negative_integer_tensors_as_signed_values() -> None:
    """Vendored protobuf parsing returns signed Python integers."""
    Event = getattr(compat_pb2, "Event")
    Summary = getattr(compat_pb2, "Summary")
    event = Event(
        wall_time=5.0,
        step=9,
        summary=Summary(
            value=[
                _tensor_value("neg_int32", 3, "int_val", -42),
                _tensor_value("neg_int64", 9, "int64_val", -(2**40)),
            ]
        ),
    )

    parsed = parse_event(event.SerializeToString())
    int32_values = parsed["values"][0]["tensor"]["values"]
    int64_values = parsed["values"][1]["tensor"]["values"]

    assert int32_values[7] == [-42]
    assert int64_values[10] == [-(2**40)]


@pytest.mark.parametrize(
    ("tag", "dtype", "field_name", "expected", "expected_dtype"),
    [
        ("neg_int32", 3, "int_val", -42, np.dtype("int32")),
        ("neg_int64", 9, "int64_val", -(2**40), np.dtype("int64")),
    ],
)
def test_parse_file_reads_negative_integer_tensor_scalars(
    tmp_path: Path,
    tag: str,
    dtype: int,
    field_name: str,
    expected: int,
    expected_dtype: np.dtype,
) -> None:
    """Negative integer tensor values parse as signed scalar series."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _event_record(5.0, 9, [_tensor_value(tag, dtype, field_name, expected)])
    )

    result = parse_file(path)
    scalar = result["scalars"][tag]

    assert scalar["value"].dtype == expected_dtype
    np.testing.assert_array_equal(
        scalar["value"], np.array([expected], dtype=expected_dtype)
    )


def test_parse_file_reads_tensor_content_scalar(tmp_path: Path) -> None:
    """Fixed-width tensor content parses as scalar series."""
    path = tmp_path / "events.out.tfevents"
    value = _tensor_content_scalar_value("accuracy", 1, struct.pack("<f", 0.75))
    path.write_bytes(_record(_event(5.0, 9, [value])))

    result = parse_file(path)
    accuracy = result["scalars"]["accuracy"]

    assert accuracy["value"].dtype == np.dtype("float32")
    np.testing.assert_array_almost_equal(
        accuracy["value"], np.array([0.75], dtype=np.float32)
    )


def test_parse_file_mixed_scalar_types_fall_back_to_double(tmp_path: Path) -> None:
    """Mixed scalar dtypes promote the exported value array."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(_event(1.0, 1, [_tensor_scalar_value("mixed", 3, 7, _varint(42))]))
        + _record(_event(2.0, 2, [_simple_value("mixed", 2.5)]))
    )

    result = parse_file(path)

    assert result["scalars"]["mixed"]["value"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(
        result["scalars"]["mixed"]["value"], np.array([42.0, 2.5])
    )


def test_parse_file_rejects_truncated_record(tmp_path: Path) -> None:
    """Strict parsing rejects truncated TFRecords."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_header(100) + b"too-short")

    with pytest.raises(ValueError, match="truncated TFRecord payload"):
        parse_file(path)


def test_filesystem_lists_fixed_tabs_for_empty_source(tmp_path: Path) -> None:
    """Empty sources still expose fixed top-level tabs."""
    fs = TensorBoardFS(tmp_path)

    listing = fs.readdir("/")

    assert sorted(listing) == sorted([".", "..", *FIXED_TABS])


def test_filesystem_exposes_nested_scalar_formats(tmp_path: Path) -> None:
    """Nested scalar tags become nested virtual paths."""
    path = tmp_path / "run-a" / "events.out.tfevents"
    path.parent.mkdir()
    path.write_bytes(_record(_event(100.0, 3, [_simple_value("train/loss", 1.25)])))
    fs = TensorBoardFS(tmp_path, step_digits=4)

    assert "run-a" in fs.readdir("/")
    assert "train" in fs.readdir("/run-a/scalars")
    assert sorted(fs.readdir("/run-a/scalars/train")) == [
        ".",
        "..",
        "loss.json",
        "loss.npz",
        "loss.tsv",
    ]

    rows = json.loads(fs.read("/run-a/scalars/train/loss.json", 10000, 0))
    assert rows == [
        {
            "epoch": None,
            "step": 3,
            "wall_time": 100.0,
            "relative_time": 0.0,
            "value": pytest.approx(1.25),
        }
    ]

    tsv = fs.read("/run-a/scalars/train/loss.tsv", 10000, 0).decode()
    assert tsv.splitlines()[0] == "epoch\tstep\twall_time\trelative_time\tvalue"


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("symlink", ("/scalars/loss.json", "/scalars/loss.xlsx")),
        ("unlink", ("/.cache",)),
        ("unlink", ("/.in_memory",)),
        ("unlink", ("/scalars/loss.json",)),
    ],
)
def test_filesystem_rejects_links_and_unlinks_as_read_only(
    tmp_path: Path, operation: str, args: tuple[str, ...]
) -> None:
    """symlink and unlink fail with EROFS and leave the tree unchanged."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path)
    before = sorted(fs.readdir("/scalars"))

    with pytest.raises(OSError) as error:
        getattr(fs, operation)(*args)

    assert error.value.errno == errno.EROFS
    assert sorted(fs.readdir("/scalars")) == before


def test_filesystem_ignores_truncated_trailing_record_during_refresh(
    tmp_path: Path,
) -> None:
    """Refresh mode ignores a truncated trailing record."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(_event(100.0, 1, [_simple_value("loss", 2.0)]))
        + _header(100)
        + b"too-short"
    )
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)

    rows = json.loads(fs.read("/scalars/loss.json", 10000, 0))

    assert len(rows) == 1
    assert rows[0]["step"] == 1


def test_filesystem_incrementally_parses_appended_records(tmp_path: Path) -> None:
    """Refresh mode parses only newly appended complete records."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)

    assert len(json.loads(fs.read("/scalars/loss.json", 10000, 0))) == 1
    first_cache = fs._state.runs[()].files[path]
    first_processed_pos = first_cache.progress.processed_pos
    path.write_bytes(
        path.read_bytes() + _record(_event(102.0, 2, [_simple_value("loss", 1.5)]))
    )

    rows = json.loads(fs.read("/scalars/loss.json", 10000, 0))
    second_cache = fs._state.runs[()].files[path]

    assert first_processed_pos < second_cache.progress.processed_pos
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["relative_time"] for row in rows] == [0.0, 2.0]


def test_filesystem_reads_records_appended_during_a_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records appended while a parse runs are read on the next refresh."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    original = tboardfs.indexer.iter_tfrecords
    appended = []

    def append_during_parse(handle: BinaryIO) -> Iterator[Record]:
        for record in original(handle):
            if not appended:
                appended.append(_event(102.0, 2, [_simple_value("loss", 1.5)]))
                with path.open("ab") as writer:
                    writer.write(_record(appended[0]))
            yield record

    monkeypatch.setattr(tboardfs.indexer, "iter_tfrecords", append_during_parse)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert len(json.loads(fs.read("/scalars/loss.json", 10000, 0))) == 1
    monkeypatch.setattr(tboardfs.indexer, "iter_tfrecords", original)

    rows = json.loads(fs.read("/scalars/loss.json", 10000, 0))

    assert [row["step"] for row in rows] == [1, 2]


def test_filesystem_lists_projector_and_profile_sidecars(tmp_path: Path) -> None:
    """Projector and profile sidecars appear in their virtual tabs."""
    event = tmp_path / "run" / "events.out.tfevents"
    event.parent.mkdir()
    event.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    projector = event.parent / "projector_config.pbtxt"
    profile = event.parent / "plugins" / "profile" / "trace.json"
    profile.parent.mkdir(parents=True)
    projector.write_text("projector-config")
    profile.write_text("profile-trace")
    fs = TensorBoardFS(tmp_path)

    assert "projector_config.pbtxt" in fs.readdir("/run/projector")
    assert "trace.json" in fs.readdir("/run/profile")
    assert (
        fs.read("/run/projector/projector_config.pbtxt", 1000, 0) == b"projector-config"
    )
    assert fs.read("/run/profile/trace.json", 1000, 0) == b"profile-trace"


def test_filesystem_empty_source_does_not_scan_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source without event files exposes no sidecars from the cwd."""
    cwd = tmp_path / "cwd"
    trace = cwd / "plugins" / "profile" / "trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("unrelated")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.chdir(cwd)
    fs = TensorBoardFS(source)

    assert "trace.json" not in fs.readdir("/profile")


def test_filesystem_sidecars_stay_with_their_nested_run(tmp_path: Path) -> None:
    """A nested run's sidecars do not also appear in the parent run."""
    record = _record(_event(100.0, 1, [_simple_value("loss", 2.0)]))
    (tmp_path / "events.out.tfevents.root").write_bytes(record)
    child = tmp_path / "child"
    trace = child / "plugins" / "profile" / "trace.json"
    trace.parent.mkdir(parents=True)
    (child / "events.out.tfevents.child").write_bytes(record)
    trace.write_text("child-trace")
    fs = TensorBoardFS(tmp_path)

    assert "trace.json" not in fs.readdir("/profile")
    assert fs.read("/child/profile/trace.json", 1000, 0) == b"child-trace"


def test_filesystem_skips_sidecar_symlinks_outside_source(tmp_path: Path) -> None:
    """Sidecar symlinks that resolve outside the source are not served."""
    secret = tmp_path / "secret.json"
    secret.write_text("secret")
    run = tmp_path / "source" / "run"
    profile = run / "plugins" / "profile"
    profile.mkdir(parents=True)
    (run / "events.out.tfevents").write_bytes(
        _record(_event(100.0, 1, [_simple_value("loss", 2.0)]))
    )
    (profile / "escape.json").symlink_to(secret)
    (profile / "outside").symlink_to(tmp_path, target_is_directory=True)
    (profile / "inside.json").write_text("inside")
    (profile / "alias.json").symlink_to(profile / "inside.json")
    fs = TensorBoardFS(tmp_path / "source")

    assert sorted(fs.readdir("/run/profile")) == [
        ".",
        "..",
        "alias.json",
        "inside.json",
    ]


def test_filesystem_accepts_mfusepy_positional_file_handles(tmp_path: Path) -> None:
    """FUSE callbacks accept file handles as positional arguments."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path)

    attrs = fs.getattr("/scalars/loss.json", None)
    listing = fs.readdir("/scalars", None)
    data = fs.read("/scalars/loss.json", 10000, 0, None)

    assert attrs["st_size"] == len(data)
    assert "loss.json" in listing
    with pytest.raises(OSError):
        fs.create("/blocked", 0, None)
    with pytest.raises(OSError):
        fs.write("/scalars/loss.json", b"{}", 0, None)
    with pytest.raises(OSError):
        fs.truncate("/scalars/loss.json", 0, None)


@pytest.mark.parametrize("owner", [(0, 0), (1234, 5678)])
def test_filesystem_gives_every_stat_the_owner(
    tmp_path: Path, owner: tuple[int, int]
) -> None:
    """Directories, files and open handles all report the configured owner."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path, owner=owner)
    handle = fs.open("/scalars/loss.json", os.O_RDONLY)

    stats = [
        fs.getattr("/scalars"),
        fs.getattr("/scalars/loss.json"),
        fs.getattr("/scalars/loss.json", handle),
    ]

    assert [(st["st_uid"], st["st_gid"]) for st in stats] == [owner] * 3


@pytest.mark.parametrize(
    "tag",
    ["../../ESCAPED", "a/../b", "./x", "back\\slash", "nul\0byte", "ctl\x1fchar"],
)
def test_virtual_paths_never_contain_traversal_components(
    tmp_path: Path, tag: str
) -> None:
    """Hostile tags and plugin names stay inside one virtual path component."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(
            _event(
                100.0,
                1,
                [
                    _simple_value(tag, 1.0),
                    _tensor_string_value(tag, b"x", plugin_name="../" + tag),
                ],
            )
        )
    )
    tree = SingleEventTree(path)

    for virtual_path in tree.list_file_paths():
        parts = virtual_path.lstrip("/").split("/")
        assert all(part not in {"", ".", ".."} for part in parts), virtual_path
        assert all("\\" not in part for part in parts), virtual_path
        assert all(ch >= " " for part in parts for ch in part), virtual_path


def test_copy_all_writes_hostile_tags_inside_outdir(tmp_path: Path) -> None:
    """copy-all keeps every output file below the output directory."""
    source = tmp_path / "src" / "events.out.tfevents"
    source.parent.mkdir()
    source.write_bytes(_record(_event(100.0, 1, [_simple_value("../../ESCAPED", 1.0)])))
    outdir = tmp_path / "a" / "b" / "out"

    count, _ = SingleEventTree(source).copy_all(outdir)

    written = [p for p in tmp_path.rglob("*") if p.is_file() and p != source]
    assert count == len(written) > 0
    assert all(p.resolve().is_relative_to(outdir.resolve()) for p in written)


def test_copy_all_refuses_symlink_escape(tmp_path: Path) -> None:
    """copy-all refuses to follow an existing symlink out of the output directory."""
    source = tmp_path / "src" / "events.out.tfevents"
    source.parent.mkdir()
    source.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 1.0)])))
    outside = tmp_path / "outside"
    outside.mkdir()
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "scalars").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="outside the output directory"):
        SingleEventTree(source).copy_all(outdir, existing="overwrite")

    assert list(outside.iterdir()) == []


def _histogram_value(tag: str, limits: list[float], counts: list[float]) -> object:
    SummaryValue = getattr(compat_pb2, "Summary").Value
    HistogramProto = getattr(compat_pb2, "HistogramProto")
    histo = HistogramProto(min=0.0, max=limits[-1], bucket_limit=limits, bucket=counts)
    return SummaryValue(tag=tag, histo=histo)


def test_histogram_step_files_hold_only_their_step(tmp_path: Path) -> None:
    """Each per-step histogram file contains the buckets of that step only."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _event_record(100.0, 1, [_histogram_value("w", [1.0, 2.0], [3.0, 4.0])])
        + _event_record(101.0, 2, [_histogram_value("w", [1.0, 2.0], [5.0, 6.0])])
    )
    fs = TensorBoardFS(tmp_path, step_digits=1)

    step1 = json.loads(fs.read("/histograms/w/1.json", 100_000, 0))
    step2 = json.loads(fs.read("/histograms/w/2.json", 100_000, 0))

    assert [row["step"] for row in step1] == [1, 1]
    assert [row["count"] for row in step1] == [3.0, 4.0]
    assert [row["count"] for row in step2] == [5.0, 6.0]


def test_distribution_without_epoch_scalar_has_one_group_per_event(
    tmp_path: Path,
) -> None:
    """Distribution percentiles use all buckets of a histogram event."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _event_record(
            100.0, 1, [_histogram_value("w", [1.0, 2.0, 3.0, 4.0], [1.0] * 4)]
        )
    )
    fs = TensorBoardFS(tmp_path, step_digits=1)

    rows = json.loads(fs.read("/distributions/w/1.json", 100_000, 0))

    # TensorBoard interpolates in the bucket, so a uniform histogram is linear.
    assert [row["percentile"] for row in rows] == [
        0.0,
        6.68,
        15.87,
        30.85,
        50.0,
        69.15,
        84.13,
        93.32,
        100.0,
    ]
    assert [row["value"] for row in rows] == pytest.approx(
        [0.0, 0.2672, 0.6348, 1.234, 2.0, 2.766, 3.3652, 3.7328, 4.0]
    )


def test_distribution_keeps_repeated_step_events_apart(tmp_path: Path) -> None:
    """Two histogram events at one step give two percentile groups."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _event_record(100.0, 1, [_histogram_value("w", [1.0, 2.0], [1.0, 0.0])])
        + _event_record(105.0, 1, [_histogram_value("w", [1.0, 2.0], [1.0, 3.0])])
    )
    fs = TensorBoardFS(tmp_path, step_digits=1)

    rows = json.loads(fs.read("/distributions/w/1.json", 100_000, 0))

    assert [row["wall_time"] for row in rows] == [100.0] * 9 + [105.0] * 9
    medians = [row["value"] for row in rows if row["percentile"] == 50.0]
    assert medians == pytest.approx([1.0, 4 / 3])


def test_histogram_with_missing_step_field_is_exported_once(tmp_path: Path) -> None:
    """A step omitted on the wire still yields one non-empty step file."""
    Event = getattr(compat_pb2, "Event")
    Summary = getattr(compat_pb2, "Summary")
    value = _histogram_value("w", [1.0, 2.0], [1.0, 1.0])
    payloads = [
        Event(wall_time=wall, summary=Summary(value=[value])).SerializeToString()
        for wall in (100.0, 101.0)
    ]
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(b"".join(_record(payload) for payload in payloads))
    fs = TensorBoardFS(tmp_path, step_digits=1)

    names = [n for n in fs.readdir("/histograms/w") if n.endswith(".json")]
    assert len(names) == 1
    rows = json.loads(fs.read(f"/histograms/w/{names[0]}", 100_000, 0))
    dist = json.loads(fs.read(f"/distributions/w/{names[0]}", 100_000, 0))
    assert len(rows) == 4
    assert len(dist) == 2 * 9


def test_masked_crc32c_matches_reference_vectors() -> None:
    """CRC32C uses the Castagnoli polynomial and the TFRecord mask."""
    crc = 0xE3069283  # CRC32C("123456789"), RFC 3720 check value
    masked = (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    assert masked_crc32c(b"123456789") == masked


def _loss_record(step: int) -> bytes:
    return _record(_event(100.0 + step, step, [_simple_value("loss", float(step))]))


def test_corrupt_length_checksum_stops_tolerant_parse(tmp_path: Path) -> None:
    """A bad length CRC is corruption: strict mode fails, tolerant mode keeps the prefix."""
    path = tmp_path / "events.out.tfevents"
    bad = struct.pack("<Q", 5) + b"\xde\xad\xbe\xef" + b"hello" + b"\0\0\0\0"
    path.write_bytes(_loss_record(1) + bad + _loss_record(2))

    with pytest.raises(TFRecordError, match="checksum") as error:
        parse_file(path)
    assert error.value.corrupt
    tree = SingleEventTree(path, step_digits=1)

    rows = json.loads(tree.read_file("/scalars/loss.json"))
    assert [row["step"] for row in rows] == [1]


def test_huge_declared_length_is_truncation_without_allocation(
    tmp_path: Path,
) -> None:
    """A length larger than the file never allocates the declared size."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_loss_record(1) + _header(2**62) + b"tail")

    with pytest.raises(TFRecordError, match="truncated TFRecord payload") as error:
        parse_file(path)
    assert not error.value.corrupt
    tree = SingleEventTree(path, step_digits=1)

    rows = json.loads(tree.read_file("/scalars/loss.json"))
    assert [row["step"] for row in rows] == [1]


def test_malformed_payload_is_skipped_in_tolerant_mode(tmp_path: Path) -> None:
    """A framed record with an undecodable payload does not hide later records."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_loss_record(1) + _record(b"\xff\xff\xff") + _loss_record(2))

    tree = SingleEventTree(path, step_digits=1)

    rows = json.loads(tree.read_file("/scalars/loss.json"))
    assert [row["step"] for row in rows] == [1, 2]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_unreadable_event_file_does_not_break_other_runs(tmp_path: Path) -> None:
    """An unreadable event file affects only its own run."""
    good = tmp_path / "good" / "events.out.tfevents"
    bad = tmp_path / "bad" / "events.out.tfevents"
    for path in (good, bad):
        path.parent.mkdir()
        path.write_bytes(_loss_record(1))
    bad.chmod(0)
    fs = TensorBoardFS(tmp_path, step_digits=1)

    try:
        listing = fs.readdir("/")
        rows = json.loads(fs.read("/good/scalars/loss.json", 10_000, 0))
    finally:
        bad.chmod(0o644)

    assert {"good", "bad"} <= set(listing)
    assert [row["step"] for row in rows] == [1]
