import base64
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tboardfs import FIXED_TABS, TensorBoardFS, find_tensorboard_files, parse_file
from tboardfs.classify import detect_extension
from tboardfs.tables import export_table


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


def _record(payload: bytes) -> bytes:
    return struct.pack("<Q", len(payload)) + b"\0\0\0\0" + payload + b"\0\0\0\0"


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


def test_export_table_json_converts_non_finite_floats_to_null() -> None:
    """Strict JSON table export replaces non-finite floats with null."""
    series = {
        "bucket_left": np.asarray([float("-inf"), 0.0, 1.0], dtype=np.float64),
        "bucket_right": np.asarray([0.0, 1.0, float("inf")], dtype=np.float64),
    }

    rows = json.loads(export_table(series, "json"))

    assert rows == [
        {"bucket_left": None, "bucket_right": 0.0},
        {"bucket_left": 0.0, "bucket_right": 1.0},
        {"bucket_left": 1.0, "bucket_right": None},
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
    path.write_bytes(struct.pack("<Q", 100) + b"\0\0\0\0" + b"too-short")

    with pytest.raises(ValueError, match="truncated TFRecord payload"):
        parse_file(path)


def test_filesystem_lists_fixed_tabs_for_empty_source(tmp_path: Path) -> None:
    """Empty sources still expose fixed top-level tabs."""
    fs = TensorBoardFS(tmp_path)

    listing = fs.readdir("/")

    for tab in FIXED_TABS:
        assert tab in listing
    assert ".cache" in listing
    assert ".in_memory" in listing


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


def test_filesystem_registers_same_stem_scalar_symlink_format(tmp_path: Path) -> None:
    """Scalar symlink aliases register same-stem export formats."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path)

    fs.symlink("/scalars/loss.json", "/scalars/loss.xlsx")

    assert "loss.xlsx" in fs.readdir("/scalars")


@pytest.mark.parametrize(
    ("target", "source"),
    [
        ("/scalars/loss.json", "/scalars/other.xlsx"),
        ("/scalars/loss.json", "/images/loss.xlsx"),
        ("/scalars/missing.json", "/scalars/missing.xlsx"),
        ("/scalars/loss.json", "/run/scalars/loss.xlsx"),
    ],
)
def test_filesystem_rejects_invalid_scalar_symlinks(
    tmp_path: Path, target: str, source: str
) -> None:
    """Invalid scalar symlink aliases are rejected."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_record(_event(100.0, 1, [_simple_value("loss", 2.0)])))
    fs = TensorBoardFS(tmp_path)

    with pytest.raises(OSError):
        fs.symlink(target, source)


def test_filesystem_ignores_truncated_trailing_record_during_refresh(
    tmp_path: Path,
) -> None:
    """Refresh mode ignores a truncated trailing record."""
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(
        _record(_event(100.0, 1, [_simple_value("loss", 2.0)]))
        + struct.pack("<Q", 100)
        + b"\0\0\0\0"
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
    first_processed_pos = first_cache.processed_pos
    path.write_bytes(
        path.read_bytes() + _record(_event(102.0, 2, [_simple_value("loss", 1.5)]))
    )

    rows = json.loads(fs.read("/scalars/loss.json", 10000, 0))
    second_cache = fs._state.runs[()].files[path]

    assert first_processed_pos < second_cache.processed_pos
    assert [row["step"] for row in rows] == [1, 2]
    assert [row["relative_time"] for row in rows] == [0.0, 2.0]


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
