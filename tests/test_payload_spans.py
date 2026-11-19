"""Exercise protobuf extent indexing and verified extraction."""

import struct
from base64 import b64decode
from dataclasses import replace
from errno import EIO
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.benchmark_scale import _ScaleCorpus
from tboardfs import parse_file
from tboardfs.classify import detect_extension, magic
from tboardfs.crc32c import masked_crc32c
from tboardfs.extract import _extract_graph_blob, extract_binary_blob
from tboardfs.payload_spans import PayloadSpans
from tboardfs.proto import compat_pb2

_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZioAAAAASUVORK5CYII="
)


def _record(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def test_duplicate_values_and_multipart_extents(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    event = compat_pb2.Event(step=1)
    event.summary.value.add(tag="same").image.encoded_image_string = _PNG
    event.summary.value.add(tag="same").image.encoded_image_string = _PNG
    event.summary.value.add(tag="clip").audio.encoded_audio_string = b"wave bytes"
    event.summary.value.add(tag="content").tensor.CopyFrom(
        compat_pb2.TensorProto(dtype=4, tensor_content=bytes(4096))
    )
    value = event.summary.value.add(tag="images")
    value.metadata.plugin_data.plugin_name = "images"
    parts = [b"1", b"1", *([_PNG] * 128)]
    value.tensor.dtype = 7
    value.tensor.tensor_shape.dim.add(size=len(parts))
    value.tensor.string_val.extend(parts)
    payload = event.SerializeToString()
    path.write_bytes(_record(payload))

    calls: list[tuple[int, int, int]] = []
    fields = PayloadSpans.value_fields

    def count_fields(
        self: PayloadSpans, ordinal: int, field: int, nested: int
    ) -> list[tuple[int, int]]:
        calls.append((ordinal, field, nested))
        return fields(self, ordinal, field, nested)

    with patch.object(PayloadSpans, "value_fields", count_fields):
        entries = parse_file(path)["binary_entries"]
    assert calls == [(0, 4, 4), (1, 4, 4), (2, 6, 4), (3, 8, 4), (4, 8, 8)]
    assert [entry.value_ordinal for entry in entries] == [0, 1, 2, 3] + [4] * 128
    assert [entry.part for entry in entries[4:]] == list(range(2, 130))
    assert entries[1].suffix == "_v1"
    assert len({entry.blob.span for entry in entries}) == len(entries)
    with patch("tboardfs.extract.parse_event") as decode:
        for entry, expected in zip(
            entries, [_PNG, _PNG, b"wave bytes", bytes(4096), *parts[2:]], strict=True
        ):
            offset, size = entry.blob.span
            assert entry.blob.size == size == len(expected)
            assert payload[offset : offset + size] == expected
            assert extract_binary_blob(entry) == expected
    decode.assert_not_called()


def test_fallback_preserves_protobuf_oneof_and_merge(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    event = compat_pb2.Event(step=1)
    event.summary.value.add(tag="duplicate").image.encoded_image_string = b"first"
    second = compat_pb2.Event()
    second.summary.value.add(tag="duplicate").image.encoded_image_string = _PNG
    second.summary.value.add(tag="overwritten").image.encoded_image_string = _PNG
    overwritten = second.summary.value[-1].SerializeToString() + b"\x15\x00\x00\x80\x3f"
    del second.summary.value[-1]
    payload = (
        event.SerializeToString()
        + second.SerializeToString()
        + b"\x2a"
        + bytes([2 + len(overwritten), 0x0A, len(overwritten)])
        + overwritten
        + b"\x7a\x80\x20"
        + bytes(4096)
    )
    path.write_bytes(_record(payload))

    entries = parse_file(path)["binary_entries"]
    assert [(e.value_ordinal, e.blob.span, e.blob.size) for e in entries] == [
        (0, None, 5),
        (1, None, len(_PNG)),
    ]
    assert [extract_binary_blob(entry) for entry in entries] == [b"first", _PNG]


@pytest.mark.parametrize("unknown_group", [False, True])
def test_unusual_wire_fields_preserve_blob(tmp_path: Path, unknown_group: bool) -> None:
    path = tmp_path / "events.out.tfevents"
    event = compat_pb2.Event()
    event.summary.value.add(tag="image").image.encoded_image_string = _PNG
    payload = (
        b"\x90\x00\x81\x00" + event.SerializeToString() + b"\x7a\x80\x20" + bytes(4096)
    )
    if unknown_group:
        payload += b"\x7b\x08\x01\x7c"
    path.write_bytes(_record(payload))

    (entry,) = parse_file(path)["binary_entries"]
    assert (entry.blob.span is None) == unknown_group
    assert entry.step == 1
    assert entry.blob.size == len(_PNG)
    assert extract_binary_blob(entry) == _PNG


@pytest.mark.parametrize("field", ["graph_def", "meta_graph_def"])
def test_graph_extent_keeps_original_frame_checksums(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "events.out.tfevents"
    graph = b"\xa2\x06\x80\x20" + bytes(4096)
    event = compat_pb2.Event(step=1, **{field: graph})
    payload = event.SerializeToString()
    path.write_bytes(_record(payload))
    (entry,) = parse_file(path)["graphs"]
    assert entry.blob.span is not None
    offset, size = entry.blob.span
    assert size == entry.blob.size == len(graph)
    assert payload[offset : offset + size] == graph
    assert _extract_graph_blob(entry) == graph
    fallback = replace(entry, blob=replace(entry.blob, span=None))
    assert _extract_graph_blob(fallback) == graph

    with path.open("r+b") as handle:
        handle.seek(entry.frame.payload_offset + 1)
        handle.write(b"\x02")
    with pytest.raises(OSError) as error:
        _extract_graph_blob(entry)
    assert error.value.errno == EIO
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(_record(payload[:1] + b"\x02" + payload[2:]))
    with pytest.raises(OSError) as error:
        _extract_graph_blob(entry)
    assert error.value.errno == EIO


def test_noncanonical_images_cache_exact_filetype_results(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    event = compat_pb2.Event(step=1)
    blob = _ScaleCorpus.small_png(0) + b"trailing"
    event.summary.value.add(tag="image").image.encoded_image_string = blob
    path.write_bytes(_record(event.SerializeToString()) * 20)
    with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
        entries = parse_file(path)["binary_entries"]
    assert len(entries) == 20
    assert all(entry.extension == "png" for entry in entries)
    assert all(e.blob.size == len(blob) and e.blob.span is None for e in entries)
    assert 1 <= detect.call_count <= 2
    assert all(call.args[0] == blob for call in detect.call_args_list)


def test_filetype_cache_distinguishes_a_changed_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    arrow = (
        Path(__file__).parent / "data" / "file-types" / "scientific" / "stations.arrow"
    ).read_bytes()
    event = compat_pb2.Event(step=1)
    event.summary.value.add(tag="good").image.encoded_image_string = arrow
    event.summary.value.add(tag="bad").image.encoded_image_string = arrow[:-1] + b"\0"
    path.write_bytes(_record(event.SerializeToString()))

    good, bad = parse_file(path)["binary_entries"]
    assert good.extension == "arrow"
    assert bad.extension != "arrow"


def test_distinct_small_pngs_skip_libmagic(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    frames = []
    for step in range(256):
        event = compat_pb2.Event(step=step)
        event.summary.value.add(
            tag="image"
        ).image.encoded_image_string = _ScaleCorpus.small_png(step)
        frames.append(_record(event.SerializeToString()))
    path.write_bytes(b"".join(frames))
    with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
        entries = parse_file(path)["binary_entries"]
    assert len(entries) == 256
    assert all(entry.extension == "png" for entry in entries)
    detect.assert_not_called()


def test_larger_multi_chunk_png_skips_libmagic() -> None:
    png = _ScaleCorpus._png(128, 96, (b"\0" + bytes(range(128)) * 3) * 96)
    data = png[41:-16]
    midpoint = len(data) // 2
    png = (
        png[:33]
        + _ScaleCorpus._chunk(b"IDAT", data[:midpoint])
        + _ScaleCorpus._chunk(b"IDAT", data[midpoint:])
        + png[-12:]
    )
    with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
        assert detect_extension(png) == "png"
    detect.assert_not_called()


def test_png_trailer_must_end_the_first_image() -> None:
    png = _ScaleCorpus.small_png(1)
    for blob in (png[:-1], png + b"trailing" + png[-12:]):
        with patch.object(magic, "from_buffer", wraps=magic.from_buffer) as detect:
            assert detect_extension(blob) == "png"
        assert detect.call_args is not None and detect.call_args.args[0] == blob


def test_png_header_does_not_override_hdf_user_block() -> None:
    hdf = (
        Path(__file__).parent / "data" / "file-types" / "scientific" / "profile.h5"
    ).read_bytes()
    png = _ScaleCorpus.small_png(1)
    polyglot = png[:33] + _ScaleCorpus._chunk(b"vpAg", bytes(512 - 41) + hdf) + png[33:]
    assert detect_extension(polyglot) == "h5"
