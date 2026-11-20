"""Verify source-backed tensor transport and its snapshot checks."""

from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from random import Random

import pytest
from test_incremental_parse import _frame
from test_payload_spans import _record

from tboardfs import index_transport
from tboardfs.crc32c import masked_crc32c
from tboardfs.file_status import FileIdentity
from tboardfs.index_segment import Job, Segment, _SegmentError
from tboardfs.index_transport import packed_work
from tboardfs.proto import compat_pb2


def _job(path: Path) -> Job:
    status = path.stat()
    return (
        path,
        0,
        status.st_size,
        FileIdentity.fingerprint(status),
        None,
        None,
        {},
        None,
    )


def _tensor(data: bytes) -> compat_pb2.TensorProto:
    tensor = compat_pb2.TensorProto(dtype=4, tensor_content=data)
    tensor.tensor_shape.dim.add(size=len(data))
    tensor.int_val.append(0)
    return tensor


def _tensor_frame(*buffers: bytes) -> bytes:
    event = compat_pb2.Event(step=7, wall_time=2.0)
    event.summary.value.add(tag="scalar", simple_value=1.0)
    for ordinal, data in enumerate(buffers):
        event.summary.value.add(tag=f"tensor/{ordinal}").tensor.CopyFrom(_tensor(data))
    return _frame(event)


def _message(field: int, payload: bytes) -> bytes:
    length = len(payload)
    prefix = bytearray([field * 8 + 2])
    while length >= 128:
        prefix.append((length & 127) | 128)
        length >>= 7
    prefix.append(length)
    return bytes(prefix) + payload


def test_packed_transport_omits_incompressible_tensor_buffers(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    rng = Random(13)
    first, second = rng.randbytes(90_000), rng.randbytes(95_000)
    path.write_bytes(_tensor_frame(first, b"small", second))
    job = _job(path)
    stock = Segment.work(job)
    packed = packed_work(job)
    assert len(packed.segment.deferred_tensor_content) == 2
    assert (
        packed.segment.cache.json_entries[1].payload["tensor"]["tensor_content"]
        == b"small"
    )
    wire = ForkingPickler.dumps(packed)
    assert len(wire) < len(first)
    restored = ForkingPickler.loads(wire).unpack()
    assert restored.cache.json_entries == stock.cache.json_entries
    for actual, expected in zip(
        restored.cache.json_entries, stock.cache.json_entries, strict=True
    ):
        assert list(actual.payload["tensor"]) == list(expected.payload["tensor"])
    assert not restored.deferred_tensor_content


def test_repeated_tensor_content_stays_inline(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    data = Random(17).randbytes(80_000)
    tensor = _message(4, b"earlier") + _tensor(data).SerializeToString()
    value = _message(1, b"repeated") + _message(8, tensor)
    path.write_bytes(_record(_message(5, _message(1, value))))
    packed = packed_work(_job(path))
    assert not packed.segment.deferred_tensor_content
    actual = packed.unpack().cache.json_entries[0].payload["tensor"]
    assert actual["tensor_content"] == data


def test_restoration_checks_indexed_tensor_checksum(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    first, second = bytes(80_000), b"x" * 80_000
    path.write_bytes(_tensor_frame(first, second))
    packed = packed_work(_job(path))
    descriptors = packed.segment.deferred_tensor_content
    position, _, size, checksum = descriptors[0]
    _, wrong_offset, _, _ = descriptors[1]
    descriptors[0] = position, wrong_offset, size, checksum
    with pytest.raises(_SegmentError, match="indexed tensor changed"):
        packed.unpack()


def test_restore_rejects_replacement_while_reading_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents"
    frame = _tensor_frame(bytes(80_000))
    path.write_bytes(frame)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(frame)
    packed = packed_work(_job(path))

    def replace_source(data: bytes) -> int:
        replacement.replace(path)
        return masked_crc32c(data)

    monkeypatch.setattr(index_transport, "masked_crc32c", replace_source)
    with pytest.raises(_SegmentError, match="unchanged regular-file snapshot"):
        packed.unpack()
