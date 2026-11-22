"""Check TFRecord header reuse, frame validation, and iterator positions."""

import struct
from collections.abc import Buffer
from io import BufferedReader, BytesIO, FileIO
from pathlib import Path
from typing import IO

import pytest

from tboardfs import tfrecord
from tboardfs.crc32c import masked_crc32c
from tboardfs.file_status import FileIdentity
from tboardfs.indexed_frame import IndexedFrame
from tboardfs.model import TFRecordError
from tboardfs.proto import compat_pb2


def _frame(payload: bytes) -> bytes:
    """Encode one complete TFRecord frame."""
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", masked_crc32c(payload))
    )


def test_equal_length_events_keep_independent_payloads() -> None:
    """Different events of the same length are both valid in sequence."""
    a = compat_pb2.Event(step=1).SerializeToString()
    b = compat_pb2.Event(step=2).SerializeToString()
    first, second = _frame(a), _frame(b)
    assert first[:12] == second[:12]
    assert first[-4:] != second[-4:]
    records = list(tfrecord._iter_tfrecords(BytesIO(first + second)))

    assert [record.payload for record in records] == [a, b]


def test_equal_length_with_changed_header_crc_is_corrupt() -> None:
    """Matching length bytes cannot validate the next frame's damaged header."""
    frame = _frame(b"data")
    bad = frame[:8] + bytes([frame[8] ^ 1]) + frame[9:]
    records = tfrecord._iter_tfrecords(BytesIO(frame + bad))
    assert next(records).payload == b"data"

    with pytest.raises(TFRecordError, match="length checksum mismatch") as error:
        next(records)
    assert error.value.corrupt


def test_decodable_event_with_stale_payload_crc_is_corrupt() -> None:
    """A previously validated header cannot validate a changed event payload."""
    first = _frame(compat_pb2.Event(step=1).SerializeToString())
    changed = compat_pb2.Event(step=2).SerializeToString()
    assert compat_pb2.Event.FromString(changed).step == 2
    assert _frame(changed)[:12] == first[:12]
    records = tfrecord._iter_tfrecords(
        BytesIO(first + first[:12] + changed + first[-4:])
    )
    assert next(records).payload == first[12:-4]

    with pytest.raises(TFRecordError, match="payload checksum mismatch") as error:
        next(records)
    assert error.value.corrupt


def test_equal_header_frame_straddling_bound_does_not_read_past_it() -> None:
    """A cached header does not validate the next frame against an earlier bound."""
    frame = _frame(b"data")
    end = 2 * len(frame) - 1

    class BoundedStream(BytesIO):
        """Reject even an attempted read across the parser's bound."""

        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= end - self.tell()
            return super().read(size)

    records = tfrecord._iter_tfrecords(BoundedStream(frame * 2), end=end)
    assert next(records).payload == b"data"

    with pytest.raises(TFRecordError, match="truncated TFRecord payload") as error:
        next(records)
    assert not error.value.corrupt


def test_nonzero_start_keeps_absolute_offsets_for_empty_frame() -> None:
    """An empty payload consumes its frame and preserves absolute offsets."""
    prefix = b"prefix"
    empty, populated = _frame(b""), _frame(b"data")
    stream = BytesIO(prefix + empty + populated)
    stream.seek(len(prefix))

    first, second = tfrecord._iter_tfrecords(stream)

    assert first.record_offset == len(prefix)
    assert first.payload_offset == len(prefix) + 12
    assert first.payload == b""
    assert first.payload_size == 0
    assert first.total_size == len(empty)
    assert second.record_offset == len(prefix) + len(empty)
    assert second.payload_offset == second.record_offset + 12
    assert second.payload == b"data"


@pytest.mark.parametrize("size", [0, (64 << 10) - 16, (64 << 10) - 15])
def test_indexed_read_preserves_record_and_source_offsets(
    tmp_path: Path, size: int
) -> None:
    source = tmp_path / "events"
    data = _frame(b"first") + _frame(b"x" * size) + _frame(b"last")
    source.write_bytes(data)
    _, middle, _ = tfrecord.iter_tfrecords(BytesIO(data))
    ref = middle.reference(FileIdentity.of(source.stat()))

    assert IndexedFrame.read(source, ref) == middle


@pytest.mark.parametrize("size", [8 << 10, 65 << 10], ids=["small", "large"])
def test_indexed_read_accepts_positive_short_transfers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    """Reading an intact indexed frame must tolerate short raw transfers."""
    source = tmp_path / "events"
    data = _frame(b"first") + _frame(b"x" * size) + _frame(b"last")
    source.write_bytes(data)
    _, expected, _ = tfrecord.iter_tfrecords(BytesIO(data))
    ref = expected.reference(FileIdentity.of(source.stat()))

    class ShortFileIO(FileIO):
        def read(self, size: int = -1, /) -> bytes | None:
            return super().read(min(size, 4096) if size >= 0 else size)

        def readinto(self, buffer: Buffer, /) -> int | None:
            with memoryview(buffer).cast("B") as view:
                return super().readinto(view[:4096])

    original_open = Path.open

    def open_source(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[str] | IO[bytes] | FileIO | BufferedReader:
        if path != source or mode != "rb":
            return original_open(path, mode, buffering, encoding, errors, newline)
        raw = ShortFileIO(path, mode)
        if buffering == 0:
            return raw
        if buffering < 0:
            return BufferedReader(raw)
        return BufferedReader(raw, buffer_size=buffering)

    monkeypatch.setattr(Path, "open", open_source)
    assert IndexedFrame.read(source, ref) == expected


def test_public_iterator_honors_seek_between_yields() -> None:
    """A caller may skip a frame while the public iterator is suspended."""
    first, skipped, last = (_frame(payload) for payload in (b"a", b"b", b"c"))
    stream = BytesIO(first + skipped + last)
    records = tfrecord.iter_tfrecords(stream)
    assert next(records).payload == b"a"
    offset = len(first) + len(skipped)
    stream.seek(offset)

    (record,) = records

    assert record.record_offset == offset
    assert record.payload == b"c"
