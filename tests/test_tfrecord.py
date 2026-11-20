"""Check TFRecord header reuse, frame validation, and iterator positions."""

import struct
from io import BytesIO

import pytest

from tboardfs import tfrecord
from tboardfs.crc32c import masked_crc32c
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
