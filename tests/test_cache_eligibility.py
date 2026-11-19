"""Only clean, whole snapshots are eligible for a persistent index."""

import os
from pathlib import Path
from typing import BinaryIO, Never

import pytest

from tboardfs.crc32c import masked_crc32c
from tboardfs.index_segment import Segment
from tboardfs.index_transport import packed_work
from tboardfs.indexer import _EventIndexer, _ParseState
from tboardfs.model import EventFileCache, Record
from tboardfs.proto import compat_pb2

from test_incremental_parse import _frame, _loss
from test_index_transport import _job, _tensor_frame
from test_payload_spans import _record
from test_tf2_summaries import _images


_SECOND_FRAME = _loss(2, 3.0)
_BAD_FOOTER = _SECOND_FRAME[:-1] + bytes([_SECOND_FRAME[-1] ^ 1])


@pytest.mark.parametrize(
    "tail",
    [b"", b"\xff", _SECOND_FRAME[:-1], _BAD_FOOTER, _record(b"\x80")],
    ids=["clean", "short-header", "short-footer", "bad-footer", "malformed-event"],
)
def test_whole_snapshot_eligibility(tmp_path: Path, tail: bytes) -> None:
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_loss(1, 2.0) + tail)
    cache = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    assert cache.progress.provenance.cacheable is (not tail)
    assert cache.scalars["loss"]["value"].tolist() == [2.0]


def test_empty_snapshot_is_ineligible(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    path.touch()
    assert not _EventIndexer.parse_event_file(path).progress.provenance.cacheable


def test_direct_warning_is_ineligible_without_tainting_next_parse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_frame(compat_pb2.Event(meta_graph_def=b"\x80")))
    malformed = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    assert not malformed.progress.provenance.cacheable
    path.write_bytes(_loss(1, 2.0))
    clean = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    assert clean.progress.provenance.cacheable


def test_skipped_conversion_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_frame(_images(1, b"image")) + _loss(2, 3.0))

    def broken_summary(*_args: object) -> Never:
        raise OverflowError("cannot convert summary")

    monkeypatch.setattr(_EventIndexer, "add_summary", broken_summary)
    cache = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    assert not cache.progress.provenance.cacheable
    assert cache.scalars["loss"]["value"].tolist() == [3.0]


def test_changed_snapshot_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_loss(1, 2.0))
    status = path.stat()
    original = _EventIndexer.index_record

    def change_timestamp(
        cache: EventFileCache, record: Record, state: _ParseState, *, strict: bool
    ) -> None:
        original(cache, record, state, strict=strict)
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(_EventIndexer, "index_record", change_timestamp)
    cache = _EventIndexer.parse_event_file(path, retain_tensor_extents=True)
    assert not cache.progress.provenance.cacheable
    assert cache.scalars["loss"]["value"].tolist() == [2.0]


def test_full_retry_after_unstable_prefix_is_not_cacheable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents"
    path.write_bytes(_loss(1, 2.0) + _BAD_FOOTER)

    def replace_prefix(_handle: BinaryIO, _cache: EventFileCache) -> bool:
        path.write_bytes(_SECOND_FRAME)
        return False

    monkeypatch.setattr(_EventIndexer, "valid_anchor", replace_prefix)
    with path.open("rb", buffering=0) as handle:
        cache = _EventIndexer.parse_open(
            path,
            handle,
            os.fstat(handle.fileno()),
            None,
            None,
            ignore_truncated=True,
            strict_framing=False,
            state=_ParseState(retain_tensor_extents=True),
        )
    assert cache.progress.processed_pos == len(_SECOND_FRAME)
    assert cache.scalars["loss"]["value"].tolist() == [3.0]
    assert not cache.progress.provenance.cacheable


def test_tensor_extent_is_eager_in_serial_worker_and_corrective_parse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.out.tfevents"
    data = b"x" * 80_000
    frame = _tensor_frame(data)
    path.write_bytes(frame)
    expected = (frame.index(data), len(data), masked_crc32c(data))
    normal = _EventIndexer.parse_event_file(path)
    serial = _EventIndexer.parse_event_file(path, retain_tensor_extents=True)
    corrective = Segment.work(_job(path)).cache
    worker = packed_work(_job(path)).unpack().cache

    assert normal.json_entries[0].tensor_extent is None
    for cache in (serial, corrective, worker):
        assert cache.progress.provenance.cacheable
        assert cache.json_entries[0].tensor_extent == expected
        assert cache.json_entries[0].payload["tensor"]["tensor_content"] == data
        assert cache.json_entries == normal.json_entries
