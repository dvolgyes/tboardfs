"""Verify compact worker rows against source payloads after their segment is released."""

import wave
from dataclasses import replace
from io import BytesIO
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from weakref import ref

import pytest
from test_graph_summaries import _graph_event, _run_metadata
from test_incremental_parse import _frame
from test_index_transport import _job
from test_payload_spans import _PNG
from test_restart_purge import _start
from test_tf2_summaries import _images

from scripts.scale_corpus import _ScaleCorpus
from tboardfs.extract import extract_binary_blob
from tboardfs.index_binary import BinarySection
from tboardfs.index_codec import IndexCodec
from tboardfs.index_segment import Segment, _SegmentError
from tboardfs.index_transport import _PackedResult, _WorkerRows, packed_work
from tboardfs.model import EventFileCache
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.packed_binaries import PackedBinaries
from tboardfs.proto import compat_pb2
from tboardfs.range_proof import BLOCK_SIZE, range_proof
from tboardfs.verified_span import _FrameValidations, _VerifiedSpan


def _wav(sample: int, size: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes([sample]) * size)
    return buffer.getvalue()


def _write_media_frames(path: Path) -> list[bytes]:
    """Write multi-block WAVs and one PNG small enough for whole-frame reads."""
    media = [_wav(1, 5000), _wav(2, 9000), _PNG, _wav(4, 13_000)]
    frames: list[bytes] = []
    for step, data in enumerate(media, 1):
        event = compat_pb2.Event(step=step, wall_time=10.0 + step)
        value = event.summary.value.add(tag=f"media/{step}")
        if step == 3:
            value.image.encoded_image_string = data
        else:
            value.audio.encoded_audio_string = data
        frames.append(_frame(event))
    path.write_bytes(b"".join(frames))
    return media


def _assert_verified_span(path: Path, entry: BinaryEntry, data: bytes) -> None:
    blob = entry.blob
    span = blob.span
    assert span is not None
    assert blob.range_checksums == range_proof(data)
    with path.open("rb") as handle:
        verified = _VerifiedSpan(
            handle, entry.frame, span, blob.range_checksums, _FrameValidations()
        )
        begin = BLOCK_SIZE - 11
        assert verified.read(35, begin) == data[begin : begin + 35]


def test_restored_binary_evidence_outlives_segment(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    media = _write_media_frames(path)

    segment = Segment.work(_job(path))
    expected = segment.cache.binaries
    expected[1] = replace(expected[1], step=1 << 70, suffix="_exception")
    expected[3] = replace(expected[3], step=4.5, suffix="_fractional")
    packed = _PackedResult.pack(segment)
    assert list(packed.exceptions) == [1]
    restored = ForkingPickler.loads(ForkingPickler.dumps(packed)).unpack()
    cache_ref = ref(restored.cache)
    entries = restored.cache.binaries
    del restored
    assert cache_ref() is None

    assert [entry.size for entry in entries] == [len(data) for data in media]
    assert [
        (entry.tag, entry.step, entry.wall_time, entry.extension, entry.suffix)
        for entry in entries
    ] == [
        ("media/1", 1, 11.0, "wav", ""),
        ("media/2", 1 << 70, 12.0, "wav", "_exception"),
        ("media/3", 3, 13.0, "png", ""),
        ("media/4", 4.5, 14.0, "wav", "_fractional"),
    ]
    for entry, original, data in zip(entries, expected, media, strict=True):
        assert entry.blob == original.blob
        assert extract_binary_blob(entry) == data
    assert entries[2].blob.span is None
    assert entries[2].blob.range_checksums == b""
    for index in (0, 1, 3):
        _assert_verified_span(path, entries[index], media[index])


def _restore_worker(result: _PackedResult) -> PackedBinaries:
    rows = ForkingPickler.loads(ForkingPickler.dumps(result)).unpack().cache.binaries
    assert isinstance(rows, PackedBinaries)
    return rows


def test_packed_legacy_images_preserve_mixed_range(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    green, yellow, blue = (
        _ScaleCorpus.small_png(rgb) for rgb in (0x00FF00, 0xFFFF00, 0x0000FF)
    )
    first = _ScaleCorpus.event_frame(7, pixel=0xFF0000)
    gradient = _ScaleCorpus._png(128, 16, (b"\0" + bytes(range(128)) * 3) * 16)
    mixed = compat_pb2.Event(step=8, wall_time=108.0)
    mixed.summary.value.add(tag="same").image.encoded_image_string = green
    mixed.summary.value.add(tag="same").image.encoded_image_string = yellow
    large = compat_pb2.Event(step=9, wall_time=109.0)
    large.summary.value.add(tag="same").image.encoded_image_string = gradient
    tail = compat_pb2.Event(step=6, wall_time=12.0)
    tail.summary.value.add(tag="same").image.encoded_image_string = blue
    duplicate, spanned, restart, last = (
        _frame(mixed),
        _frame(large),
        _start(6),
        _frame(tail),
    )
    path.write_bytes(first + duplicate + spanned + restart + last)

    job = _job(path)
    reference = Segment.work(job).cache
    result = packed_work(job)
    packed = ForkingPickler.loads(ForkingPickler.dumps(result)).unpack().cache
    assert BinarySection.encode(packed) == BinarySection.encode(reference)
    assert packed.restarts == reference.restarts
    (mark,) = packed.restarts
    assert (mark.kind, mark.value, mark.counts) == (
        "start",
        6,
        {("scalars", "loss"): 1, ("binaries", ""): 4, ("json_entries", ""): 0},
    )

    entries = list(packed.binaries)
    assert entries == reference.binaries
    assert [
        (entry.kind, entry.tag, entry.extension, entry.value_ordinal, entry.suffix)
        for entry in entries
    ] == [
        ("images", "series", "png", 1, ""),
        ("images", "same", "png", 0, ""),
        ("images", "same", "png", 1, "_v1"),
        ("images", "same", "png", 0, ""),
        ("images", "same", "png", 0, ""),
    ]
    span_offset = len(first) + len(duplicate)
    tail_offset = span_offset + len(spanned) + len(restart)
    for entry, data, offset, frame in zip(
        entries,
        (_ScaleCorpus.small_png(0xFF0000), green, yellow, gradient, blue),
        (0, len(first), len(first), span_offset, tail_offset),
        (first, duplicate, duplicate, spanned, last),
        strict=True,
    ):
        assert entry.size == len(data)
        assert (entry.frame.offset, entry.frame.size, entry.frame.checksum) == (
            offset,
            len(frame),
            int.from_bytes(frame[-4:], "little"),
        )
        assert extract_binary_blob(entry) == data
    for index in (0, 1, 2, 4):
        assert entries[index].blob.span is None
        assert entries[index].blob.range_checksums == b""
    _assert_verified_span(path, entries[3], gradient)


def test_packed_legacy_image_range_rejects_bad_tail_checksum(tmp_path: Path) -> None:
    path = tmp_path / "events.out.tfevents"
    first, last = (_ScaleCorpus.event_frame(step) for step in (1, 2))
    path.write_bytes(first + last[:-1] + bytes([last[-1] ^ 1]))
    message = rf"payload checksum mismatch at offset {len(first)}"
    with pytest.raises(_SegmentError, match=message):
        packed_work(_job(path))


@pytest.fixture
def mixed_worker_index(
    tmp_path: Path,
) -> tuple[EventFileCache, EventFileCache, list[bytes]]:
    """Make two worker batches with different labels and a shared eager label."""
    path = tmp_path / "events.out.tfevents"
    media = _write_media_frames(path)
    graph = _run_metadata(2)
    second_png = _ScaleCorpus.small_png(0x00FF00)
    with path.open("ab") as handle:
        handle.write(_graph_event(graph, plugin="graph_run_metadata", step=5))
        boundary = handle.tell()
        images = _images(6, _PNG, second_png)
        images.summary.value[0].tag = "é/文"
        handle.write(_frame(images))

    job = _job(path)
    left = Segment.work((job[0], 0, boundary, *job[3:]))
    assert isinstance(left.cache.binaries, list)
    left_rows = left.cache.binaries
    left_rows[0] = replace(left_rows[0], step=-(2**63), wall_time=-0.0)
    left_rows[1] = replace(left_rows[1], step=2**64 - 1)
    left_rows[3] = replace(left_rows[3], step=4.5, suffix="_β")
    graph_entry = left_rows[4]
    assert isinstance(graph_entry, GraphSummaryEntry)
    assert graph_entry.converts and graph_entry.profiles
    result = _PackedResult.pack(left)
    assert set(result.exceptions) == {1, 4}
    result.exceptions = dict(reversed(result.exceptions.items()))
    right_job = (job[0], boundary, *job[2:])
    right_reference = Segment.work(right_job).cache.binaries
    eager_chunk = [replace(left_rows[0], step=99), right_reference[0]]
    packed_rows = _restore_worker(result)
    packed_rows.extend(eager_chunk)
    packed_rows.extend(_restore_worker(packed_work(right_job)))

    reference = Segment.work(job).cache
    reference.binaries = [*left_rows, *eager_chunk, *right_reference]
    packed = replace(reference, binaries=packed_rows)
    payloads = [*media, graph, media[0], _PNG, _PNG, second_png]
    return packed, reference, payloads


def test_packed_worker_index_matches_eager_encoding(
    mixed_worker_index: tuple[EventFileCache, EventFileCache, list[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed, reference, payloads = mixed_worker_index
    expected_bytes = IndexCodec.encode(reference)
    entry_at = _WorkerRows.entry_at

    def guard_ordinary_rows(rows: _WorkerRows, position: int) -> BinaryEntry:
        assert position in rows.exceptions, (
            "persistence materialized an ordinary packed worker row"
        )
        return entry_at(rows, position)

    with monkeypatch.context() as patch:
        patch.setattr(_WorkerRows, "entry_at", guard_ordinary_rows)
        encoded = IndexCodec.encode(packed)
    assert encoded == expected_bytes

    path = packed.source_path
    with path.open("rb") as handle:
        restored = IndexCodec.decode(encoded, path, path.stat(), handle, {})
    assert restored is not None
    assert list(restored.binaries) == reference.binaries
    for entry, payload in zip(restored.binaries, payloads, strict=True):
        assert extract_binary_blob(entry) == payload
        assert entry.frame.identity is restored.progress.identity
        assert entry.frame.fingerprint is restored.progress.checked_status
        if entry.blob.span is not None:
            assert entry.blob.range_checksums == range_proof(payload)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("frame", "indexed frame"),
        ("snapshot", "source snapshot"),
        ("proof", "range proof"),
        ("span_size", "span"),
        ("frame_offset_overflow", "frame"),
    ],
)
def test_packed_worker_index_rejects_invalid_evidence(
    tmp_path: Path, damage: str, message: str
) -> None:
    path = tmp_path / "events.out.tfevents"
    _write_media_frames(path)
    segment = Segment.work(_job(path))
    cache = segment.cache
    assert isinstance(cache.binaries, list)
    if damage == "proof":
        entry = cache.binaries[0]
        proof = entry.blob.range_checksums
        assert len(proof) > 4
        cache.binaries[0] = replace(
            entry, blob=replace(entry.blob, range_checksums=proof[:-4])
        )
    elif damage == "span_size":
        entry = cache.binaries[0]
        assert entry.blob.span is not None
        cache.binaries[0] = replace(
            entry, blob=replace(entry.blob, span=(entry.blob.span[0], entry.size - 1))
        )
    elif damage == "frame_offset_overflow":
        entry = cache.binaries[0]
        cache.binaries[0] = replace(
            entry,
            blob=replace(entry.blob, frame=replace(entry.frame, offset=2**64 - 1)),
        )
    cache = _PackedResult.pack(segment).unpack().cache
    if damage == "frame":
        cache.progress.frames.checksums[0] ^= 1
    elif damage == "snapshot":
        cache.progress.checked_status = ()
    with pytest.raises(ValueError, match=message):
        BinarySection.encode(cache)


def test_packed_worker_index_enforces_exact_budget(
    mixed_worker_index: tuple[EventFileCache, EventFileCache, list[bytes]],
) -> None:
    packed, reference, _ = mixed_worker_index
    expected = BinarySection.encode(reference)
    assert BinarySection.encode(packed, limit=len(expected)) == expected
    with pytest.raises(ValueError, match="budget"):
        BinarySection.encode(packed, limit=len(expected) - 1)
