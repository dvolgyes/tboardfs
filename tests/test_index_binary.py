"""Exercise cache-v2 binary rows and their binding to the source index."""

import math
import os
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tboardfs.crc32c import masked_crc32c
from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.index_binary import BinarySection
from tboardfs.index_codec import IndexCodec
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.range_proof import range_proof


def _record(payload: bytes, checksum: int) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", masked_crc32c(length))
        + payload
        + struct.pack("<I", checksum)
    )


@pytest.fixture
def binary_cache(tmp_path: Path) -> EventFileCache:
    """Build rows spanning two source frames and more than one proof block."""
    cache = EventFileCache(tmp_path / "old")
    progress = cache.progress
    payload = b"a" * 4200
    first_payload = bytes(104)
    second_payload = bytes(60) + payload + bytes(4604)
    first_crc = masked_crc32c(first_payload)
    second_crc = masked_crc32c(second_payload)
    source = _record(first_payload, first_crc) + _record(second_payload, second_crc)
    cache.source_path.write_bytes(source)
    status = cache.source_path.stat()
    progress.processed_pos = progress.last_checked_size = len(source)
    progress.identity = FileIdentity.of(status)
    progress.checked_status = FileIdentity.fingerprint(status)
    progress.frames.append(120, first_crc)
    progress.frames.append(len(source), second_crc)
    first = FrameRef(progress.identity, 0, 120, first_crc, progress.checked_status)
    second = FrameRef(
        progress.identity, 120, len(source) - 120, second_crc, progress.checked_status
    )
    proven = IndexedBlob(second, len(payload), (60, len(payload)), range_proof(payload))
    fallback = IndexedBlob(first, 53)
    cache.binaries = [
        BinaryEntry(
            fallback, cache.source_path, "á/文", "images", -(2**63), 2.5, "png", 3
        ),
        BinaryEntry(
            proven,
            cache.source_path,
            "tag",
            "audio",
            np.uint64(2**64 - 1),
            -0.0,
            "wav",
            0,
            0,
            "_01",
        ),
        BinaryEntry(
            proven,
            cache.source_path,
            "tag",
            "audio",
            np.int64(2**63 - 1),
            7.25,
            "wav",
            2,
            15,
            "_01",
        ),
        GraphSummaryEntry(
            fallback,
            cache.source_path,
            "model",
            "graphs",
            float("nan"),
            8.5,
            "pb",
            2,
            0,
            ".run_metadata",
            True,
            False,
        ),
        GraphSummaryEntry(
            fallback,
            cache.source_path,
            "model",
            "graphs",
            np.float32(1.25),
            8.5,
            "pb",
            2,
            0,
            ".run_metadata",
            False,
            True,
        ),
    ]
    return cache


@pytest.mark.parametrize("view", [False, True])
def test_binary_rows_round_trip_against_live_source(
    binary_cache: EventFileCache, view: bool
) -> None:
    original = binary_cache
    live = EventFileCache(
        original.source_path.with_name("live"),
        replace(
            original.progress,
            identity=FileIdentity(9, 8, 7),
            checked_status=(9, 8, 7, 9000, 6, 5),
        ),
    )
    section = BinarySection.encode(original)
    rows = BinarySection.decode(memoryview(section) if view else section, live)
    for before, after in zip(original.binaries, rows, strict=True):
        assert type(after) is type(before)
        assert after.source_path is live.source_path
        assert after.frame.identity is live.progress.identity
        assert after.frame.fingerprint is live.progress.checked_status
        assert isinstance(after.blob.range_checksums, bytes)
        if math.isnan(before.step):
            assert math.isnan(after.step)
            after = replace(after, step=before.step)
        elif isinstance(before.step, np.integer):
            assert type(after.step) is int
            assert after.step == int(before.step)
        assert after == replace(
            before,
            source_path=live.source_path,
            blob=replace(
                before.blob,
                frame=replace(
                    before.frame,
                    identity=live.progress.identity,
                    fingerprint=live.progress.checked_status,
                ),
            ),
        )
    assert math.copysign(1.0, rows[1].wall_time) == -1.0


def test_binary_rows_reject_wrong_live_frame_layout(
    binary_cache: EventFileCache,
) -> None:
    section = BinarySection.encode(binary_cache)
    binary_cache.progress.frames.ends[0] += 1
    with pytest.raises(ValueError, match="frame"):
        BinarySection.decode(section, binary_cache)


@pytest.mark.parametrize("change", ["row_count", "truncated_proof", "extra_proof"])
def test_binary_rows_reject_wrong_section_length(
    binary_cache: EventFileCache, change: str
) -> None:
    section = BinarySection.encode(binary_cache)
    if change == "row_count":
        section = struct.pack("<I", 2**32 - 1) + section[4:]
    elif change == "truncated_proof":
        section = section[:-1]
    else:
        section += b"\x00"
    with pytest.raises(ValueError):
        BinarySection.decode(section, binary_cache)


@pytest.mark.parametrize("damage", ["late_frame_checksum", "short_proof", "long_proof"])
def test_index_decode_rejects_damaged_binary_rows_before_return(
    binary_cache: EventFileCache, damage: str
) -> None:
    section = BinarySection.encode(binary_cache)
    body = IndexCodec.encode(binary_cache)
    binary_block = struct.pack("<Q", len(section)) + section
    assert body.endswith(binary_block)
    if damage == "late_frame_checksum":
        damaged = bytearray(section)
        row_count, _, _ = struct.unpack_from("<III", section)
        last_row = struct.calcsize("<III") + (row_count - 1) * struct.calcsize(
            "<BqdIIIIIIII"
        )
        checksum = last_row + struct.calcsize("<BqdIII")
        damaged[checksum] ^= 1
        section = bytes(damaged)
    elif damage == "short_proof":
        section = section[:-1]
    else:
        section += b"\x00"
    block = struct.pack("<Q", len(section)) + section
    damaged_body = body[: -len(binary_block)] + block
    with binary_cache.source_path.open("rb") as source:
        status = os.fstat(source.fileno())
        assert IndexCodec.decode(body, binary_cache.source_path, status, source, {})
        with pytest.raises(ValueError):
            IndexCodec.decode(
                damaged_body, binary_cache.source_path, status, source, {}
            )


def _mixed_proof_rows(
    binary_cache: EventFileCache,
) -> tuple[BinaryEntry, BinaryEntry, BinaryEntry, BinaryEntry]:
    fallback, multi, *_ = binary_cache.binaries
    empty = replace(fallback, blob=IndexedBlob(fallback.frame, 0, (0, 0)))
    single = replace(
        fallback,
        blob=IndexedBlob(fallback.frame, 1, (0, 1), range_proof(b"\x00")),
    )
    return multi, empty, fallback, single


@pytest.mark.parametrize("writable", [False, True])
def test_binary_rows_keep_distinct_proof_slices_and_frames(
    binary_cache: EventFileCache, writable: bool
) -> None:
    binary_cache.binaries = list(_mixed_proof_rows(binary_cache))
    section = BinarySection.encode(binary_cache)
    if writable:
        buffer = bytearray(section)
        with memoryview(buffer) as view:
            rows = BinarySection.decode(view, binary_cache)
        buffer[:] = bytes(len(buffer))
    else:
        rows = BinarySection.decode(section, binary_cache)
    assert rows == binary_cache.binaries


def test_binary_rows_preserve_proofs_and_snapshot_across_batches(
    binary_cache: EventFileCache,
) -> None:
    multi, empty, _, single = _mixed_proof_rows(binary_cache)
    cycle = [multi, single, empty]
    count = 65_540
    binary_cache.binaries = cycle * (count // 3) + cycle[: count % 3]
    rows = BinarySection.decode(BinarySection.encode(binary_cache), binary_cache)

    frames = binary_cache.progress.frames
    for i in range(len(frames.ends)):
        frames.ends[i] += 1
        frames.checksums[i] ^= 1

    assert len(rows) == count
    for position in (0, 1, 32_768, 65_535, 65_536, 65_537, count - 1):
        expected = cycle[position % len(cycle)]
        restored = rows[position]
        assert restored == expected
        assert restored.frame.identity is expected.frame.identity
        assert restored.frame.fingerprint is expected.frame.fingerprint


def test_binary_rows_reject_foreign_source(binary_cache: EventFileCache) -> None:
    entry = binary_cache.binaries[0]
    binary_cache.binaries[0] = replace(
        entry, source_path=entry.source_path.with_name("foreign")
    )
    with pytest.raises(ValueError, match="source"):
        BinarySection.encode(binary_cache)


def test_binary_rows_reject_bad_span_proof(binary_cache: EventFileCache) -> None:
    entry = binary_cache.binaries[1]
    binary_cache.binaries[1] = replace(
        entry, blob=replace(entry.blob, range_checksums=b"x")
    )
    with pytest.raises(ValueError, match="proof"):
        BinarySection.encode(binary_cache)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (0, 128, "flags"),
        (0, 8, "graph flags"),
        (3, 100, "label index"),
        (4, 100, "frame index"),
        (6, 0, "indexed frame"),
        (7, 105, "blob size"),
        (8, 1, "span"),
        (9, 104, "ordinal"),
        (10, 1, "part"),
    ],
)
def test_binary_rows_reject_invalid_fields(
    binary_cache: EventFileCache, field: int, value: int, message: str
) -> None:
    section = bytearray(BinarySection.encode(binary_cache))
    offset = struct.calcsize("<III")
    row = struct.Struct("<BqdIIIIIIII")
    fields = list(row.unpack_from(section, offset))
    fields[field] = value
    row.pack_into(section, offset, *fields)
    with pytest.raises(ValueError, match=message):
        BinarySection.decode(bytes(section), binary_cache)


def test_binary_rows_reject_unsafe_filename_label(binary_cache: EventFileCache) -> None:
    section = BinarySection.encode(binary_cache).replace(b"imagespng", b"images/pg")
    with pytest.raises(ValueError, match="filename label"):
        BinarySection.decode(section, binary_cache)


def test_empty_binary_section(binary_cache: EventFileCache) -> None:
    binary_cache.binaries.clear()
    assert BinarySection.decode(BinarySection.encode(binary_cache), binary_cache) == []


@pytest.mark.parametrize("row", [0, 1])
def test_binary_section_enforces_exact_budget(
    binary_cache: EventFileCache, row: int
) -> None:
    # The first row uses a multi-byte label; the second has a multi-block proof.
    binary_cache.binaries = [binary_cache.binaries[row]]
    section = BinarySection.encode(binary_cache)
    minimum = BinarySection.minimum_size(binary_cache)
    for limit in (minimum - 1, minimum, len(section) - 1):
        with pytest.raises(ValueError, match="exceeds index budget"):
            BinarySection.encode(binary_cache, limit=limit)
    assert BinarySection.encode(binary_cache, limit=len(section)) == section


def test_binary_section_preflights_non_ascii_label(
    binary_cache: EventFileCache,
) -> None:
    class OversizeLabel(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            pytest.fail("oversize UTF-8 label must be rejected before encoding")

    entry = binary_cache.binaries[0]
    tag = "🌲" * 16
    binary_cache.binaries = [replace(entry, tag=tag)]
    section = BinarySection.encode(binary_cache)
    binary_cache.binaries = [replace(entry, tag=OversizeLabel(tag))]
    other_fields = len(entry.kind) + len(entry.extension) + len(entry.suffix)
    with pytest.raises(ValueError, match="UTF-8 text exceeds index budget"):
        BinarySection.encode(binary_cache, limit=len(section) - other_fields - 1)
