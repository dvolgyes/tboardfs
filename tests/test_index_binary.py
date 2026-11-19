"""Exercise disk binary rows independently of the enclosing index codec."""

from dataclasses import replace
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from tboardfs.file_status import FileIdentity
from tboardfs.frame_ref import FrameRef
from tboardfs.index_binary import BinarySection
from tboardfs.model import BinaryEntry, EventFileCache
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.indexed_blob import IndexedBlob
from tboardfs.range_proof import range_proof


@pytest.fixture
def binary_cache(tmp_path: Path) -> EventFileCache:
    """Build rows spanning two source frames and more than one proof block."""
    cache = EventFileCache(tmp_path / "old")
    progress = cache.progress
    progress.processed_pos = 9000
    progress.identity = FileIdentity(1, 2, None)
    progress.checked_status = (1, 2, None, 9000, 3, 4)
    progress.frames.append(120, 11)
    progress.frames.append(9000, 33)
    first = FrameRef(progress.identity, 0, 120, 11, progress.checked_status)
    second = FrameRef(progress.identity, 120, 8880, 33, progress.checked_status)
    payload = b"a" * 4200
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
