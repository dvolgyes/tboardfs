"""Verify compact worker rows against source payloads after their segment is released."""

import wave
from dataclasses import replace
from io import BytesIO
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from weakref import ref

from test_incremental_parse import _frame
from test_index_transport import _job
from test_payload_spans import _PNG

from tboardfs.extract import extract_binary_blob
from tboardfs.index_segment import Segment
from tboardfs.index_transport import _PackedResult
from tboardfs.model.binary_entry import BinaryEntry
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
