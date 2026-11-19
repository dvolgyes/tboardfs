"""Speculative indexing leaves authoritative skip diagnostics to serial replay."""

from pathlib import Path
from typing import Never

from loguru import logger
import pytest

from tboardfs import TensorBoardFS, filesystem
from tboardfs import parallel_index as cold
from tboardfs.file_status import FileIdentity
from tboardfs.index_segment import Metadata, Segment, _SegmentError
from tboardfs.indexer import _EventIndexer
from tboardfs.model import EventFileCache

from test_incremental_parse import _frame, _loss
from test_tf2_summaries import _images


def test_summary_failure_replays_without_losing_sibling_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents.host"
    event = _images(1, b"image")
    event.summary.value.add(tag="sibling", simple_value=3)
    path.write_bytes(_frame(event))
    status = path.stat()

    def invalid_value(*_args: object) -> Never:
        raise OverflowError("cannot convert summary")

    monkeypatch.setattr(_EventIndexer, "add_summary", invalid_value)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        with pytest.raises(_SegmentError, match="cannot convert summary"):
            Segment.work(
                (
                    path,
                    0,
                    status.st_size,
                    FileIdentity.fingerprint(status),
                    None,
                    None,
                    None,
                    None,
                )
            )
        assert messages == []
        cache = _EventIndexer.parse_event_file(path, ignore_truncated=True)
    finally:
        logger.remove(sink)
    assert cache.scalars["sibling"]["value"].tolist() == [3.0]
    assert len(messages) == 1
    assert "cannot convert summary" in messages[0]


def test_malformed_plugin_after_prefix_warns_once_on_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.out.tfevents.host"
    step = cold._PREFIX_RECORDS + 1
    event = _images(step, b"unused")
    plugin = event.summary.value[0].metadata.plugin_data
    plugin.plugin_name = "hparams"
    plugin.content = b"\x80"
    event.summary.value.add(tag="sibling", simple_value=3)
    prefix = b"".join(_loss(i, 1.0) for i in range(step))
    path.write_bytes(prefix + _frame(event))
    monkeypatch.setattr(cold, "_available_cpus", lambda: 2)
    monkeypatch.setattr(filesystem, "_PARALLEL_MIN_BYTES", 0)
    original = cold.parallel_index
    attempted: list[EventFileCache | None] = []

    def observe(
        source: Path,
        workers: int,
        *,
        initial_metadata: Metadata | None = None,
        mountpoint: Path | None = None,
    ) -> EventFileCache | None:
        cache = original(
            source, workers, initial_metadata=initial_metadata, mountpoint=mountpoint
        )
        attempted.append(cache)
        return cache

    monkeypatch.setattr(cold, "parallel_index", observe)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        mount = TensorBoardFS(tmp_path, index_workers=2)
        mount.readdir("/")
    finally:
        logger.remove(sink)
    assert attempted == [None]
    assert len(messages) == 1
    assert f"skipping hparams record img at step {step}" in messages[0]
    cache = mount._state.runs[()].files[path]
    assert cache.scalars["sibling"]["value"].tolist() == [3.0]
