"""Executor resource failures do not hide readable event files."""

from collections.abc import Callable
from concurrent.futures import Future
from errno import EAGAIN
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS
from tboardfs import parallel_index as cold
from tboardfs.index_segment import Job
from tboardfs.index_transport import _PackedResult

from test_incremental_parse import _loss
from test_parallel_index import _frames


def test_mount_reads_late_entry_when_pool_submission_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "events.out.tfevents.host"
    source.write_bytes(b"".join(_frames(padding=8 << 20)) + _loss(9, 17.0))
    expected = TensorBoardFS(tmp_path).read("/scalars/loss.json", 1 << 20, 0)
    assert b"17.0" in expected
    first: Future[_PackedResult] = Future()
    attempts: list[Job] = []

    class ExhaustedPool:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(
            self, _fn: Callable[[Job], _PackedResult], job: Job
        ) -> Future[_PackedResult]:
            attempts.append(job)
            if len(attempts) == 2:
                raise OSError(EAGAIN, "not enough processes")
            return first

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait and cancel_futures
            assert first.cancel()

    monkeypatch.setattr(cold, "_available_cpus", lambda: 2)
    monkeypatch.setattr(cold, "ProcessPoolExecutor", ExhaustedPool)
    assert (
        TensorBoardFS(tmp_path, index_workers=2).read("/scalars/loss.json", 1 << 20, 0)
        == expected
    )
    assert len(attempts) == 2
    assert first.cancelled()
