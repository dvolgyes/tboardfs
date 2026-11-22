"""A row takes the epoch scalar written at its step.

PyTorch Lightning writes each metric and then the epoch scalar at the same
step, so the epoch read before the metric is one epoch late.
"""

import json
from pathlib import Path
from typing import Any

from test_incremental_parse import _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).resolve().parent.parent


def _scalar(step: int, tag: str, value: float) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag=tag, simple_value=value)
    return _frame(event)


def _histogram(step: int) -> bytes:
    event = compat_pb2.Event(wall_time=100.0 + step, step=step)
    histogram = event.summary.value.add(tag="weights").histo
    histogram.max = 1.0
    histogram.bucket_limit.append(1.0)
    histogram.bucket.append(1.0)
    return _frame(event)


def _epochs(fs: TensorBoardFS, path: str) -> list[tuple[int, Any]]:
    rows = json.loads(fs.read(path, 1 << 20, 0))
    return sorted({(row["step"], row["epoch"]) for row in rows})


def _lightning_epoch(step: int, epoch: int) -> bytes:
    """Return a metric, a histogram and then the epoch, as Lightning writes them."""
    return _scalar(step, "loss", 1.0) + _histogram(step) + _scalar(step, "epoch", epoch)


def test_a_metric_takes_the_epoch_written_after_it_at_its_step(tmp_path: Path) -> None:
    """Each table row gets the epoch of its own step."""
    data = b"".join(_lightning_epoch(step, step // 2) for step in range(4))
    (tmp_path / "events.out.tfevents.0").write_bytes(data)
    fs = TensorBoardFS(tmp_path)

    expected = [(0, 0.0), (1, 0.0), (2, 1.0), (3, 1.0)]
    assert _epochs(fs, "/scalars/loss.json") == expected
    for step, epoch in expected:
        name = f"{step:06d}.json"
        assert _epochs(fs, f"/histograms/weights/{name}") == [(step, epoch)]
        assert _epochs(fs, f"/distributions/weights/{name}") == [(step, epoch)]


def test_a_step_without_an_epoch_keeps_the_last_epoch(tmp_path: Path) -> None:
    """A step with no epoch scalar uses the epoch read before it."""
    data = b"".join(
        [
            _scalar(0, "epoch", 4.0),
            _scalar(0, "loss", 1.0),
            _scalar(5, "loss", 2.0),
            _scalar(9, "epoch", 6.0),
        ]
    )
    (tmp_path / "events.out.tfevents.0").write_bytes(data)

    assert _epochs(TensorBoardFS(tmp_path), "/scalars/loss.json") == [
        (0, 4.0),
        (5, 4.0),
    ]


def test_the_epoch_of_a_step_can_be_in_a_later_file(tmp_path: Path) -> None:
    """The epoch scalar applies to its step across the files of a run."""
    (tmp_path / "events.out.tfevents.0").write_bytes(_scalar(7, "loss", 1.0))
    (tmp_path / "events.out.tfevents.1").write_bytes(_scalar(7, "epoch", 3.0))

    assert _epochs(TensorBoardFS(tmp_path), "/scalars/loss.json") == [(7, 3.0)]


def test_lightning_rows_have_the_epoch_of_their_step() -> None:
    """The fixture log trains 2 epochs of 2 batches."""
    fs = TensorBoardFS(ROOT / "test-logs/lightning/version_0")

    assert _epochs(fs, "/scalars/train_loss.json") == [
        (0, 0.0),
        (1, 0.0),
        (2, 1.0),
        (3, 1.0),
    ]
    assert _epochs(fs, "/scalars/val_loss.json") == [(1, 0.0), (3, 1.0)]


def test_large_steps_sort_stably_after_epoch_carry_and_append(tmp_path: Path) -> None:
    first = 2**53
    (tmp_path / "events.out.tfevents.0").write_bytes(
        _scalar(first + 1, "loss", 1) + _scalar(first, "epoch", 4)
    )
    later = tmp_path / "events.out.tfevents.1"
    later.write_bytes(
        _scalar(first - 1, "loss", 2)
        + _histogram(first - 1)
        + _scalar(first + 1, "epoch", 5)
        + _scalar(first, "loss", 3)
    )
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, source_check_seconds=0)

    def rows() -> list[tuple[int, float, float]]:
        data = json.loads(fs.read("/scalars/loss.json", 1 << 20, 0))
        return [(row["step"], row["epoch"], row["value"]) for row in data]

    assert rows() == [(first - 1, 4, 2), (first, 4, 3), (first + 1, 5, 1)]
    assert _epochs(fs, f"/histograms/weights/{first - 1}.json") == [(first - 1, 4)]
    with later.open("ab") as handle:
        handle.write(_scalar(first + 1, "loss", 4) + _scalar(first + 1, "epoch", 6))
    assert rows() == [
        (first - 1, 4, 2),
        (first, 4, 3),
        (first + 1, 6, 1),
        (first + 1, 6, 4),
    ]


def test_logged_nan_epoch_ends_cross_file_carry(tmp_path: Path) -> None:
    (tmp_path / "events.out.tfevents.0").write_bytes(_scalar(1, "epoch", 4))
    (tmp_path / "events.out.tfevents.1").write_bytes(
        _scalar(2, "loss", 1)
        + _scalar(3, "epoch", float("nan"))
        + _scalar(4, "loss", 2)
    )
    assert _epochs(TensorBoardFS(tmp_path), "/scalars/loss.json") == [
        (2, 4),
        (4, None),
    ]
