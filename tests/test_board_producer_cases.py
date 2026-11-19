"""Read real producer edge cases without importing either writer stack."""

import json
from pathlib import Path

from tboardfs import TensorBoardFS

from test_incremental_parse import _files

CASES = Path(__file__).parent / "data" / "producer-cases"


def test_native_tensorboard_retains_unknown_events_and_binary_metadata() -> None:
    run = CASES / "tensorboard"
    fs = TensorBoardFS(run)
    frames = [
        fs.read(path, 1 << 20, 0)
        for path in sorted(_files(fs, "/raw"))
        if path.endswith(".tfrecord")
    ]
    assert b"".join(frames) == (run / "events.out.tfevents.tensorboard").read_bytes()
    assert any(frame[12:-4] == b"\xa0\x06\x07" for frame in frames)
    assert any(b"\x00\xffmetadata" in frame[12:-4] for frame in frames)


def test_native_tensorboard_boundary_tags_remain_distinct() -> None:
    """Similar spellings and a file/directory collision must preserve each series."""
    fs = TensorBoardFS(CASES / "tensorboard")
    series = [
        [(row["step"], row["value"]) for row in json.loads(fs.read(path, 65536, 0))]
        for path in _files(fs, "/scalars")
        if path.endswith(".json")
    ]
    for value in (2.5, 3.75, 9.0, 10.0, 4.0):
        assert [(2, value)] in series
    assert [(1, 8.0), (2, 1.25), (2, 6.0), (2**53 + 7, 7.0)] in series


def test_tensorboardx_restart_order_and_epochs() -> None:
    fs = TensorBoardFS(CASES / "tensorboardx")
    rows = json.loads(fs.read("/scalars/train/batch/loss.json", 65536, 0))
    assert [(row["step"], row["value"], row["epoch"]) for row in rows] == [
        (0, 12.0, 1),
        (3, 9.0, 2),
        (3, 10.0, 2),
        (2**53, 8.0, 2**53),
        (2**53 + 1, 7.0, 2**54),
    ]
    histogram = json.loads(fs.read("/histograms/weights/000003.json", 65536, 0))
    assert histogram and all(
        row["step"] == 3 and row["epoch"] == 2 for row in histogram
    )
