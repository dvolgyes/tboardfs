"""Read real producer edge cases without importing either writer stack."""

from base64 import b64decode
import json
from pathlib import Path
import struct

from tboardfs import TensorBoardFS

from test_incremental_parse import _files
from test_proto3_reference import _fields, _values

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
    assert _fields(_fields(frames[0][12:-4], 10)[0], 1) == [
        b"tensorboard.summary.writer.event_file_writer"
    ]


def test_historical_non_summary_events_and_obsolete_value() -> None:
    """Checkpoint, stop, and log messages keep both scalar rows at the same step."""
    run = CASES / "tb115"
    fs = TensorBoardFS(run)
    rows = json.loads(fs.read("/scalars/loss.json", 4096, 0))
    assert [(r["step"], r["wall_time"], r["value"]) for r in rows] == [
        (1, 2.0, 2.5),
        (1, 6.0, 1.25),
    ]
    assert fs.readdir("/histograms") == [".", ".."]
    assert fs.readdir("/distributions") == [".", ".."]

    frames = [fs.read(p, 4096, 0) for p in sorted(_files(fs, "/raw"))]
    assert b"".join(frames) == (run / "events.out.tfevents.tb115").read_bytes()
    header, first, log, checkpoint, stop, _ = (frame[12:-4] for frame in frames)
    assert _fields(header, 3) == [b"brain.Event:2"]
    assert not _fields(header, 10)
    values = _values(first)
    assert _fields(values["historical"], 3) == [b"\x00\xffopaque"]
    assert _fields(values["metadata-only"], 7) == [b"legacy/node"]
    assert _fields(_fields(log, 6)[0], 2) == [b"retained warning"]
    assert _fields(_fields(checkpoint, 7)[0], 2) == [b"checkpoint-1"]
    assert _fields(_fields(stop, 7)[0], 3) == [b"paused"]


def test_legacy_debug_summary_and_beholder_sidecar() -> None:
    """A debugger Event has a generic plugin view; a bare Summary is not an Event."""
    run = CASES / "tb115"
    fs = TensorBoardFS(run)
    entry = json.loads(fs.read("/plugins/debugger/activation/000001.json", 4096, 0))
    assert json.loads(entry["plugin_content"]) == {"device": "CPU:0", "outputSlot": 0}
    tensor = entry["tensor"]
    assert (tensor["shape"], tensor["dtype"]) == ([15], 2)
    assert struct.unpack("<15d", b64decode(tensor["tensor_content"]["base64"])) == (
        1,
        4,
        1,
        0,
        1,
        1,
        1,
        0,
        -2,
        2,
        0,
        8 / 3,
        -1,
        1,
        4,
    )
    (value,) = _fields((run / "plugins/beholder/frame.summary").read_bytes(), 1)
    assert _fields(value, 1) == [b"beholder-frame"]
    assert fs.readdir("/raw")[2:] == ["events.out.tfevents.tb115"]
    assert fs.readdir("/plugins")[2:] == ["debugger"]


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
