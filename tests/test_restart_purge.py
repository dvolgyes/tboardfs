"""A resumed run hides the rows that a SessionLog.START replaces.

Expected values follow TensorBoard's classic EventAccumulator.
"""

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import event_pb2

from test_incremental_parse import PNG, _contents, _frame

VERSION_2 = "brain.Event:2"
# Wire bytes as TensorFlow writes them, so a wrong field number in the vendored
# descriptor fails: file_version is field 3, session_log is field 7 with
# status START (field 1, value 1).
FILE_VERSION_KEY = b"\x1a"
SESSION_START = b"\x3a\x02\x08\x01"


def _version(text: str = VERSION_2) -> bytes:
    raw = FILE_VERSION_KEY + bytes([len(text)]) + text.encode()
    return _frame(event_pb2.Event.FromString(raw))


def _start(step: int | None) -> bytes:
    event = event_pb2.Event(wall_time=2.0)
    if step is not None:
        event.step = step
    return _frame(event_pb2.Event.FromString(event.SerializeToString() + SESSION_START))


def _values(step: int, value: float) -> bytes:
    """Return one event with a scalar, an image and a histogram."""
    event = event_pb2.Event(wall_time=100.0 + step, step=step)
    event.summary.value.add(tag="loss", simple_value=value)
    image = event.summary.value.add(tag="img").image
    image.height = image.width = 1
    image.encoded_image_string = PNG
    histogram = event.summary.value.add(tag="weights").histo
    histogram.min = 0.0
    histogram.max = value
    histogram.bucket_limit.extend([value])
    histogram.bucket.extend([1.0])
    return _frame(event)


def _run(*parts: bytes) -> bytes:
    return b"".join(parts)


def _loss(fs: TensorBoardFS) -> list[tuple[int, float]]:
    data = json.loads(fs.read("/scalars/loss.json", 1 << 20, 0))
    return [(row["step"], row["value"]) for row in data]


def _histogram_steps(fs: TensorBoardFS) -> list[tuple[int, float]]:
    rows = []
    for name in fs.readdir("/histograms/weights"):
        if name.endswith(".npz"):
            data = fs.read(f"/histograms/weights/{name}", 1 << 20, 0)
            arrays = np.load(BytesIO(data))
            rows.append((int(arrays["step"][0]), float(arrays["bucket_right"][0])))
    return rows


def _write(tmp_path: Path, *files: bytes) -> TensorBoardFS:
    for index, data in enumerate(files):
        (tmp_path / f"events.out.tfevents.{index}").write_bytes(data)
    return TensorBoardFS(tmp_path)


FIRST_RUN = _run(_version(), *(_values(step, 10.0 + step) for step in range(6)))
RESUMED = [(0, 10.0), (1, 11.0), (2, 12.0), (3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)]


def test_start_in_a_later_file_drops_earlier_rows_from_its_step(
    tmp_path: Path,
) -> None:
    """A run resumed at step 3 shows steps 3 and up from the new file only."""
    second = _run(_version(), _start(3), *(_values(s, float(s)) for s in range(3, 7)))
    fs = _write(tmp_path, FIRST_RUN, second)

    assert _loss(fs) == RESUMED
    assert _histogram_steps(fs) == RESUMED
    assert fs.readdir("/images/img")[2:] == [f"{step:06d}.png" for step in range(7)]


def test_start_in_the_same_file_drops_only_rows_before_it(tmp_path: Path) -> None:
    """Rows after the START stay, also when their step is not new."""
    data = _run(FIRST_RUN, _start(3), *(_values(s, float(s)) for s in range(3, 7)))
    fs = _write(tmp_path, data)

    assert _loss(fs) == RESUMED


def test_start_without_a_step_drops_all_earlier_rows(tmp_path: Path) -> None:
    """A missing step counts as step 0, as in TensorBoard."""
    fs = _write(tmp_path, FIRST_RUN, _run(_version(), _start(None), _values(9, 9.0)))

    assert _loss(fs) == [(9, 9.0)]


@pytest.mark.parametrize(("start", "kept"), [(0, False), (1, True)])
def test_rows_without_a_step_count_as_step_zero(
    tmp_path: Path, start: int, kept: bool
) -> None:
    """A row whose writer left out step 0 is dropped by a START at step 0."""
    event = event_pb2.Event(wall_time=100.0)
    event.summary.value.add(tag="loss", simple_value=5.0)
    fs = _write(
        tmp_path, _run(_version(), _frame(event)), _run(_version(), _start(start))
    )

    assert ("loss.json" in fs.readdir("/scalars")) == kept


def test_file_version_applies_to_later_files(tmp_path: Path) -> None:
    """A file without its own version uses the version of the previous file."""
    fs = _write(tmp_path, FIRST_RUN, _run(_start(5), _values(5, 0.0)))

    assert _loss(fs)[-2:] == [(4, 14.0), (5, 0.0)]


@pytest.mark.parametrize("version", [None, "brain.Event:1", "no version"])
def test_start_in_old_files_is_ignored(tmp_path: Path, version: str | None) -> None:
    """Without file version 2 or later, TensorBoard does not purge on START."""
    head = b"" if version is None else _version(version)
    data = _run(head, _values(0, 1.0), _values(1, 2.0), _start(1), _values(1, 3.0))
    fs = _write(tmp_path, data)

    assert _loss(fs) == [(0, 1.0), (1, 2.0), (1, 3.0)]


def test_graphs_are_not_dropped(tmp_path: Path) -> None:
    """TensorBoard keeps the graph of a run after a restart."""
    graph = _frame(event_pb2.Event(wall_time=1.5, step=4, graph_def=b"graph"))
    fs = _write(tmp_path, _run(FIRST_RUN, graph), _run(_version(), _start(0)))

    assert fs.read("/graphs/graph.pb", 100, 0) == b"graph"
    assert fs.readdir("/scalars")[2:] == []


@pytest.mark.parametrize("seed", range(10))
def test_growing_resumed_run_equals_a_fresh_parse(tmp_path: Path, seed: int) -> None:
    """Appends split at any byte give the purge result of one full parse."""
    rng = np.random.default_rng(seed)
    second = _run(
        _version(),
        _values(7, 0.0),
        _start(int(rng.integers(0, 8))),
        *(_values(s, float(s)) for s in range(2, 9)),
        _start(int(rng.integers(0, 8))),
        _values(4, -1.0),
    )
    (tmp_path / "events.out.tfevents.0").write_bytes(FIRST_RUN)
    event = tmp_path / "events.out.tfevents.1"
    growing = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    start = 0
    cuts = np.sort(rng.integers(1, len(second), size=4)).tolist()
    for cut in [*cuts, len(second)]:
        with event.open("ab") as handle:
            handle.write(second[start:cut])
        start = cut
        _contents(growing)

    assert _contents(growing) == _contents(TensorBoardFS(tmp_path))
