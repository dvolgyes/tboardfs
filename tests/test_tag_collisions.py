"""Different tags that map to the same virtual path each keep their own files."""

import json
from pathlib import Path

from loguru import logger
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

from test_incremental_parse import _files, _frame

DT_FLOAT = 1


def _scalars(step: int, **values: float) -> bytes:
    event = compat_pb2.Event(wall_time=float(step), step=step)
    for tag, value in values.items():
        event.summary.value.add(tag=tag, simple_value=value)
    return _frame(event)


def _write(tmp_path: Path, *records: bytes) -> None:
    with (tmp_path / "events.out.tfevents.host").open("ab") as handle:
        handle.write(b"".join(records))


def _json(fs: TensorBoardFS, path: str) -> object:
    return json.loads(fs.read(path, 1 << 20, 0))


def test_an_empty_tag_does_not_hide_the_untagged_tag(tmp_path: Path) -> None:
    """The tag that the path spells keeps it; the empty tag gets a suffix."""
    _write(tmp_path, _scalars(1, **{"untagged": 1.0, "": 2.0}))
    fs = TensorBoardFS(tmp_path)

    assert [row["value"] for row in _json(fs, "/scalars/untagged.json")] == [1.0]
    assert [row["value"] for row in _json(fs, "/scalars/untagged~2.json")] == [2.0]


def test_tags_that_differ_in_slashes_keep_their_own_files(tmp_path: Path) -> None:
    """A tensor tag with a double slash does not replace the plain one."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    for tag, number in (("a//b", 2.0), ("a/b", 1.0)):
        value = event.summary.value.add(tag=tag)
        value.metadata.plugin_data.plugin_name = "tensors"
        value.tensor.dtype = DT_FLOAT
        value.tensor.tensor_shape.dim.add(size=2)
        value.tensor.float_val.extend([number, number])
    _write(tmp_path, _frame(event))
    fs = TensorBoardFS(tmp_path)

    assert _json(fs, "/tensors/a/b/000001.json")["values"] == [1.0, 1.0]  # type: ignore[index]
    assert _json(fs, "/tensors/a/b~2/000001.json")["values"] == [2.0, 2.0]  # type: ignore[index]


def test_a_renamed_tag_has_the_same_name_in_every_tab(tmp_path: Path) -> None:
    """The histogram of the empty tag follows its scalar, with no collision there."""
    event = compat_pb2.Event(wall_time=1.0, step=1)
    event.summary.value.add(tag="untagged", simple_value=1.0)
    histo = event.summary.value.add(tag="").histo
    histo.bucket_limit.extend([1.0, 2.0])
    histo.bucket.extend([3.0, 4.0])
    _write(tmp_path, _frame(event))
    fs = TensorBoardFS(tmp_path)

    histograms = [path for path in _files(fs) if path.startswith("/histograms/")]
    assert histograms
    assert all(path.startswith("/histograms/untagged~2/") for path in histograms)


@pytest.fixture
def warnings() -> list[str]:
    """Return the warning messages that tboardfs logs during the test."""
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages  # type: ignore[misc]
    logger.remove(sink)


def test_a_collision_is_logged_once_per_run(
    tmp_path: Path, warnings: list[str]
) -> None:
    """The warning names both tags, and a growing run does not repeat it."""
    _write(tmp_path, _scalars(1, **{"untagged": 1.0, "": 2.0}))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    first = list(_files(fs))
    _write(tmp_path, _scalars(2, **{"untagged": 3.0, "": 4.0}))
    second = list(_files(fs))

    assert first == second
    assert [row["value"] for row in _json(fs, "/scalars/untagged~2.json")] == [2.0, 4.0]
    assert len(warnings) == 1
    assert "'untagged'" in warnings[0]
    assert "''" in warnings[0]
    assert "untagged~2" in warnings[0]


def _tensors(step: int, *tags: str) -> bytes:
    event = compat_pb2.Event(wall_time=float(step), step=step)
    for number, tag in enumerate(tags):
        value = event.summary.value.add(tag=tag)
        value.metadata.plugin_data.plugin_name = "tensors"
        value.tensor.dtype = DT_FLOAT
        value.tensor.tensor_shape.dim.add(size=2)
        value.tensor.float_val.extend([number, number])
    return _frame(event)


@pytest.mark.parametrize(
    "tags",
    [("a", "a/000001.npy"), ("a/000001.npy", "a")],
    ids=["file first", "dir first"],
)
def test_a_tag_named_like_a_step_file_gets_a_suffix(
    tmp_path: Path, tags: tuple[str, str]
) -> None:
    """The step file of tag a keeps its name; the tag a/000001.npy moves."""
    _write(tmp_path, _tensors(1, *tags))
    fs = TensorBoardFS(tmp_path)
    number = {tag: float(index) for index, tag in enumerate(tags)}

    assert fs.readdir("/", None)
    assert fs.getattr("/tensors/a/000001.npy")["st_size"] > 0
    assert _json(fs, "/tensors/a/000001.json")["values"] == [number["a"]] * 2  # type: ignore[index]
    moved = _json(fs, "/tensors/a/000001.npy~2/000001.json")
    assert moved["values"] == [number["a/000001.npy"]] * 2  # type: ignore[index]


def test_a_tag_named_like_a_scalar_file_moves_in_every_tab(tmp_path: Path) -> None:
    """The tag loss.json/x does not hide loss.json; its tensor moves with it."""
    _write(tmp_path, _scalars(1, **{"loss": 1.0, "loss.json/x": 2.0}))
    _write(tmp_path, _tensors(1, "loss.json/x"))
    fs = TensorBoardFS(tmp_path)

    assert [row["value"] for row in _json(fs, "/scalars/loss.json")] == [1.0]
    assert [row["value"] for row in _json(fs, "/scalars/loss.json~2/x.json")] == [2.0]
    assert "/tensors/loss.json~2/x/000001.json" in _files(fs)


def test_a_file_conflict_in_one_run_does_not_hide_other_runs(
    tmp_path: Path, warnings: list[str]
) -> None:
    """Every run stays readable, and the warning names the file in the way."""
    for run in ("bad", "good"):
        (tmp_path / run).mkdir()
        _write(tmp_path / run, _tensors(1, "b", "b/000001.json/x"))
    _write(tmp_path / "good", _scalars(1, value=3.0))
    fs = TensorBoardFS(tmp_path)

    assert fs.readdir("/", None)[-2:] == ["bad", "good"]
    assert [row["value"] for row in _json(fs, "/good/scalars/value.json")] == [3.0]
    assert "/bad/tensors/b/000001.json~2/x/000001.json" in _files(fs)
    assert len([message for message in warnings if "'bad'" in message]) == 1
    assert "tensors/b/000001.json" in warnings[0]


def test_a_clash_without_a_tag_leaves_one_file_out(
    tmp_path: Path, warnings: list[str]
) -> None:
    """Two profile sidecars meet at profile/a; no suffix can separate them."""
    _write(tmp_path, _scalars(1, value=1.0))
    (tmp_path / "plugins" / "profile").mkdir(parents=True)
    (tmp_path / "plugins" / "profile" / "a").write_text("file")
    (tmp_path / "x" / "profile" / "a").mkdir(parents=True)
    (tmp_path / "x" / "profile" / "a" / "c").write_text("below")
    fs = TensorBoardFS(tmp_path)

    assert fs.read("/profile/a", 100, 0) == b"file"
    assert [row["value"] for row in _json(fs, "/scalars/value.json")] == [1.0]
    assert len(warnings) == 1
    assert "'profile/a/c' is left out" in warnings[0]
