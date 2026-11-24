"""Different tags that map to the same virtual path each keep their own files."""

import json
from pathlib import Path

import pytest
from loguru import logger
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.file_tree import SingleEventTree
from tboardfs.model import RunCache, SidecarEntry
from tboardfs.paths import _DirectoryNode
from tboardfs.proto import compat_pb2
from tboardfs.render_node import render_node
from tboardfs.run_files import RunFiles
from tboardfs.run_tree import RunTree

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
    """The empty-component marker and an ordinary tag remain independent."""
    _write(tmp_path, _scalars(1, **{"untagged": 1.0, "": 2.0}))
    fs = TensorBoardFS(tmp_path)

    assert [row["value"] for row in _json(fs, "/scalars/untagged.json")] == [1.0]
    assert [row["value"] for row in _json(fs, "/scalars/~empty.json")] == [2.0]


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
    assert _json(fs, "/tensors/a/~empty/b/000001.json")["values"] == [2.0, 2.0]  # type: ignore[index]


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
    assert all(path.startswith("/histograms/~empty/") for path in histograms)


@pytest.fixture
def warnings() -> list[str]:
    """Return the warning messages that tboardfs logs during the test."""
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages  # type: ignore[misc]
    logger.remove(sink)


def test_a_file_directory_collision_is_logged_once_per_run(
    tmp_path: Path, warnings: list[str]
) -> None:
    """An append does not repeat a tag's existing rename warning."""
    _write(tmp_path, _scalars(1, **{"loss": 1.0, "loss.json/x": 2.0}))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, source_check_seconds=0)
    first = [path for path in _files(fs) if path.startswith("/scalars/")]
    _write(tmp_path, _scalars(2, **{"loss": 3.0, "loss.json/x": 4.0}))
    second = [path for path in _files(fs) if path.startswith("/scalars/")]

    assert first == second
    assert [row["value"] for row in _json(fs, "/scalars/loss.json~2/x.json")] == [
        2.0,
        4.0,
    ]
    assert len(warnings) == 1
    assert "'loss.json/x'" in warnings[0]


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

    assert fs.readdir("/", None)[2:] == ["bad", "good"]
    assert [row["value"] for row in _json(fs, "/good/scalars/value.json")] == [3.0]
    assert "/bad/tensors/b/000001.json~2/x/000001.json" in _files(fs)
    assert len([message for message in warnings if "'bad'" in message]) == 1
    assert "tensors/b/000001.json" in warnings[0]


def test_a_fixed_file_directory_clash_keeps_both(tmp_path: Path) -> None:
    """A trimmed profile directory and a same-name file remain readable."""
    _write(tmp_path, _scalars(1, value=1.0))
    (tmp_path / "plugins" / "profile").mkdir(parents=True)
    (tmp_path / "plugins" / "profile" / "a").write_text("file")
    (tmp_path / "x" / "profile" / "a").mkdir(parents=True)
    (tmp_path / "x" / "profile" / "a" / "c").write_text("below")
    fs = TensorBoardFS(tmp_path)
    fs.readdir("/")

    assert fs.read("/profile/a~2", 100, 0) == b"file"
    assert fs.read("/profile/a/c", 100, 0) == b"below"
    assert [row["value"] for row in _json(fs, "/scalars/value.json")] == [1.0]


def test_named_tags_and_fixed_sources_share_a_collision_namespace() -> None:
    """Fixed directories stay put; tagged directories and colliding leaves move."""
    files = RunFiles()
    run = RunCache(())
    tree = _DirectoryNode()
    existing = {"type": "text", "data": "existing"}
    occupied = {"type": "text", "data": "occupied"}
    sidecar = {"type": "text", "data": "sidecar"}
    scalar = {"type": "scalar", "data": "scalar"}
    nested = {"type": "scalar", "data": "nested"}
    step = {"type": "tensor", "data": "step"}
    files.add_fixed(("scalars", "loss.json", "existing"), existing)
    files.add_fixed(("scalars", "loss.json~2"), occupied)
    files.add_fixed(("scalars", "loss.json"), sidecar, identity="source")
    files.add_named(("scalars",), "loss", "json", scalar)
    files.add_named(("scalars",), "loss.json/x", "json", nested)
    files.add(("tensors",), "loss.json/x", "000001.json", step)

    files.fill(tree, run)

    children = tree["children"]
    scalars = children["scalars"]["children"]
    assert scalars["loss.json"]["children"] == {"existing": existing}
    assert scalars["loss.json~2"] is occupied
    assert scalars["loss.json~3"]["children"] == {"x.json": nested}
    assert scalars["loss.json~4"] is scalar
    assert scalars["loss.json~5"] is sidecar
    tensors = children["tensors"]["children"]
    assert tensors["loss.json~3"]["children"]["x"]["children"] == {"000001.json": step}


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_sidecar_leaves_and_plugin_roots(tmp_path: Path, reverse: bool) -> None:
    event = compat_pb2.Event(wall_time=1, step=1)
    plugins = ["\\", "%5C"]
    paths = ["plugins/profile", "profile"]
    if reverse:
        plugins.reverse()
        paths.reverse()
    for plugin in plugins:
        value = event.summary.value.add(tag="tag")
        value.metadata.plugin_data.plugin_name = plugin
        value.metadata.plugin_data.content = plugin.encode()
        value.tensor.dtype = DT_FLOAT
        value.tensor.float_val.append(1)
    _write(tmp_path, _frame(event))
    entries = []
    for path in paths:
        directory = tmp_path / path
        directory.mkdir(parents=True)
        (directory / "trace.txt").write_text(path)
        source = directory / "trace.txt"
        stat = source.stat()
        entries.append(
            SidecarEntry(
                source, "profile", ("trace.txt",), stat.st_size, stat.st_mtime_ns
            )
        )
    run = RunCache((), sidecars=tuple(entries))
    nodes = RunTree.build(run, ("json",), 6)["children"]["profile"]["children"]
    assert render_node(nodes["trace.txt"]) == b"plugins/profile"
    assert render_node(nodes["trace.txt~2"]) == b"profile"
    fs = TensorBoardFS(tmp_path)
    assert fs.read("/profile/trace.txt", 100, 0) == b"plugins/profile"
    assert fs.read("/profile/trace.txt~2", 100, 0) == b"profile"
    for name, plugin in [("%5C", "\\"), ("%255C", "%5C")]:
        assert _json(fs, f"/plugins/{name}/tag/000001.json")["plugin_name"] == plugin


def test_unusual_tags_list_read_and_copy_without_loss(tmp_path: Path) -> None:
    tags = {
        "/a//": "~empty/a/~empty/~empty",
        "~empty": "%7Eempty",
        ".": "%2E",
        "..": "%2E%2E",
        "%2E": "%252E",
        "literal~2": "literal%7E2",
        "\n": "%0A",
    }
    _write(tmp_path, _scalars(1, **dict.fromkeys(tags, 1.0)))
    tree = SingleEventTree(tmp_path / "events.out.tfevents.host", scalar_format="json")
    fs = TensorBoardFS(tmp_path, scalar_formats="json")
    paths = [f"/scalars/{name}.json" for name in tags.values()]
    assert tree.list_file_paths() == sorted(paths)
    destination = tmp_path / "out"
    tree.copy_all(destination)
    for path in paths:
        data = tree.read_file(path)
        assert data == fs.read(path, 1 << 20, 0)
        assert data == (destination / path.lstrip("/")).read_bytes()
