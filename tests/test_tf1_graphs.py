"""TF1 meta graphs and tagged run metadata give TensorBoard's graph and profiles.

test-logs/tf1_graphs and tests/golden/tf1_graphs come from
scripts/generate_tf1_graph_logs.py. The goldens are what the TensorBoard 2.21
event_accumulator keeps as the graph and the meta graph of a run, the op graph
of a graph_run_metadata summary, and the profile text that the graph plugin
shows for a RunMetadata.
"""

import errno
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from loguru import logger
from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

ROOT = Path(__file__).resolve().parent.parent

LOGS = ROOT / "test-logs/tf1_graphs"
GOLDEN = ROOT / "tests/golden/tf1_graphs"
GOLDENS = sorted(
    (run.name, golden.name) for run in GOLDEN.iterdir() for golden in run.iterdir()
)


def _virtual_path(name: str) -> str:
    """Return the virtual path of a golden: graph.pb, meta_graph.pb or <tag>.<step>.<ext>."""
    if name in {"graph.pb", "meta_graph.pb"}:
        return f"/graphs/{name}"
    tag, step, extension = name.split(".")
    suffix = "run_metadata.pbtxt" if extension == "pbtxt" else "pb"
    return f"/graphs/{tag}/{int(step):06d}.{suffix}"


@pytest.mark.parametrize(("run", "name"), GOLDENS)
def test_tf1_graph_files_are_the_files_of_tensorboard(run: str, name: str) -> None:
    """Graphs equal TensorBoard's as messages; native and text files equal its bytes.

    TensorBoard serializes a graph with its maps in an order that changes
    between processes, so graphs compare as parsed messages.
    """
    fs = TensorBoardFS(LOGS / run)
    data = fs.read(_virtual_path(name), 1 << 20, 0)
    golden = (GOLDEN / run / name).read_bytes()

    if name.endswith(".pb") and name != "meta_graph.pb":
        parse = compat_pb2.GraphDef.FromString
        assert parse(data) == parse(golden)
    else:
        assert data == golden


@pytest.mark.parametrize(
    ("run", "files"),
    [
        (
            "graph_first",
            ["graph.pb", "meta_graph.pb", "meta_graph_0.pb", "meta_graph_1.pb"],
        ),
        ("no_graph", ["graph.pb", "meta_graph.pb"]),
        (
            "profile",
            [
                "traced/000004.pb",
                "traced/000004.run_metadata.pb",
                "traced/000004.run_metadata.pbtxt",
            ],
        ),
    ],
)
def test_tf1_graph_trees(run: str, files: list[str]) -> None:
    """A graph_def wins over a later meta graph; each meta graph is also listed."""
    fs = TensorBoardFS(LOGS / run)

    assert sorted(_files(fs, "/graphs")) == [f"/graphs/{name}" for name in files]


def _log(tmp_path: Path, *events: compat_pb2.Event) -> TensorBoardFS:
    frames = b"".join(_frame(event) for event in events)
    (tmp_path / "events.out.tfevents.host").write_bytes(frames)
    return TensorBoardFS(tmp_path)


def _meta_graph(node: str) -> bytes:
    meta_graph = compat_pb2.MetaGraphDef()
    meta_graph.graph_def.node.add(name=node, op="Const")
    return meta_graph.SerializeToString()


@contextmanager
def _warnings() -> Iterator[list[str]]:
    """Collect the warnings logged inside the block."""
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        yield warnings
    finally:
        logger.remove(sink)


def test_graph_of_the_newest_meta_graph(tmp_path: Path) -> None:
    """Without a graph_def, graph.pb is the graph of the newest meta graph."""
    fs = _log(
        tmp_path,
        compat_pb2.Event(step=1, meta_graph_def=_meta_graph("old")),
        compat_pb2.Event(step=2, meta_graph_def=_meta_graph("new")),
    )

    graph = compat_pb2.GraphDef.FromString(fs.read("/graphs/graph.pb", 1 << 20, 0))
    assert [node.name for node in graph.node] == ["new"]
    assert fs.read("/graphs/meta_graph_0.pb", 1 << 20, 0) == _meta_graph("old")


def test_malformed_meta_graph_has_no_graph_file(tmp_path: Path) -> None:
    """A newest meta graph that does not parse keeps its native file but has no graph.pb."""
    events = (
        compat_pb2.Event(step=1, meta_graph_def=_meta_graph("old")),
        compat_pb2.Event(step=2, meta_graph_def=b"\xff\xff"),
    )
    with _warnings() as warnings:
        fs = _log(tmp_path, *events)
        files = fs.readdir("/graphs")[2:]
        native = fs.read("/graphs/meta_graph.pb", 1 << 20, 0)

    assert files == ["meta_graph.pb", "meta_graph_0.pb", "meta_graph_1.pb"]
    assert native == b"\xff\xff"
    assert len(warnings) == 1
    assert warnings[0].startswith("meta graph at offset ")


def test_tagged_run_metadata_is_a_profile(tmp_path: Path) -> None:
    """A tagged RunMetadata gives its native file and profile text, but no graph."""
    run_metadata = compat_pb2.RunMetadata()
    run_metadata.step_stats.dev_stats.add(device="/device:CPU:0")
    event = compat_pb2.Event(step=3)
    event.tagged_run_metadata.tag = "train/step"
    event.tagged_run_metadata.run_metadata = run_metadata.SerializeToString()
    fs = _log(tmp_path, event)

    assert sorted(_files(fs, "/graphs")) == [
        "/graphs/train/step/000003.run_metadata.pb",
        "/graphs/train/step/000003.run_metadata.pbtxt",
    ]
    text = fs.read("/graphs/train/step/000003.run_metadata.pbtxt", 1 << 20, 0)
    assert text == str(run_metadata).encode()


def test_malformed_tagged_run_metadata_has_no_profile_file(tmp_path: Path) -> None:
    """A tagged RunMetadata that does not parse keeps only its native file."""
    event = compat_pb2.Event(step=3)
    event.tagged_run_metadata.tag = "t"
    event.tagged_run_metadata.run_metadata = b"\xff\xff"
    with _warnings() as warnings:
        fs = _log(tmp_path, event)
        files = sorted(_files(fs, "/graphs"))
        native = fs.read("/graphs/t/000003.run_metadata.pb", 1 << 20, 0)

    assert files == ["/graphs/t/000003.run_metadata.pb"]
    assert native == b"\xff\xff"
    assert len(warnings) == 1
    assert warnings[0].startswith("run metadata t at step 3 in ")


def _tagged(run_metadata: bytes) -> compat_pb2.Event:
    event = compat_pb2.Event(step=3)
    event.tagged_run_metadata.tag = "t"
    event.tagged_run_metadata.run_metadata = run_metadata
    return event


@pytest.mark.parametrize(
    ("good", "bad", "path"),
    [
        (
            compat_pb2.Event(step=1, meta_graph_def=b"\x08\x01"),
            compat_pb2.Event(step=1, meta_graph_def=b"\xff\xff"),
            "/graphs/graph.pb",
        ),
        (
            _tagged(b"\x08\x01"),
            _tagged(b"\xff\xff"),
            "/graphs/t/000003.run_metadata.pbtxt",
        ),
    ],
)
def test_converted_file_of_a_changed_event_file_is_eio(
    tmp_path: Path, good: compat_pb2.Event, bad: compat_pb2.Event, path: str
) -> None:
    """A payload that no longer parses after indexing gives EIO on read."""
    fs = _log(tmp_path, good)
    fs.refresh_age_seconds = float("inf")
    directory, name = path.rsplit("/", 1)
    # readdir does not render the file, so the read below parses it again.
    assert name in fs.readdir(directory)
    log = tmp_path / "events.out.tfevents.host"
    log.write_bytes(_frame(bad))

    with pytest.raises(OSError) as raised:
        fs.read(path, 1 << 20, 0)
    assert raised.value.errno == errno.EIO
