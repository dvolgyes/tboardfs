"""TF2 graph summaries give their native files and TensorBoard's GraphDefs.

test-logs/graphs and tests/golden/graphs come from
scripts/generate_graph_logs.py; each golden is the GraphDef that the
TensorBoard 2.21 graph plugin builds for one summary.
"""

import errno
import json
from pathlib import Path

from click.testing import CliRunner
from google.protobuf import unknown_fields
from google.protobuf.message import Message
import pytest

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main
from tboardfs.file_tree import SingleEventTree
from tboardfs.proto import event_pb2, graph_pb2

from test_incremental_parse import _files, _frame
from test_restart_purge import _start, _version

DT_STRING = 7
RUN = Path("test-logs/graphs")
GOLDEN = Path("tests/golden/graphs")
TAGS = sorted(path.stem for path in GOLDEN.iterdir())
UNCONVERTIBLE = "/graphs/func_in_seq/000000.conceptual.pb"


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    """Return the generated graph run."""
    return TensorBoardFS(RUN)


def _graph_files(fs: TensorBoardFS, tag: str) -> tuple[str, str]:
    """Return the native and the converted file of the only step of a tag.

    readdir, not getattr: getattr renders the file, which can fail with EIO.
    """
    native, converted = (
        f"/graphs/{tag}/{name}" for name in fs.readdir(f"/graphs/{tag}")[2:]
    )
    if native.endswith(".pb"):
        native, converted = converted, native
    return native, converted


@pytest.mark.parametrize("tag", TAGS)
def test_graphs_are_the_graphs_of_tensorboard(fs: TensorBoardFS, tag: str) -> None:
    """The converted file is TensorBoard's GraphDef, or EIO where TensorBoard fails."""
    native, converted = _graph_files(fs, tag)
    golden = GOLDEN / f"{tag}.pb"

    if native.endswith(".json"):
        assert converted.endswith(".conceptual.pb")
        assert "class_name" in json.loads(fs.read(native, 1 << 20, 0))
    else:
        assert native.endswith(".run_metadata.pb")
        assert graph_pb2.RunMetadata.FromString(
            fs.read(native, 1 << 20, 0)
        ).function_graphs
    if not golden.exists():
        assert (GOLDEN / f"{tag}.error").exists()
        with pytest.raises(OSError) as raised:
            fs.read(converted, 1 << 20, 0)
        assert raised.value.errno == errno.EIO
        return
    assert _comparable(fs.read(converted, 1 << 20, 0)) == _comparable(
        golden.read_bytes()
    )


def _comparable(data: bytes) -> tuple[graph_pb2.GraphDef, list[object]]:
    """Return a GraphDef without unknown fields, and its unknown fields.

    tboardfs keeps the fields that the vendored descriptor does not name, such
    as NodeDef device, as TensorFlow wrote them. TensorBoard writes them again
    in field order and sorts the maps inside them, so an unknown field is
    compared by number, wire type and the multiset of its bytes.
    """
    graph = graph_pb2.GraphDef.FromString(data)
    unknown: list[object] = []
    _collect_unknown(graph, "", unknown)
    graph.DiscardUnknownFields()
    return graph, unknown


def _collect_unknown(message: Message, path: str, unknown: list[object]) -> None:
    fields = unknown_fields.UnknownFieldSet(message)
    unknown.append(
        (
            path,
            sorted(
                (
                    f.field_number,
                    f.wire_type,
                    bytes(sorted(f.data)) if isinstance(f.data, bytes) else f.data,
                )
                for f in fields
            ),
        )
    )
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.message_type.GetOptions().map_entry:
            items = [(key, value[key]) for key in sorted(value)]
        elif field.is_repeated:
            items = list(enumerate(value))
        else:
            items = [("", value)]
        for key, item in items:
            if isinstance(item, Message):
                _collect_unknown(item, f"{path}.{field.name}[{key}]", unknown)


def test_conceptual_graph_has_layers_classes_and_dtypes(fs: TensorBoardFS) -> None:
    """Each layer is a node with its Keras class, dtype and inbound layers."""
    graph = graph_pb2.GraphDef.FromString(
        fs.read("/graphs/functional/000000.conceptual.pb", 1 << 20, 0)
    )
    nodes = {node.name: node for node in graph.node}

    assert list(nodes) == ["func/x", "func/dense_2", "func/out"]
    assert list(nodes["func/out"].input) == ["func/dense_2"]
    assert nodes["func/out"].attr["keras_class"].s == b"Dense"
    assert nodes["func/out"].attr["dtype"].type == 1


def test_function_graphs_are_merged_with_prefixes(fs: TensorBoardFS) -> None:
    """Several function graphs are graph_<i>/ scopes; called functions follow."""
    graph = graph_pb2.GraphDef.FromString(
        fs.read("/graphs/calls/000001.pb", 1 << 20, 0)
    )
    call = next(node for node in graph.node if node.op == "PartitionedCall")

    assert {node.name.split("/")[0] for node in graph.node} == {"graph_1", "graph_2"}
    assert call.name == "graph_2/PartitionedCall"
    assert list(call.input) == ["graph_2/values"]
    assert [f.signature.name for f in graph.library.function] == [
        call.attr["f"].func.name
    ]
    assert call.attr["f"].func.name.startswith("graph_2_")


def test_tf2_fixture_graphs_leave_plugins_and_tensors() -> None:
    """Keras and trace graph records are graph files only."""
    files = sorted(_files(TensorBoardFS(Path("test-logs/tf2"))))

    assert [path for path in files if "/graphs/" in path] == [
        "/keras/train/graphs/keras/000000.conceptual.pb",
        "/keras/train/graphs/keras/000000.json",
        "/summary/graphs/square/000000.pb",
        "/summary/graphs/square/000000.run_metadata.pb",
    ]
    assert not any("/plugins/" in path for path in files)
    assert {path.split("/")[3] for path in files if "/tensors/" in path} == {"raw"}


def _graph_event(
    blob: bytes,
    *,
    plugin: str = "graph_run_metadata_graph",
    content: bytes = b"1",
    step: int = 0,
) -> bytes:
    event = event_pb2.Event(wall_time=1.0, step=step)
    value = event.summary.value.add(tag="g")
    value.metadata.plugin_data.plugin_name = plugin
    value.metadata.plugin_data.content = content
    value.tensor.dtype = DT_STRING
    value.tensor.string_val.append(blob)
    return _frame(event)


def _run_metadata(*producers: int) -> bytes:
    run_metadata = graph_pb2.RunMetadata()
    for producer in producers:
        graph = run_metadata.function_graphs.add().pre_optimization_graph
        graph.versions.producer = producer
        graph.node.add(name="a", op="Const")
    return run_metadata.SerializeToString()


def _log(tmp_path: Path, *frames: bytes) -> TensorBoardFS:
    (tmp_path / "events.out.tfevents.host").write_bytes(b"".join(frames))
    return TensorBoardFS(tmp_path)


def test_graph_without_plugin_content_1_is_not_a_graph(tmp_path: Path) -> None:
    """TensorBoard shows a graph summary only with plugin content b"1"."""
    fs = _log(tmp_path, _graph_event(_run_metadata(1), content=b""))

    assert not any(path.startswith("/graphs") for path in _files(fs))


@pytest.mark.parametrize(
    ("blob", "plugin", "converted"),
    [
        (b"{not json", "graph_keras_model", "/graphs/g/000000.conceptual.pb"),
        (b'{"config": []}', "graph_keras_model", "/graphs/g/000000.conceptual.pb"),
        (b"\xff\xff", "graph_run_metadata_graph", "/graphs/g/000000.pb"),
        (_run_metadata(1, 2), "graph_run_metadata", "/graphs/g/000000.pb"),
    ],
)
def test_unconvertible_graph_is_eio(
    tmp_path: Path, blob: bytes, plugin: str, converted: str
) -> None:
    """A malformed config or graphs of different versions give EIO; the blob stays."""
    fs = _log(tmp_path, _graph_event(blob, plugin=plugin))
    native = next(
        f"/graphs/g/{name}"
        for name in fs.readdir("/graphs/g")[2:]
        if name not in converted
    )

    with pytest.raises(OSError) as raised:
        fs.read(converted, 1 << 20, 0)
    assert raised.value.errno == errno.EIO
    assert fs.read(native, 1 << 20, 0) == blob


def test_graph_summaries_are_dropped_on_restart(tmp_path: Path) -> None:
    """A graph summary is a tensor summary, so a START drops it as TensorBoard does."""
    first = _version() + _graph_event(_run_metadata(1), step=4)
    (tmp_path / "events.out.tfevents.0").write_bytes(first)
    (tmp_path / "events.out.tfevents.1").write_bytes(_version() + _start(2))

    assert not any(
        path.startswith("/graphs") for path in _files(TensorBoardFS(tmp_path))
    )


def test_copy_all_reports_an_unconvertible_graph_and_copies_the_rest(
    tmp_path: Path,
) -> None:
    """copy-all names the file that gives EIO, copies the others and fails."""
    result = CliRunner().invoke(main, ["copy-all", str(RUN), str(tmp_path / "out")])
    copied = sorted(
        "/" + path.relative_to(tmp_path / "out").as_posix()
        for path in (tmp_path / "out").rglob("*")
        if path.is_file()
    )
    listed = CliRunner().invoke(main, ["list", str(RUN)]).output.splitlines()

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"cannot convert: {UNCONVERTIBLE}" in result.stderr
    assert f"copied {len(listed) - 1} files" in result.stderr
    assert copied == sorted(set(listed) - {UNCONVERTIBLE})


def test_get_of_an_unconvertible_graph_is_an_error(tmp_path: Path) -> None:
    """get reports the file that gives EIO instead of a traceback."""
    result = CliRunner().invoke(main, ["get", str(RUN), UNCONVERTIBLE, "-o", "-"])

    assert result.exit_code == 1
    assert f"cannot convert virtual file: {UNCONVERTIBLE}" in result.stderr
    assert result.stdout_bytes == b""


def test_merged_gradients_name_the_prefixed_functions(tmp_path: Path) -> None:
    """A gradient of a function graph refers to the prefixed function names."""
    run_metadata = graph_pb2.RunMetadata.FromString(_run_metadata(1, 1))
    library = run_metadata.function_graphs[1].pre_optimization_graph.library
    library.function.add().signature.name = "f"
    library.gradient.add(function_name="f", gradient_func="f_grad")
    fs = _log(tmp_path, _graph_event(run_metadata.SerializeToString()))

    graph = graph_pb2.GraphDef.FromString(fs.read("/graphs/g/000000.pb", 1 << 20, 0))

    assert [node.name for node in graph.node] == ["graph_1/a", "graph_2/a"]
    assert [f.signature.name for f in graph.library.function] == ["graph_2_f"]
    assert [(g.function_name, g.gradient_func) for g in graph.library.gradient] == [
        ("graph_2_f", "graph_2_f_grad")
    ]


def test_copy_all_does_not_hide_other_read_errors(tmp_path: Path) -> None:
    """Only EIO is a file that cannot be converted; a vanished event file is an error."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.out.tfevents.host").write_bytes(_graph_event(_run_metadata(1)))
    tree = SingleEventTree(run)
    (run / "events.out.tfevents.host").unlink()

    with pytest.raises(OSError) as raised:
        tree.copy_all(tmp_path / "out")
    assert raised.value.errno == errno.ENOENT
