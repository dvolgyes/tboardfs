"""TF2 graph summaries give their native files and TensorBoard's GraphDefs.

test-logs/graphs and tests/golden/graphs come from
scripts/generate_graph_logs.py; each golden is the GraphDef that the
TensorBoard 2.21 graph plugin builds for one summary.
"""

import errno
import json
import os
from pathlib import Path

from click.testing import CliRunner
from google.protobuf import unknown_fields
from google.protobuf.message import Message
from loguru import logger
import pytest

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main
from tboardfs.proto import compat_pb2, graph_pb2

from test_incremental_parse import _files, _frame
from test_restart_purge import _start, _version

DT_STRING = 7
RUN = Path("test-logs/graphs")
GOLDEN = Path("tests/golden/graphs")
TAGS = sorted(path.stem for path in GOLDEN.iterdir())


@pytest.fixture(scope="module")
def fs() -> TensorBoardFS:
    """Return the generated graph run."""
    return TensorBoardFS(RUN)


def _graph_files(fs: TensorBoardFS, tag: str) -> tuple[str, str | None]:
    """Return the native and the converted file of the only step of a tag."""
    names = sorted(fs.readdir(f"/graphs/{tag}")[2:], key=len)
    native = next(
        name for name in names if name.endswith((".json", ".run_metadata.pb"))
    )
    converted = [
        f"/graphs/{tag}/{name}"
        for name in names
        if name != native and not name.endswith(".pbtxt")
    ]
    return f"/graphs/{tag}/{native}", converted[0] if converted else None


@pytest.mark.parametrize("tag", TAGS)
def test_graphs_are_the_graphs_of_tensorboard(fs: TensorBoardFS, tag: str) -> None:
    """The converted file is TensorBoard's GraphDef, or absent where TensorBoard fails."""
    native, converted = _graph_files(fs, tag)
    golden = GOLDEN / f"{tag}.pb"

    if native.endswith(".json"):
        assert converted is None or converted.endswith(".conceptual.pb")
        assert "class_name" in json.loads(fs.read(native, 1 << 20, 0))
    else:
        assert native.endswith(".run_metadata.pb")
        assert graph_pb2.RunMetadata.FromString(
            fs.read(native, 1 << 20, 0)
        ).function_graphs
    if not golden.exists():
        assert (GOLDEN / f"{tag}.error").exists()
        assert converted is None
        return
    assert converted is not None
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
    event = compat_pb2.Event(wall_time=1.0, step=step)
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
    ("blob", "plugin", "native", "profile"),
    [
        (b"{not json", "graph_keras_model", "000000.json", []),
        (b'{"config": []}', "graph_keras_model", "000000.json", []),
        (b"\xff\xff", "graph_run_metadata_graph", "000000.run_metadata.pb", []),
        (
            _run_metadata(1, 2),
            "graph_run_metadata",
            "000000.run_metadata.pb",
            ["000000.run_metadata.pbtxt"],
        ),
    ],
)
def test_unconvertible_graph_has_only_its_native_file(
    tmp_path: Path, blob: bytes, plugin: str, native: str, profile: list[str]
) -> None:
    """A malformed config or graphs of different versions have no GraphDef file.

    A RunMetadata that parses keeps its profile text.
    """
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        fs = _log(tmp_path, _graph_event(blob, plugin=plugin))
        files = fs.readdir("/graphs/g")[2:]
        data = fs.read(f"/graphs/g/{native}", 1 << 20, 0)
    finally:
        logger.remove(sink)

    assert files == [native, *profile]
    assert data == blob
    assert len(warnings) == 1
    assert warnings[0].startswith("graph g at step 0 in ")


def test_graph_of_a_changed_event_file_is_eio(tmp_path: Path) -> None:
    """A summary that no longer converts after indexing gives EIO on read."""
    name = b"d" * 100
    config = b'{"config": {"name": "m", "layers": [{"config": {"name": "%s"}}]}}' % name
    fs = _log(tmp_path, _graph_event(config, plugin="graph_keras_model"))
    assert "000000.conceptual.pb" in fs.readdir("/graphs/g")
    path = tmp_path / "events.out.tfevents.host"
    status = path.stat()
    broken = config.replace(b"}}]}}", b"}}]} ")
    path.write_bytes(_graph_event(broken, plugin="graph_keras_model"))
    # The same size, mtime and first bytes: the mount does not index it again.
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))

    with pytest.raises(OSError) as raised:
        fs.read("/graphs/g/000000.conceptual.pb", 1 << 20, 0)
    assert raised.value.errno == errno.EIO


def test_graph_summaries_are_dropped_on_restart(tmp_path: Path) -> None:
    """A graph summary is a tensor summary, so a START drops it as TensorBoard does."""
    first = _version() + _graph_event(_run_metadata(1), step=4)
    (tmp_path / "events.out.tfevents.0").write_bytes(first)
    (tmp_path / "events.out.tfevents.1").write_bytes(_version() + _start(2))

    assert not any(
        path.startswith("/graphs") for path in _files(TensorBoardFS(tmp_path))
    )


def test_copy_all_copies_a_run_with_an_unconvertible_graph(tmp_path: Path) -> None:
    """copy-all copies each listed file; the unconvertible GraphDef is not listed."""
    run = tmp_path / "run"
    run.mkdir()
    scalar = compat_pb2.Event(wall_time=1.0, step=0)
    scalar.summary.value.add(tag="loss", simple_value=1.0)
    _log(run, _frame(scalar), _graph_event(b"{not json", plugin="graph_keras_model"))
    result = CliRunner().invoke(main, ["copy-all", str(run), str(tmp_path / "out")])
    copied = sorted(
        "/" + path.relative_to(tmp_path / "out").as_posix()
        for path in (tmp_path / "out").rglob("*")
        if path.is_file()
    )
    listed = CliRunner().invoke(main, ["list", str(run)]).stdout.splitlines()

    assert result.exit_code == 0
    assert "/graphs/g/000000.json" in listed
    assert not any(path.endswith(".conceptual.pb") for path in listed)
    assert copied == sorted(listed)


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
