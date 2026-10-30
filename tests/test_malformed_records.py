"""A record that tboardfs cannot convert has no file and does not hide others."""

from collections.abc import Iterator
from contextlib import contextmanager
import itertools
from pathlib import Path

from loguru import logger
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import event_pb2

from test_incremental_parse import _frame

GARBAGE = b"\xff\xff\xff"
# Plugin names, plugin contents and tensors of the sweep below.
PLUGINS = (
    "", "scalars", "histograms", "images", "audio", "text", "pr_curves", "mesh",
    "hparams", "custom_scalars", "tensors", "videos", "graph_keras_model",
    "graph_run_metadata", "graph_run_metadata_graph", "graph_tagged_run_metadata",
    "projector", "profile", "other/plugin",
)  # fmt: skip
PLUGIN_CONTENTS = (b"", b"1", GARBAGE, b"\x0a\x03abc")
DTYPES = (None, 1, 2, 3, 7, 9, 10, 14, 19, 99)
SHAPES = ((), (0,), (3,), (2, 2), (2, 3), (1 << 40,), (3, 0))
TENSORS = (
    {},
    {"tensor_content": b"\x01\x02\x03"},
    {"tensor_content": b"\x00\x00\x80?\x00\x00\x00@"},
    {"string_val": [b"\x89PNG\r\n\x1a\n", b"2", b"abc", b"{}"]},
    {"string_val": [b"\xff\xfe", b"\x80", b"{not json"]},
    {"float_val": [float("nan"), float("inf"), 1.0]},
    {"int_val": [1, -5, 1 << 30]},
)


def _plugin_value(
    event: event_pb2.Event, plugin: str, content: bytes, tag: str = "t"
) -> event_pb2.SummaryValue:
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = plugin
    value.metadata.plugin_data.content = content
    return value


def _event(plugin: str, content: bytes) -> event_pb2.Event:
    """Return a record with one plugin value and one valid scalar."""
    event = event_pb2.Event(step=5, wall_time=10.0)
    value = _plugin_value(event, plugin, content)
    value.tensor.dtype = 7
    value.tensor.string_val.append(GARBAGE)
    event.summary.value.add(tag="loss", simple_value=1.0)
    return event


def _log(tmp_path: Path, *events: event_pb2.Event) -> TensorBoardFS:
    header = event_pb2.Event(wall_time=1.0, file_version="brain.Event:2")
    frames = b"".join(_frame(event) for event in (header, *events))
    (tmp_path / "events.out.tfevents.1.host").write_bytes(frames)
    return TensorBoardFS(tmp_path)


def _files(fs: TensorBoardFS, path: str = "/") -> Iterator[str]:
    """Yield every file path of a mount, after a read of its whole content."""
    for name in fs.readdir(path)[2:]:
        child = f"{path.rstrip('/')}/{name}"
        status = fs.getattr(child)
        if status["st_mode"] & 0o040000:
            yield from _files(fs, child)
            continue
        assert len(fs.read(child, status["st_size"] + 1, 0)) == status["st_size"]
        yield child


@contextmanager
def _warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


@pytest.mark.parametrize(
    ("plugin", "content", "hidden"),
    [
        ("hparams", GARBAGE, "/hparams/hparams.json"),
        ("mesh", GARBAGE, "/meshes"),
        ("custom_scalars", b"", "/custom_scalars/layout.json"),
    ],
)
def test_record_that_does_not_decode_has_no_file(
    tmp_path: Path, plugin: str, content: bytes, hidden: str
) -> None:
    """The record is skipped with one warning; the scalar beside it stays."""
    with _warnings() as messages:
        files = list(_files(_log(tmp_path, _event(plugin, content))))

    assert "/scalars/loss.json" in files
    assert not [path for path in files if path.startswith(hidden)]
    assert len(messages) == 1
    assert f"{plugin} record t at step 5" in messages[0]


@pytest.mark.parametrize("plugin", ["hparams", "custom_scalars"])
def test_record_that_does_not_decode_leaves_the_valid_record(
    tmp_path: Path, plugin: str
) -> None:
    """A valid record of the same plugin still gives its file."""
    valid = event_pb2.Event(step=6, wall_time=11.0)
    value = _plugin_value(valid, plugin, b"")
    value.tensor.dtype = 7
    value.tensor.string_val.append(b"")
    fs = _log(tmp_path, _event(plugin, GARBAGE), valid)

    directory = "/hparams" if plugin == "hparams" else "/custom_scalars"
    assert fs.readdir(directory)[2:] == [
        "hparams.json" if plugin == "hparams" else "layout.json"
    ]


@pytest.mark.parametrize(
    ("index", "plugin", "content", "fields"),
    [
        (index, *combination)
        for index, combination in enumerate(
            itertools.product(PLUGINS, PLUGIN_CONTENTS, TENSORS)
        )
    ],
)
def test_malformed_summaries_give_only_readable_files(
    tmp_path: Path, index: int, plugin: str, content: bytes, fields: dict[str, object]
) -> None:
    """Each plugin, content and tensor gives a tree whose files all read whole.

    Every plugin, plugin content and tensor appear together once; dtypes and
    shapes cycle through the combinations.
    """
    event = event_pb2.Event(step=5, wall_time=10.0)
    value = _plugin_value(event, plugin, content) if plugin else None
    tensor = (value or event.summary.value.add(tag="t")).tensor
    dtype = DTYPES[index % len(DTYPES)]
    if dtype is not None:
        tensor.dtype = dtype
    for size in SHAPES[index % len(SHAPES)]:
        tensor.tensor_shape.dim.add(size=size)
    for name, data in fields.items():
        if isinstance(data, list):
            getattr(tensor, name).extend(data)
        else:
            setattr(tensor, name, data)

    fs = _log(tmp_path, event)

    # _files lists every directory and asserts that each file reads whole.
    for path in _files(fs):
        assert fs.getattr(path)["st_mode"] & 0o100000
