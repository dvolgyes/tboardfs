"""A value that tboardfs cannot convert hides its own file, not the mount."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_malformed_records import _files, _log, _plugin_value

from tboardfs import TensorBoardFS
from tboardfs.histogram_files import _HistogramFiles
from tboardfs.indexer import _EventIndexer
from tboardfs.mesh import _MeshFiles
from tboardfs.plugin_files import _TypedPluginFiles
from tboardfs.proto import compat_pb2
from tboardfs.proto import hparams_plugin_data_pb2 as hparams_pb2
from tboardfs.proto import mesh_plugin_data_pb2 as mesh_pb2


def _with_loss(event: compat_pb2.Event) -> compat_pb2.Event:
    event.summary.value.add(tag="loss", simple_value=1.0)
    return event


def _tensor_event(
    shape: list[int], plugin: str = "", dtype: int = 1
) -> compat_pb2.Event:
    event = compat_pb2.Event(step=1, wall_time=3.0)
    value = (
        _plugin_value(event, plugin, b"")
        if plugin
        else event.summary.value.add(tag="t")
    )
    value.tensor.dtype = dtype
    for size in shape:
        value.tensor.tensor_shape.dim.add(size=size)
    if dtype == 7:
        value.tensor.string_val.append(b"x")
    else:
        value.tensor.float_val.extend([1.0, 2.0])
    return event


def _mesh_value(
    event: compat_pb2.Event, content_type: int, array: np.ndarray, suffix: str
) -> None:
    metadata = mesh_pb2.MeshPluginData(
        name="m", content_type=content_type, shape=list(array.shape)
    )
    value = _plugin_value(event, "mesh", metadata.SerializeToString(), f"m{suffix}")
    value.tensor.dtype = 1
    for size in array.shape:
        value.tensor.tensor_shape.dim.add(size=size)
    value.tensor.tensor_content = np.asarray(array, "<f4").tobytes()


def test_a_deeply_nested_keras_config_has_no_graph_file(tmp_path: Path) -> None:
    """The JSON config stays; the conceptual graph is not converted."""
    event = compat_pb2.Event(step=2, wall_time=3.0)
    value = _plugin_value(event, "graph_keras_model", b"1", "keras")
    value.tensor.dtype = 7
    value.tensor.string_val.append(b"[" * 100_000)
    files = list(_files(_log(tmp_path, _with_loss(event))))

    assert "/scalars/loss.json" in files
    assert "/graphs/keras/000002.json" in files
    assert "/graphs/keras/000002.conceptual.pb" not in files


def test_hparams_with_infinite_domain_value_use_the_json_spelling(
    tmp_path: Path,
) -> None:
    data = hparams_pb2.HParamsPluginData(version=0)
    info = data.experiment.hparam_infos.add(name="lr")
    info.domain_discrete.values.add().number_value = math.inf
    event = compat_pb2.Event(step=0, wall_time=3.0)
    value = _plugin_value(event, "hparams", data.SerializeToString())
    value.tensor.dtype = 1
    value.tensor.float_val.append(0)
    fs = _log(tmp_path, _with_loss(event))

    text = fs.read("/hparams/hparams.json", 1 << 20, 0).decode()
    assert '"Infinity"' in text
    assert "/scalars/loss.json" in list(_files(fs))


def test_hparams_with_nan_list_value_use_the_json_spelling(tmp_path: Path) -> None:
    data = hparams_pb2.HParamsPluginData(version=0)
    hparam = data.session_start_info.hparams["opt"]
    hparam.list_value.values.add().number_value = math.nan
    event = compat_pb2.Event(step=0, wall_time=3.0)
    _plugin_value(event, "hparams", data.SerializeToString())
    fs = _log(tmp_path, _with_loss(event))

    text = fs.read("/hparams/hparams.json", 1 << 20, 0).decode()
    assert '"NaN"' in text
    assert "/scalars/loss.json" in list(_files(fs))


@pytest.mark.parametrize(
    ("content_type", "suffix", "array"),
    [
        (2, "_FACE", np.array([[[0, 1, np.nan]]])),
        (2, "_FACE", np.array([[[0, 1, 2.5]]])),
        (2, "_FACE", np.array([[[0, 1, 1e30]]])),
        (3, "_COLOR", np.full((1, 3, 3), np.nan)),
    ],
)
def test_a_mesh_with_bad_faces_or_colors_keeps_its_vertices(
    tmp_path: Path, content_type: int, suffix: str, array: np.ndarray
) -> None:
    """The NPZ and metadata files stay; the OBJ file has only valid lines."""
    event = compat_pb2.Event(step=1, wall_time=3.0)
    _mesh_value(event, 1, np.zeros((1, 3, 3)), "_VERTEX")
    _mesh_value(event, content_type, array, suffix)
    fs = _log(tmp_path, _with_loss(event))
    files = list(_files(fs))

    assert "/scalars/loss.json" in files
    assert "/meshes/m/000001.npz" in files
    obj = fs.read("/meshes/m/000001.obj", 1 << 20, 0).decode()
    assert obj.count("\nv ") == 3
    assert "nan" not in obj
    assert "\nf " not in obj


@pytest.mark.parametrize(
    ("shape", "plugin", "dtype"),
    [
        ([1 << 62, 1 << 62, 0], "", 1),
        ([2] + [1] * 64, "", 1),
        ([(1 << 63) - 1, 0], "text", 7),
        ([1 << 62, 1 << 62, 0], "histograms", 2),
    ],
)
def test_a_tensor_with_an_impossible_shape_has_no_file(
    tmp_path: Path, shape: list[int], plugin: str, dtype: int
) -> None:
    """The record keeps its raw JSON file, but has no converted array."""
    files = list(
        _files(_log(tmp_path, _with_loss(_tensor_event(shape, plugin, dtype))))
    )

    assert "/scalars/loss.json" in files
    assert not [
        path
        for path in files
        if path != "/llms.txt" and path.endswith((".npy", ".txt"))
    ]


def test_a_value_that_fails_does_not_hide_later_values(tmp_path: Path) -> None:
    """The bad histogram is the first value of its record; the scalar is later."""
    event = _tensor_event([1 << 62, 1 << 62, 0], "histograms", 2)
    event.summary.value.add(tag="later", simple_value=7.0)
    files = list(_files(_log(tmp_path, event)))

    assert "/scalars/later.json" in files


@pytest.mark.parametrize("wall_time", [math.nan, math.inf, -math.inf, 1e300, -5.0])
def test_a_file_time_is_a_valid_timestamp(tmp_path: Path, wall_time: float) -> None:
    event = compat_pb2.Event(step=1, wall_time=wall_time)
    event.summary.value.add(tag="loss", simple_value=1.0)
    fs: TensorBoardFS = _log(tmp_path, event)

    mtime = fs.getattr("/scalars/loss.json")["st_mtime"]
    assert math.isfinite(mtime)
    assert 0 <= mtime < 1 << 33


def _two_tensors(bad: str = "bad") -> compat_pb2.Event:
    event = compat_pb2.Event(step=1, wall_time=3.0)
    for tag in (bad, "good"):
        value = event.summary.value.add(tag=tag)
        value.tensor.dtype = 1
        value.tensor.tensor_shape.dim.add(size=2)
        value.tensor.float_val.extend([1.0, 2.0])
    return event


def test_a_value_that_fails_to_index_hides_only_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _EventIndexer.add_plugin
    monkeypatch.setattr(
        _EventIndexer,
        "add_plugin",
        lambda cache, value, identity: (
            original(cache, value, identity)
            if identity[0] != "bad"
            else _raise(RuntimeError("indexer bug"))
        ),
    )
    event = _with_loss(_two_tensors())
    files = list(_files(_log(tmp_path, event)))

    assert "/scalars/loss.json" in files
    assert "/tensors/good/000001.npy" in files
    assert not [path for path in files if "/bad" in path]


def test_a_value_that_fails_to_index_stops_a_strict_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _EventIndexer,
        "add_plugin",
        lambda *_: _raise(RuntimeError("indexer bug")),
    )
    _log(tmp_path, _two_tensors())

    with pytest.raises(RuntimeError, match="indexer bug"):
        _EventIndexer.parse_event_file(next(tmp_path.glob("events.*")))


def test_a_tensor_that_fails_to_build_hides_only_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tensor keeps its raw JSON record; the other tensor converts."""
    original = _TypedPluginFiles.add_tensor
    monkeypatch.setattr(
        _TypedPluginFiles,
        "add_tensor",
        lambda files, entry, digits: (
            original(files, entry, digits)
            if entry.tag != "bad"
            else _raise(RuntimeError("tensor bug"))
        ),
    )
    files = list(_files(_log(tmp_path, _two_tensors())))

    assert "/tensors/good/000001.npy" in files
    assert "/tensors/bad/000001.npy" not in files


def test_a_mesh_that_fails_to_build_hides_only_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _MeshFiles.files
    monkeypatch.setattr(
        _MeshFiles,
        "files",
        lambda step, group: (
            original(step, group)
            if group.entries[0].tag != "bad_VERTEX"
            else _raise(RuntimeError("mesh bug"))
        ),
    )
    event = compat_pb2.Event(step=1, wall_time=3.0)
    for name in ("bad", "good"):
        _mesh_value(event, 1, np.zeros((1, 3, 3)), "_VERTEX")
        event.summary.value[-1].tag = f"{name}_VERTEX"
    files = list(_files(_log(tmp_path, event)))

    assert "/meshes/good/000001.obj" in files
    assert not [path for path in files if path.startswith("/meshes/bad/")]


def test_a_histogram_that_fails_to_build_hides_only_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _HistogramFiles.add_step_table
    monkeypatch.setattr(
        _HistogramFiles,
        "add_step_table",
        lambda files, item, digits: (
            original(files, item, digits)
            if item["tag"] != "bad"
            else _raise(RuntimeError("histogram bug"))
        ),
    )
    event = compat_pb2.Event(step=1, wall_time=3.0)
    for tag in ("bad", "good"):
        histogram = event.summary.value.add(tag=tag).histo
        histogram.bucket_limit.append(1.0)
        histogram.bucket.append(1.0)
    files = list(_files(_log(tmp_path, event)))

    assert "/histograms/good/000001.json" in files
    assert not [path for path in files if "/bad/" in path]


def _raise(error: Exception) -> Any:
    raise error
