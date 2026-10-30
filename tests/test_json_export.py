"""Every virtual JSON file is strict JSON and keeps non-finite values and bytes."""

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.tables import _TableExport

from test_incremental_parse import _files, _frame

DT_FLOAT = 1
CONTENT = b"\x08\x96\x01\xff\xfe"


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON token {name}")


def _strict(data: bytes) -> Any:
    return json.loads(data, parse_constant=_refuse_constant)


@pytest.fixture
def fs(tmp_path: Path) -> TensorBoardFS:
    """Return a log with non-finite scalars and tensors and binary plugin data."""
    records = []
    for step, loss in enumerate([1.0, math.inf, -math.inf, math.nan]):
        event = compat_pb2.Event(wall_time=100.0 + step, step=step)
        event.summary.value.add(tag="loss", simple_value=loss)
        records.append(_frame(event))
    event = compat_pb2.Event(wall_time=200.0, step=7)
    value = event.summary.value.add(tag="act")
    value.metadata.plugin_data.plugin_name = "tensors"
    value.tensor.dtype = DT_FLOAT
    value.tensor.tensor_shape.dim.add(size=4)
    value.tensor.float_val.extend([math.nan, math.inf, -math.inf, 2.0])
    custom = event.summary.value.add(tag="custom")
    custom.metadata.plugin_data.plugin_name = "my_plugin"
    custom.metadata.plugin_data.content = CONTENT
    custom.tensor.dtype = DT_FLOAT
    custom.tensor.float_val.append(1.0)
    records.append(_frame(event))
    (tmp_path / "events.out.tfevents.host").write_bytes(b"".join(records))
    return TensorBoardFS(tmp_path)


def test_every_json_file_is_strict_json(fs: TensorBoardFS) -> None:
    """No file holds the NaN or Infinity tokens that RFC 8259 forbids."""
    paths = [path for path in _files(fs) if path.endswith(".json")]

    assert len(paths) >= 3
    for path in paths:
        _strict(fs.read(path, 1 << 20, 0))


def test_scalar_json_keeps_infinities_apart_from_nan(fs: TensorBoardFS) -> None:
    """A diverged loss reads as Infinity; NaN stays null as a missing value."""
    rows = _strict(fs.read("/scalars/loss.json", 1 << 20, 0))

    assert [row["value"] for row in rows] == [1.0, "Infinity", "-Infinity", None]


def test_tensor_json_spells_non_finite_values(fs: TensorBoardFS) -> None:
    """Tensor values use the protobuf JSON spelling for non-finite floats."""
    data = _strict(fs.read("/tensors/act/000007.json", 1 << 20, 0))

    assert data["values"] == ["NaN", "Infinity", "-Infinity", 2.0]


def test_binary_plugin_content_round_trips(fs: TensorBoardFS) -> None:
    """Bytes that are not UTF-8 are base64 encoded instead of corrupted."""
    data = _strict(fs.read("/plugins/my_plugin/custom/000007.json", 1 << 20, 0))

    assert base64.b64decode(data["plugin_content"]["base64"]) == CONTENT


def test_numpy_values_in_a_document_are_converted() -> None:
    """Arrays and NumPy scalars inside a document use the same spelling."""
    data = {"array": np.array([np.nan, 1.0]), "scalar": np.float32(-np.inf)}

    assert _strict(_TableExport.json_document(data)) == {
        "array": ["NaN", 1.0],
        "scalar": "-Infinity",
    }


def test_text_in_a_document_is_utf8() -> None:
    """Non-ASCII text is UTF-8, as RFC 8259 asks, not escaped."""
    data = _TableExport.json_document({"name": "épsilon δ"})

    assert "épsilon δ".encode() in data
    assert _strict(data) == {"name": "épsilon δ"}


def test_table_json_numbers_read_back_exactly() -> None:
    """Each number of a table reads back as the value of its column."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=200) * 10.0 ** rng.integers(-320, 300, size=200)
    series = {"step": np.arange(200, dtype=np.int64) * 2**40, "value": values}

    rows = _strict(_TableExport.json(series))

    assert [row["step"] for row in rows] == series["step"].tolist()
    assert [row["value"] for row in rows] == values.tolist()
