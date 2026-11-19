"""Proto3 default values read as TensorBoard reads them.

test-logs/proto3 holds events that TensorBoard's proto3 classes wrote, so
every field with its default value is absent; tests/golden/proto3 holds what
TensorBoard 2.21 reads from them (scripts/generate_proto3_logs.py).
"""

import errno
import io
import json
from pathlib import Path
from typing import Any

from google.protobuf import empty_pb2, unknown_fields
import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.tfrecord import iter_tfrecords

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "test-logs" / "proto3"
GOLDEN = ROOT / "tests" / "golden" / "proto3"


def _cases() -> list[Any]:
    cases = []
    for golden in sorted(GOLDEN.glob("*.json")):
        view = json.loads(golden.read_text())
        for kind in ("scalars", "histograms", "tensors"):
            for tag, expected in view[kind].items():
                key = (golden.stem, kind, tag)
                cases.append(pytest.param(*key, expected, id="/".join(key)))
    return cases


def _json(fs: TensorBoardFS, path: str) -> Any:
    return json.loads(fs.read(path, 1 << 20, 0))


def _fields(payload: bytes, number: int) -> list[Any]:
    """Return the wire data of each field with this number, without a schema."""
    message = empty_pb2.Empty.FromString(payload)
    fields = unknown_fields.UnknownFieldSet(message)
    return [field.data for field in fields if field.field_number == number]


def _events(run: str) -> list[bytes]:
    """Return the event payloads of a run, after the file version event."""
    (path,) = (LOGS / run).glob("events.out.tfevents.*")
    with path.open("rb") as handle:
        return [record.payload for record in iter_tfrecords(handle)][1:]


def _values(event: bytes) -> dict[str, bytes]:
    """Return the summary values of an event by tag (Event.summary, Summary.value)."""
    (summary,) = _fields(event, 5)
    return {_fields(value, 1)[0].decode(): value for value in _fields(summary, 1)}


def test_fixtures_leave_out_the_default_fields() -> None:
    """The reference holds only if the writer left the zero values out.

    The numbers are the upstream field numbers: Event.step 2 and wall_time 1,
    Value.tag 1, Value.histo 5 with min 1 and max 2, Value.tensor 8 with tensor_shape 2,
    dim 2 and size 1, and Value.metadata 9 with plugin_data 1 and content 2.
    """
    step_zero = _events("step_zero")
    assert [_fields(event, 2) for event in step_zero] == [[], [1]]
    assert [_fields(event, 1) for event in _events("wall_time_zero")] == [[], []]
    (histograms,) = _events("histogram_zero_edges")
    edges = {
        tag: (_fields(histo, 1), _fields(histo, 2))
        for tag, value in _values(histograms).items()
        for histo in _fields(value, 5)
    }
    assert {tag: (bool(low), bool(high)) for tag, (low, high) in edges.items()} == {
        "zero_min": (False, True),
        "zero_max": (True, False),
        "all_zero": (False, False),
    }
    (tensors,) = _events("tensor_zero_dims")
    (empty_rows,) = _fields(_values(tensors)["empty_rows"], 8)
    (shape,) = _fields(empty_rows, 2)
    assert [_fields(dim, 1) for dim in _fields(shape, 2)] == [[3], []]
    for event in _events("empty_plugin_content"):
        (metadata,) = _fields(_values(event)["note"], 9)
        (plugin_data,) = _fields(metadata, 1)
        assert _fields(plugin_data, 2) == []
    for event in _events("empty_tag"):
        (summary,) = _fields(event, 5)
        assert [_fields(value, 1) for value in _fields(summary, 1)] == [[]]


@pytest.mark.parametrize(("run", "kind", "tag", "expected"), _cases())
def test_proto3_defaults_read_as_tensorboard(
    run: str, kind: str, tag: str, expected: Any
) -> None:
    """Each scalar, histogram and tensor has TensorBoard's values.

    A tensor with one number is a scalar in tboardfs, a string tensor of the
    text plugin is a text file, and a value without a tag is "~empty".
    """
    fs = TensorBoardFS(LOGS / run)
    tag = tag or "~empty"

    if kind == "scalars":
        rows = _json(fs, f"/scalars/{tag}.json")
        assert [[r["step"], r["wall_time"], r["value"]] for r in rows] == expected
        return
    for step, golden in expected.items():
        name = f"{int(step):06d}"
        if kind == "histograms" and not golden:
            # A histogram of no values has no step file.
            with pytest.raises(OSError) as raised:
                fs.getattr(f"/histograms/{tag}/{name}.json")
            assert raised.value.errno == errno.ENOENT
        elif kind == "histograms":
            rows = _json(fs, f"/histograms/{tag}/{name}.json")
            got = [[r["bucket_left"], r["bucket_right"], r["count"]] for r in rows]
            assert got == golden
        elif golden["dtype"] == "object":
            (text,) = golden["values"]
            assert fs.read(f"/text/{tag}/{name}.txt", 1 << 20, 0).decode() == text
        elif not golden["shape"]:
            (value,) = golden["values"]
            rows = _json(fs, f"/scalars/{tag}.json")
            assert [[r["step"], r["value"]] for r in rows] == [[int(step), value]]
        else:
            data = fs.read(f"/tensors/{tag}/{name}.npy", 1 << 20, 0)
            array = np.load(io.BytesIO(data))
            assert str(array.dtype) == golden["dtype"]
            assert list(array.shape) == golden["shape"]
            assert array.ravel().tolist() == golden["values"]
