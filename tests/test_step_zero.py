"""An event without a step field is step 0, as proto3 writers mean it."""

import json
from pathlib import Path

from tboardfs import TensorBoardFS
from tboardfs.proto import event_pb2

from test_incremental_parse import _files, _frame


def test_an_event_at_step_zero_is_step_zero(tmp_path: Path) -> None:
    """Proto3 writers leave out step 0, so its absence is step 0, not NaN."""
    event = event_pb2.Event(wall_time=1.0)
    event.summary.value.add(tag="loss", simple_value=0.5)
    tensor = event.summary.value.add(tag="w").tensor
    tensor.dtype = 1
    tensor.tensor_shape.dim.add(size=2)
    tensor.float_val.extend([1.0, 2.0])
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)

    rows = json.loads(fs.read("/scalars/loss.json", 1 << 20, 0))

    assert [row["step"] for row in rows] == [0]
    assert "/tensors/w/000000.npy" in _files(fs)
