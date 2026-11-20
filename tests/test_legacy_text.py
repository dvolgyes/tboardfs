"""TF 1.2 uses node_name for both tensor series, but registers only one for Text."""

import json
from pathlib import Path

import pytest
from test_incremental_parse import _frame

from tboardfs import TensorBoardFS
from tboardfs.file_tree import SingleEventTree
from tboardfs.proto import compat_pb2
from tboardfs.tfrecord import iter_tfrecords

RUN = Path(__file__).parent / "data/producer-cases/legacy-text"
EVENT = RUN / "events.out.tfevents.legacy-text"
TABLES = {
    "registered_text": [["legacy", "registered"], ["café", ""]],
    "unregistered_text": [["legacy", "ordinary"], ["x|y", "two\nlines"]],
}


def test_native_registry_and_event() -> None:
    registry = RUN / "plugins/tensorboard_text/tensors.json"
    assert json.loads(registry.read_bytes()) == ["registered_text"]
    with EVENT.open("rb") as handle:
        events = [
            compat_pb2.Event.FromString(r.payload) for r in iter_tfrecords(handle)
        ]
    (summary,) = (e for e in events if e.HasField("summary"))
    assert summary.step == 4
    assert [v.node_name for v in summary.summary.value] == list(TABLES)
    for v in summary.summary.value:
        assert not v.tag and not v.HasField("metadata")
        assert v.WhichOneof("value") == "tensor" and v.tensor.dtype == 7
        assert [d.size for d in v.tensor.tensor_shape.dim] == [2, 2]
        assert list(v.tensor.string_val) == [
            cell.encode() for row in TABLES[v.node_name] for cell in row
        ]


@pytest.mark.parametrize("source", [RUN, EVENT], ids=["directory", "standalone"])
def test_distinct_text_exports_and_exact_raw(source: Path) -> None:
    tree = SingleEventTree(source)
    for tag, cells in TABLES.items():
        base = f"/text/{tag}/000004"
        table = json.loads(tree.read_file(base + ".json"))
        assert table["shape"] == [2, 2] and table["values"] == cells
        markdown = tree.read_file(base + ".txt").decode()
        assert f"| legacy | {cells[0][1]} |" in markdown
    with EVENT.open("rb") as handle:
        for record in iter_tfrecords(handle):
            path = f"/raw/{EVENT.name}/{record.record_offset:020d}.tfrecord"
            assert (
                tree.read_file(path) == record.header + record.payload + record.footer
            )


def test_tag_precedes_node_name_and_unnamed_tensor_stays_unnamed(
    tmp_path: Path,
) -> None:
    event = compat_pb2.Event(step=4)
    for tag, name, cell in [("chosen", "ignored", b"explicit"), ("", "", b"unnamed")]:
        tensor = event.summary.value.add(tag=tag, node_name=name).tensor
        tensor.dtype = 7
        tensor.string_val.append(cell)
    (tmp_path / EVENT.name).write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)
    assert fs.read("/text/chosen/000004.txt", 64, 0) == b"explicit"
    assert fs.read("/text/~empty/000004.txt", 64, 0) == b"unnamed"
