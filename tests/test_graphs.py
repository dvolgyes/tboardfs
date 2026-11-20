"""Every graph of a run is readable; graph.pb is the newest, as in TensorBoard."""

from pathlib import Path

from test_incremental_parse import _files, _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2


def _graphs(tmp_path: Path, *graphs: bytes) -> TensorBoardFS:
    records = b"".join(
        _frame(compat_pb2.Event(wall_time=1.0 + i, graph_def=graph))
        for i, graph in enumerate(graphs)
    )
    (tmp_path / "events.out.tfevents.host").write_bytes(records)
    return TensorBoardFS(tmp_path)


def test_one_graph_is_graph_pb(tmp_path: Path) -> None:
    """A run with one graph has only graph.pb."""
    fs = _graphs(tmp_path, b"only")

    assert [p for p in _files(fs) if p.startswith("/graphs/")] == ["/graphs/graph.pb"]
    assert fs.read("/graphs/graph.pb", 100, 0) == b"only"


def test_several_graphs_are_numbered(tmp_path: Path) -> None:
    """Each graph has a numbered file in write order; graph.pb is the last."""
    fs = _graphs(tmp_path, b"first", b"second", b"third")

    assert fs.read("/graphs/graph.pb", 100, 0) == b"third"
    assert [fs.read(f"/graphs/graph_{i}.pb", 100, 0) for i in range(3)] == [
        b"first",
        b"second",
        b"third",
    ]
