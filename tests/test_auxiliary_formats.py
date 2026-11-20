"""Checkpoint tensors and DebugEvents are not training Summary records."""

import json
from pathlib import Path

from test_incremental_parse import _files

from tboardfs import TensorBoardFS

RUN = Path(__file__).parent / "data/producer-cases/auxiliary"


def test_checkpoint_projector_retains_config_without_synthesizing_tensors() -> None:
    assert (RUN / "weights.index").stat().st_size > 0
    assert (RUN / "weights.data-00000-of-00001").stat().st_size == 16
    fs = TensorBoardFS(RUN)
    assert list(_files(fs, "/projector")) == ["/projector/projector_config.pbtxt"]
    config = (RUN / "projector_config.pbtxt").read_bytes()
    assert (
        config
        == b'model_checkpoint_path: "weights"\nembeddings {\n  tensor_name: "vectors"\n}\n'
    )
    assert fs.read("/projector/projector_config.pbtxt", 1024, 0) == config


def test_debugger_streams_are_not_discovered_as_training_events() -> None:
    streams = list((RUN / "debug").glob("tfdbg_events.*"))
    assert {p.suffix for p in streams} == {
        ".metadata",
        ".source_files",
        ".stack_frames",
        ".graphs",
        ".execution",
        ".graph_execution_traces",
    }
    assert all(p.stat().st_size > 0 for p in streams)
    fs = TensorBoardFS(RUN)
    rows = json.loads(fs.read("/scalars/control.json", 4096, 0))
    assert [(r["step"], r["value"]) for r in rows] == [(3, 1.25)]
    event = RUN / "events.out.tfevents.auxiliary"
    assert fs.readdir("/raw")[2:] == [event.name]
    frames = sorted(_files(fs, "/raw"))
    assert b"".join(fs.read(p, 4096, 0) for p in frames) == event.read_bytes()
    assert all(p.startswith(("/scalars/", "/raw/", "/projector/")) for p in _files(fs))
