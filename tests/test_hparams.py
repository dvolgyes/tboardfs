"""hparams.json follows the TensorBoard hparams plugin for one run."""

import json
from pathlib import Path
from typing import Any

from test_incremental_parse import _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2, hparams_plugin_data_pb2

EXPERIMENT = "_hparams_/experiment"
SESSION_START = "_hparams_/session_start_info"
SESSION_END = "_hparams_/session_end_info"


def _plugin(tag: str, data: Any) -> bytes:
    event = compat_pb2.Event(wall_time=1.0)
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = "hparams"
    value.metadata.plugin_data.content = data.SerializeToString()
    value.tensor.dtype = 1
    return _frame(event)


def _experiment(*metrics: str) -> bytes:
    """Declare metrics; ``group/tag`` declares a metric of a sub-run group."""
    data = hparams_plugin_data_pb2.HParamsPluginData()
    data.experiment.hparam_infos.add(name="lr")
    for metric in metrics:
        name = data.experiment.metric_infos.add().name
        name.group, _, name.tag = metric.rpartition("/")
    return _plugin(EXPERIMENT, data)


def _start(lr: float, time: float) -> bytes:
    data = hparams_plugin_data_pb2.HParamsPluginData()
    data.session_start_info.hparams["lr"].number_value = lr
    data.session_start_info.start_time_secs = time
    return _plugin(SESSION_START, data)


def _end(status: int, time: float) -> bytes:
    data = hparams_plugin_data_pb2.HParamsPluginData()
    data.session_end_info.status = status
    data.session_end_info.end_time_secs = time
    return _plugin(SESSION_END, data)


def _scalar(tag: str, step: int, value: float) -> bytes:
    event = compat_pb2.Event(wall_time=2.0 + step, step=step)
    event.summary.value.add(tag=tag, simple_value=value)
    return _frame(event)


def _hparams(tmp_path: Path, *records: bytes) -> dict[str, Any]:
    (tmp_path / "events.out.tfevents.host").write_bytes(b"".join(records))
    fs = TensorBoardFS(tmp_path)
    data: dict[str, Any] = json.loads(fs.read("/hparams/hparams.json", 1 << 20, 0))
    return data


def test_declared_metrics_are_the_metrics(tmp_path: Path) -> None:
    """Declared metric tags give the last value; a missing one is null."""
    data = _hparams(
        tmp_path,
        _experiment("accuracy", "never_logged"),
        _start(0.1, 5.0),
        _scalar("accuracy", 0, 0.5),
        _scalar("accuracy", 1, 0.875),
        _scalar("hparam/loss", 1, 0.2),
    )

    assert data["metrics"] == {"accuracy": 0.875, "never_logged": None}


def test_without_an_experiment_every_scalar_is_a_metric(tmp_path: Path) -> None:
    """TensorBoard uses all scalar tags of the run when nothing is declared."""
    data = _hparams(
        tmp_path, _start(0.1, 5.0), _scalar("loss", 3, 0.25), _scalar("acc", 3, 0.75)
    )

    assert data["metrics"] == {"acc": 0.75, "loss": 0.25}


def test_every_session_is_listed(tmp_path: Path) -> None:
    """The top-level session is the first, as in TensorBoard; none is lost."""
    data = _hparams(
        tmp_path,
        _start(0.1, 5.0),
        _end(1, 6.0),
        _start(0.2, 7.0),
        _end(2, 8.0),
    )

    assert data["session_start"]["hparams"] == {"lr": 0.1}
    assert data["session_end"]["status"] == "STATUS_SUCCESS"
    assert [
        (s["session_start"]["hparams"]["lr"], s["session_end"]["status"])
        for s in data["sessions"]
    ] == [(0.1, "STATUS_SUCCESS"), (0.2, "STATUS_FAILURE")]


def test_an_unfinished_session_has_no_end(tmp_path: Path) -> None:
    """A session that is still running has an empty end record."""
    data = _hparams(tmp_path, _start(0.1, 5.0), _end(1, 6.0), _start(0.2, 7.0))

    assert data["sessions"][1]["session_end"] == {}


def test_proto3_default_values_are_kept(tmp_path: Path) -> None:
    """Zero and unset enum values are written instead of left out."""
    data = _hparams(tmp_path, _experiment(), _start(0.0, 0.0), _end(0, 0.0))

    assert data["session_end"] == {"status": "STATUS_UNKNOWN", "end_time_secs": 0.0}
    assert data["session_start"]["hparams"] == {"lr": 0.0}
    assert data["experiment"]["hparam_infos"][0]["type"] == "DATA_TYPE_UNSET"


def test_metrics_of_a_sub_run_group_have_no_value_here(tmp_path: Path) -> None:
    """A metric in group "validation" lives in another run, not in this one."""
    data = _hparams(
        tmp_path, _experiment("loss", "validation/acc"), _scalar("loss", 1, 0.5)
    )

    assert data["metrics"] == {"loss": 0.5}


def test_an_end_without_an_open_session_starts_a_new_one(tmp_path: Path) -> None:
    """A second end record does not replace the end of a finished session."""
    data = _hparams(tmp_path, _start(0.1, 5.0), _end(1, 6.0), _end(2, 8.0))

    assert [s["session_end"]["status"] for s in data["sessions"]] == [
        "STATUS_SUCCESS",
        "STATUS_FAILURE",
    ]
    assert data["sessions"][1]["session_start"] == {}


def test_the_experiment_may_follow_the_session(tmp_path: Path) -> None:
    """The experiment record is found wherever it is in the run."""
    data = _hparams(tmp_path, _start(0.1, 5.0), _experiment("loss"))

    assert data["experiment"]["metric_infos"][0]["name"]["tag"] == "loss"
