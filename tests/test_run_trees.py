"""A growing run publishes immutable trees equivalent to a fresh mount."""

import json
from pathlib import Path
from typing import Any

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2
from tboardfs.render_node import render_node

from test_hparams import _start as _hparams_start
from test_incremental_parse import _contents, _frame
from test_open_handles import _append
from test_restart_purge import _start, _version

EVENTS = "events.out.tfevents.host"


def _append_records(path: Path, *records: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(b"".join(records))


def _scalars(step: int, wall_time: float, **values: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=wall_time)
    for tag, value in values.items():
        event.summary.value.add(tag=tag, simple_value=value)
    return _frame(event)


def _tensor(step: int, wall_time: float, tag: str, number: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=wall_time)
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = "tensors"
    value.tensor.dtype = compat_pb2.DT_FLOAT
    value.tensor.tensor_shape.dim.add(size=2)
    value.tensor.float_val.extend([number, number])
    return _frame(event)


def _node(tree: dict[str, Any], path: str) -> dict[str, Any]:
    for component in path.strip("/").split("/"):
        tree = tree["children"][component]
    return tree


def _listing(fs: TensorBoardFS, path: str = "/") -> dict[str, list[str]]:
    """Return every directory listing below ``path``."""
    entries = fs.readdir(path)[2:]
    listing = {path: entries}
    for name in entries:
        child = f"{path.rstrip('/')}/{name}"
        if fs.getattr(child)["st_nlink"] == 2:
            listing.update(_listing(fs, child))
    return listing


def _assert_fresh(fs: TensorBoardFS, source: Path) -> None:
    fresh = TensorBoardFS(source, scalar_formats="json")
    directories = _listing(fs)
    assert directories == _listing(fresh)
    assert _contents(fs) == _contents(fresh)
    for path in directories:
        assert fs.getattr(path) == fresh.getattr(path)


def _two_runs(tmp_path: Path) -> list[Path]:
    paths = []
    for name in ("run_a", "run_b"):
        (tmp_path / name).mkdir()
        paths.append(tmp_path / name / EVENTS)
        _append(paths[-1], 0)
    return paths


def test_growing_run_reuses_untouched_typed_subtree(tmp_path: Path) -> None:
    """Repeated appends change only the affected branches of a published tree."""
    run_a, _ = _two_runs(tmp_path)
    stable = "/run_a/scalars/stable"
    loss = "/run_a/scalars/loss.json"
    new_tag = "/run_a/scalars/new_tag.json"
    _append_records(run_a, _scalars(0, 100, **{"stable/metric": 42.0}))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    before = fs.virtual_tree()
    old_branch = _node(before, stable)
    old_loss = _node(before, loss)

    _append(run_a, 1)
    middle = fs.virtual_tree()
    assert _node(middle, stable) is old_branch
    _append_records(run_a, _scalars(2, 102, loss=0.25, new_tag=9.0))
    after = fs.virtual_tree()

    _assert_fresh(fs, tmp_path)
    assert _node(after, stable) is old_branch
    assert _node(after, "/run_b/scalars") is _node(before, "/run_b/scalars")
    assert _node(after, loss) is not old_loss
    assert "new_tag.json" not in _node(before, "/run_a/scalars")["children"]
    assert "new_tag.json" not in _node(middle, "/run_a/scalars")["children"]
    assert [row["value"] for row in json.loads(render_node(old_loss))] == [1.0]
    middle_rows = json.loads(render_node(_node(middle, loss)))
    assert [row["value"] for row in middle_rows] == [1.0, 0.5]
    assert [row["value"] for row in json.loads(fs.read(loss, 4096, 0))] == [
        1.0,
        0.5,
        0.25,
    ]
    assert [row["value"] for row in json.loads(fs.read(new_tag, 4096, 0))] == [9.0]
    metric = f"{stable}/metric.json"
    old_metric = render_node(_node(before, metric))
    assert [row["value"] for row in json.loads(old_metric)] == [42.0]
    assert fs.read(metric, 4096, 0) == old_metric


def test_duplicate_step_can_lower_directory_times(tmp_path: Path) -> None:
    """A later record wins a step path, even when its wall time goes backward."""
    event = tmp_path / EVENTS
    event.write_bytes(_tensor(1, 20, "metric", 1) + _tensor(3, 100, "metric", 3))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    before = fs.virtual_tree()
    winning_path = "/tensors/metric/000003.json"
    assert fs.readdir("/images") == [".", ".."]
    assert fs.getattr("/images")["st_mtime"] == 100

    _append_records(event, _tensor(2, 10, "metric", 2), _tensor(3, 30, "metric", 4))

    _assert_fresh(fs, tmp_path)
    assert json.loads(fs.read(winning_path, 4096, 0))["values"] == [4.0, 4.0]
    assert json.loads(render_node(_node(before, winning_path)))["values"] == [
        3.0,
        3.0,
    ]
    assert fs.getattr("/tensors/metric")["st_mtime"] == 30
    assert fs.getattr("/tensors")["st_mtime"] == 30
    assert fs.getattr("/images")["st_mtime"] == 30
    assert _node(before, "/images")["mtime"] == 100


def test_new_file_directory_collision_renames_tags_across_tabs(tmp_path: Path) -> None:
    """An appended scalar file can rename a tag already published in two tabs."""
    event = tmp_path / EVENTS
    tag = "loss.json/nested"
    event.write_bytes(_scalars(1, 10, **{tag: 1.0}) + _tensor(1, 10, tag, 2))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    before = fs.virtual_tree()
    tensor = f"/tensors/{tag}/000001.json"
    original = fs.read(tensor, 4096, 0)

    _append_records(event, _scalars(2, 20, loss=3.0))

    _assert_fresh(fs, tmp_path)
    assert fs.readdir("/scalars") == [".", "..", "loss.json", "loss.json~2"]
    assert fs.readdir("/tensors") == [".", "..", "loss.json~2"]
    assert fs.read("/tensors/loss.json~2/nested/000001.json", 4096, 0) == original
    assert render_node(_node(before, tensor)) == original


def test_append_to_earlier_file_preserves_duplicate_step_precedence(
    tmp_path: Path,
) -> None:
    """File order, not arrival or wall time, decides a duplicate step's winner."""
    earlier = tmp_path / f"{EVENTS}.0"
    earlier.write_bytes(_tensor(1, 10, "metric", 1))
    (tmp_path / f"{EVENTS}.1").write_bytes(_tensor(1, 20, "metric", 2))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    path = "/tensors/metric/000001.json"
    winner = fs.read(path, 4096, 0)
    assert json.loads(winner)["values"] == [2.0, 2.0]

    _append_records(earlier, _tensor(1, 30, "metric", 3), _tensor(2, 40, "metric", 4))

    _assert_fresh(fs, tmp_path)
    assert fs.read(path, 4096, 0) == winner
    step_two = fs.read("/tensors/metric/000002.json", 4096, 0)
    assert json.loads(step_two)["values"] == [4.0, 4.0]


def test_restart_after_append_removes_previously_published_paths(
    tmp_path: Path,
) -> None:
    """A session restart can remove tags and refresh inherited empty-tab times."""
    event = tmp_path / EVENTS
    event.write_bytes(_version() + _scalars(0, 10, kept=1.0))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    fs.virtual_tree()
    _append_records(event, _scalars(2, 100, removed=2.0))
    before = fs.virtual_tree()
    assert fs.getattr("/images")["st_mtime"] == 100

    _append_records(event, _start(2), _scalars(2, 30, resumed=3.0))

    _assert_fresh(fs, tmp_path)
    assert fs.readdir("/scalars") == [".", "..", "kept.json", "resumed.json"]
    assert fs.getattr("/images")["st_mtime"] == 30
    rows = json.loads(render_node(_node(before, "/scalars/removed.json")))
    assert [row["value"] for row in rows] == [2.0]


def test_epoch_append_updates_previously_published_scalar_rows(tmp_path: Path) -> None:
    """An epoch scalar is an aggregate dependency of other scalar exports."""
    event = tmp_path / EVENTS
    event.write_bytes(_scalars(1, 10, epoch=1.0) + _scalars(2, 20, loss=2.0))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    before = fs.virtual_tree()
    loss = "/scalars/loss.json"
    assert json.loads(fs.read(loss, 4096, 0))[0]["epoch"] == 1.0

    _append_records(event, _scalars(2, 30, epoch=2.0))

    _assert_fresh(fs, tmp_path)
    assert json.loads(fs.read(loss, 4096, 0))[0]["epoch"] == 2.0
    assert json.loads(render_node(_node(before, loss)))[0]["epoch"] == 1.0


def test_scalar_append_refreshes_an_existing_hparams_aggregate(tmp_path: Path) -> None:
    """Metric values change even when no new HParams record was appended."""
    event = tmp_path / EVENTS
    event.write_bytes(_hparams_start(0.25, 0.0) + _scalars(1, 10, loss=1.0))
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0, scalar_formats="json")
    before = fs.virtual_tree()
    path = "/hparams/hparams.json"
    assert json.loads(fs.read(path, 4096, 0))["metrics"] == {"loss": 1.0}

    _append_records(event, _scalars(2, 20, loss=0.5))

    _assert_fresh(fs, tmp_path)
    assert json.loads(fs.read(path, 4096, 0))["metrics"] == {"loss": 0.5}
    assert json.loads(render_node(_node(before, path)))["metrics"] == {"loss": 1.0}


def test_run_versions_are_unique_and_change_with_the_run(tmp_path: Path) -> None:
    """Only a changed run gets a new version; a re-created run gets an unused one."""
    run_a, run_b = _two_runs(tmp_path)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    fs.virtual_tree()
    before = {parts: run.version for parts, run in fs._state.runs.items() if parts}
    seen = set(before.values())

    _append(run_a, 1)
    fs.readdir("/run_a")
    after = {parts: run.version for parts, run in fs._state.runs.items() if parts}
    run_b.unlink()
    fs.readdir("/run_b")
    _append(run_b, 0)
    fs.readdir("/run_b")
    recreated = fs._state.runs[("run_b",)].version

    assert len(seen) == 2
    assert after[("run_a",)] not in seen
    assert after[("run_b",)] == before[("run_b",)]
    assert recreated not in seen | set(after.values())


def test_removed_nested_run_leaves_no_entries_in_its_parent_run(
    tmp_path: Path,
) -> None:
    """A run inside a tab-named directory of another run can go away cleanly."""
    (tmp_path / "a" / "scalars").mkdir(parents=True)
    _append(tmp_path / "a" / EVENTS, 0)
    nested = tmp_path / "a" / "scalars" / EVENTS
    _append(nested, 0)
    fs = TensorBoardFS(tmp_path, refresh_age_seconds=0)
    assert "images" in fs.readdir("/a/scalars~2")

    nested.unlink()

    assert _listing(fs) == _listing(TensorBoardFS(tmp_path))
