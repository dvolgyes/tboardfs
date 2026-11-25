"""Read per-run analysis through the mounted and file command views."""

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from test_incremental_parse import PNG, _frame

from tboardfs import TensorBoardFS
from tboardfs.file_cli import main
from tboardfs.proto import compat_pb2

EVENT_NAME = "events.out.tfevents.analysis"
STEP = 2**53 + 1
ODD_TAG = "../metric\twith\ncontrols"
RUN_NAME = "run"


def _event(step: int, *values: tuple[str, str, float | bytes]) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=100.0 + step % 100)
    for tag, kind, value in values:
        summary = event.summary.value.add(tag=tag)
        if kind == "scalar":
            assert isinstance(value, float)
            summary.simple_value = value
        else:
            assert isinstance(value, bytes)
            summary.image.height = 1
            summary.image.width = 1
            summary.image.encoded_image_string = value
    return _frame(event)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / RUN_NAME
    run.mkdir()
    event = run / EVENT_NAME
    event.write_bytes(
        _event(
            STEP,
            (ODD_TAG, "scalar", 1.0),
            ("train/loss", "scalar", 1.0),
        )
        + _event(STEP, ("train/loss", "scalar", 2.0))
        + _event(STEP + 1, ("train/loss", "scalar", 3.0))
        + _event(STEP + 1, ("train/loss", "scalar", 4.0))
        + _event(STEP + 1, ("train/epoch", "scalar", 4.0))
        + _event(STEP + 2, ("other/unknown", "scalar", 9.0))
        + _event(STEP + 2, ("other/epoch", "scalar", 2.0))
        + _event(STEP + 2, ("train/loss", "scalar", 5.0))
        + _event(STEP + 3, ("noepoch/value", "scalar", 7.0))
        + _event(STEP, ("val/loss", "scalar", 10.0))
        + _event(STEP, ("val/epoch", "scalar", 8.0))
        + _event(
            STEP + 1,
            ("train/image/a", "image", PNG),
            ("train/image/b", "image", PNG),
        )
        + _event(STEP, ("val/image", "image", PNG))
    )
    return run


def _read(fs: TensorBoardFS, path: str) -> bytes:
    return fs.read(path, 1 << 20, 0)


def _catalog(fs: TensorBoardFS) -> dict[str, Any]:
    return json.loads(_read(fs, "/catalog.json"))


def _scope_rows(
    fs: TensorBoardFS,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_json = _read(fs, "/epochs.json")
    raw_tsv = _read(fs, "/epochs.tsv")
    tsv_rows = csv.DictReader(StringIO(raw_tsv.decode(), newline=""), delimiter="\t")
    return json.loads(raw_json), list(tsv_rows)


def _get_bytes(source: Path, path: str) -> bytes:
    result = CliRunner().invoke(
        main,
        ["get", str(source), path, "-o", "-", "--scalar-format", "json,tsv", "-E"],
    )
    assert result.exit_code == 0, result.output
    return result.stdout_bytes


def _append(path: Path, frame: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(frame)


def test_catalog_exists_without_epoch_breakdown(run_dir: Path) -> None:
    fs = TensorBoardFS(run_dir, index_workers=1)

    catalog = _catalog(fs)
    assert catalog["version"] == 1
    assert any(item["tag"] == ODD_TAG for item in catalog["tags"])
    assert "epochs.json" not in fs.readdir("/")
    assert "epochs.tsv" not in fs.readdir("/")


def test_catalog_describes_canonical_files_and_safe_original_tags(
    run_dir: Path,
) -> None:
    fs = TensorBoardFS(run_dir, scalar_formats="json,tsv", index_workers=1)
    catalog = _catalog(fs)
    metric = next(item for item in catalog["tags"] if item["tag"] == ODD_TAG)

    assert metric["kind"] == "scalars"
    assert metric["formats"] == ["json", "tsv"]
    assert metric["file_count"] == 2
    assert metric["observations"] == 1
    assert metric["distinct_steps"] == 1
    assert metric["step_min"] == STEP
    assert metric["step_max"] == STEP
    assert metric["observations_without_epoch"] == 1
    assert metric["known_epoch_ranges"] == []
    assert metric["paths"] == {
        fmt: f"scalars/%2E%2E/metric%09with%0Acontrols.{fmt}" for fmt in ("json", "tsv")
    }
    for path in metric["paths"].values():
        assert _read(fs, "/" + path)
        assert ".." not in Path(path).parts
        assert "\t" not in path and "\n" not in path
    assert catalog["files"] == []


def test_catalog_counts_series_once_and_ignores_aliases(run_dir: Path) -> None:
    fs = TensorBoardFS(run_dir, epoch_breakdown=True, raw=True, index_workers=1)
    ordinary = TensorBoardFS(run_dir, index_workers=1)
    catalog = _catalog(fs)
    assert catalog == _catalog(ordinary) | {"reports": catalog["reports"]}

    loss = next(item for item in catalog["tags"] if item["tag"] == "train/loss")
    assert loss["kind"] == "scalars"
    assert loss["observations"] == 5
    assert loss["distinct_steps"] == 3
    assert loss["step_min"] == STEP
    assert loss["step_max"] == STEP + 2
    assert loss["observations_without_epoch"] == 2
    assert loss["known_epoch_ranges"] == [[4, 4]]
    image = next(item for item in catalog["tags"] if item["tag"] == "train/image/a")
    assert image["directories"] == ["images/train/image/a"]
    assert image["distinct_steps"] == 1
    assert image["known_epoch_ranges"] == [[4, 4]]


def test_epoch_tables_match_rows_and_global_column_schema(run_dir: Path) -> None:
    fs = TensorBoardFS(run_dir, epoch_breakdown=True, index_workers=1)
    rows, tsv_rows = _scope_rows(fs)
    assert _catalog(fs)["reports"] == {
        "epochs": {"json": "epochs.json", "tsv": "epochs.tsv"}
    }
    assert tsv_rows == [{key: str(value) for key, value in row.items()} for row in rows]

    columns = [
        "epoch",
        "scope",
        "step_min",
        "step_max",
        "epoch_dir",
        "image_steps",
        "metric_rows",
    ]
    assert [list(row) for row in rows] == [columns] * 3
    assert [tuple(row.values()) for row in rows] == [
        (2, "other", STEP + 2, STEP + 2, "epoch-breakdown/epoch-2/", 0, 1),
        (4, "train", STEP + 1, STEP + 2, "epoch-breakdown/epoch-4/", 1, 3),
        (8, "val", STEP, STEP, "epoch-breakdown/epoch-8/", 1, 1),
    ]


@pytest.mark.parametrize("source_kind", ["directory", "event"])
def test_file_get_matches_mount_for_analysis_files(
    run_dir: Path, source_kind: str
) -> None:
    fs = TensorBoardFS(
        run_dir, scalar_formats="json,tsv", epoch_breakdown=True, index_workers=1
    )
    if source_kind == "directory":
        source = run_dir.parent
        prefix = f"/{RUN_NAME}"
    else:
        source = run_dir / EVENT_NAME
        prefix = ""

    for name in ("catalog.json", "epochs.tsv", "epochs.json"):
        assert _get_bytes(source, f"{prefix}/{name}") == _read(fs, f"/{name}")


def test_epoch_report_files_need_enabled_breakdown_and_epoch_dirs(
    run_dir: Path, tmp_path: Path
) -> None:
    ordinary = TensorBoardFS(run_dir, index_workers=1)
    assert "epochs.json" not in ordinary.readdir("/")
    assert "epochs.tsv" not in ordinary.readdir("/")
    assert _catalog(ordinary)["reports"] == {}

    no_epochs = tmp_path / "no-epochs"
    no_epochs.mkdir()
    (no_epochs / EVENT_NAME).write_bytes(_event(7, ("train/loss", "scalar", 1.0)))
    enabled = TensorBoardFS(no_epochs, epoch_breakdown=True, index_workers=1)
    assert "epoch-breakdown" not in enabled.readdir("/")
    assert "epochs.json" not in enabled.readdir("/")
    assert "epochs.tsv" not in enabled.readdir("/")
    assert _catalog(enabled)["reports"] == {}


def test_marker_only_epoch_reports_core_fields(tmp_path: Path) -> None:
    scope = "train\twith\ncontrols"
    (tmp_path / EVENT_NAME).write_bytes(_event(STEP, (f"{scope}/epoch", "scalar", 4.0)))
    fs = TensorBoardFS(tmp_path, epoch_breakdown=True, index_workers=1)
    rows, tsv = _scope_rows(fs)

    assert rows == [
        {
            "epoch": 4,
            "scope": scope,
            "step_min": STEP,
            "step_max": STEP,
            "epoch_dir": "epoch-breakdown/epoch-4/",
        }
    ]
    assert tsv == [{key: str(value) for key, value in rows[0].items()}]


def test_catalog_refreshes_after_append_without_epoch_breakdown(run_dir: Path) -> None:
    fs = TensorBoardFS(
        run_dir, index_workers=1, refresh_age_seconds=0, source_check_seconds=0
    )
    before = next(item for item in _catalog(fs)["tags"] if item["tag"] == "train/loss")
    event = run_dir / EVENT_NAME
    _append(
        event,
        _event(STEP + 3, ("new/tag", "scalar", 2.0), ("train/loss", "scalar", 7.0)),
    )

    updated = _catalog(fs)
    loss = next(item for item in updated["tags"] if item["tag"] == "train/loss")
    assert loss["observations"] == before["observations"] + 1
    assert loss["step_max"] == STEP + 3
    assert any(item["tag"] == "new/tag" for item in updated["tags"])
    assert "epochs.json" not in fs.readdir("/")


def test_epoch_tables_refresh_after_append_when_enabled(run_dir: Path) -> None:
    fs = TensorBoardFS(
        run_dir,
        epoch_breakdown=True,
        index_workers=1,
        refresh_age_seconds=0,
        source_check_seconds=0,
    )
    before, _ = _scope_rows(fs)
    _append(
        run_dir / EVENT_NAME,
        _event(STEP + 4, ("train/loss", "scalar", 5.0))
        + _event(STEP + 4, ("train/epoch", "scalar", 5.0)),
    )

    after, tsv_rows = _scope_rows(fs)
    assert len(after) > len(before)
    assert len(tsv_rows) == len(after)
    assert any(row["epoch"] == 5 and row["scope"] == "train" for row in after)
