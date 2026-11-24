"""Per-step files preserve the generic export bytes and NumPy archive contents."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from test_incremental_parse import _frame
from test_restart_purge import _start, _version

from tboardfs import TensorBoardFS
from tboardfs._step_npz import StepArchives
from tboardfs._step_text import step_text
from tboardfs.histogram_compress import NORMAL_HISTOGRAM_BPS, compress_histogram
from tboardfs.proto import compat_pb2
from tboardfs.step_export import StepColumns
from tboardfs.tables import export_table

_FORMATS = ("json", "tsv", "csv", "npz")
_STEP = 2**53 + 1


def _metadata(
    step: int, epoch: float, wall_times: list[float]
) -> dict[str, np.ndarray]:
    return {
        "step": np.full(len(wall_times), step, dtype=np.int64),
        "epoch": np.full(len(wall_times), epoch, dtype=np.float64),
        "wall_time": np.array(wall_times, dtype=np.float64),
        "relative_time": np.array(wall_times, dtype=np.float64) + 0.0,
    }


def _assert_npz(data: bytes, columns: dict[str, np.ndarray]) -> None:
    expected = BytesIO()
    np.savez(expected, **columns)
    assert data == expected.getvalue()
    with ZipFile(BytesIO(data)) as archive:
        assert archive.testzip() is None
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        assert archive.files == list(columns)
        for name, column in columns.items():
            result = archive[name]
            assert result.dtype == column.dtype
            assert result.shape == column.shape
            assert result.tobytes() == column.tobytes()


def _histogram(step: int, wall: float, right: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=wall)
    histo = event.summary.value.add(tag="weights").histo
    histo.min = 0.0
    histo.max = right
    histo.bucket_limit.append(right)
    histo.bucket.append(1.0)
    return _frame(event)


def _epoch(step: int, value: float) -> bytes:
    event = compat_pb2.Event(step=step, wall_time=3.0)
    event.summary.value.add(tag="epoch", simple_value=value)
    return _frame(event)


def test_projected_step_files_match_plain_exports_across_restart(
    tmp_path: Path,
) -> None:
    """Adjacent large steps and each observation retain their own metadata."""
    first, before, restarted = _STEP, _STEP + 1, _STEP + 2
    (tmp_path / "events.out.tfevents").write_bytes(
        _version()
        + _histogram(first, -0.0, 1.0)
        + _histogram(before, 1.0, 2.0)
        + _epoch(before, -0.0)
        + _histogram(restarted, 4.0, 99.0)
        + _epoch(restarted, 99.0)
        + _start(restarted)
        + _histogram(restarted, -0.0, 4.0)
        + _histogram(restarted, 0.0, 8.0)
        + _histogram(restarted, 2.0, 16.0)
        + _epoch(restarted, 7.0)
    )
    fs = TensorBoardFS(tmp_path, histogram_formats=_FORMATS, step_digits=1)
    for step, epoch, walls, rights in (
        (first, np.nan, [-0.0], [1.0]),
        (before, -0.0, [1.0], [2.0]),
        (restarted, 7.0, [-0.0, 0.0, 2.0], [4.0, 8.0, 16.0]),
    ):
        metadata = _metadata(step, epoch, walls)
        histogram = {
            **metadata,
            "bucket_left": np.zeros(len(walls)),
            "bucket_right": np.array(rights),
            "count": np.ones(len(walls)),
        }
        width = len(NORMAL_HISTOGRAM_BPS)
        distribution = {
            **{name: values.repeat(width) for name, values in metadata.items()},
            "percentile": np.tile(np.array(NORMAL_HISTOGRAM_BPS) / 100, len(walls)),
            "value": np.array(
                [
                    value
                    for right in rights
                    for value in compress_histogram(np.array([[0.0, right, 1.0]]))
                ]
            ),
        }
        for tab, columns in (
            ("histograms", histogram),
            ("distributions", distribution),
        ):
            for fmt in _FORMATS:
                name = f"/{tab}/weights/{step}.{fmt}"
                actual = fs.read(name, 1 << 20, 0)
                assert actual == export_table(columns, fmt), name
                if fmt == "npz":
                    _assert_npz(actual, columns)


def test_small_step_serializers_emit_compatible_bytes() -> None:
    """Supported columns take the native path, including cached archive layouts."""
    archives = StepArchives()
    for step, count in ((_STEP, np.inf), (_STEP + 1, 2.0)):
        columns = {
            **_metadata(step, -0.0, [-0.0] * 3),
            "bucket_left": np.array([-np.inf, -0.0, 5e-324]),
            "bucket_right": np.array([-0.0, 5e-324, np.inf]),
            "count": np.array([count, np.nan, 1.0]),
        }
        for fmt in ("json", "tsv", "csv"):
            assert step_text(columns, fmt) == export_table(columns, fmt)
        actual = archives.encode(columns)
        assert actual is not None
        _assert_npz(actual, columns)


@pytest.mark.parametrize(
    "case", ["empty", "large", "strided", "matrix", "float32", "mixed"]
)
def test_unsupported_step_tables_match_plain_exports(case: str) -> None:
    """Fallback retains columns, array layouts and values outside fixed schemas."""
    rows = {"empty": 0, "large": 1024}.get(case, 3)
    columns = {
        **_metadata(_STEP, np.nan, [1.0] * rows),
        "bucket_left": np.arange(rows, dtype=np.float64),
        "bucket_right": np.arange(1, rows + 1, dtype=np.float64),
        "count": np.ones(rows),
    }
    if case == "strided":
        columns["count"] = np.arange(2 * rows, dtype=np.float64)[::2]
    elif case == "matrix":
        columns["count"] = np.arange(2 * rows, dtype=np.float64).reshape(rows, 2)
    elif case == "float32":
        columns["count"] = np.array([-0.0, 0.1, np.nan], dtype=np.float32)
    elif case == "mixed":
        columns["note"] = np.array(["a\tb", 'say "x"', "c,d"])
    projected = StepColumns(columns, StepArchives())
    formats = ("json", "npz") if case == "matrix" else _FORMATS
    for fmt in formats:
        actual = projected.render(fmt)
        assert actual == export_table(columns, fmt)
        if fmt == "npz":
            _assert_npz(actual, columns)
