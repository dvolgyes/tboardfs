"""PR-curve tables have one labeled row per threshold, as in TensorBoard."""

import io
import json
from pathlib import Path

import numpy as np
from test_incremental_parse import _frame

from tboardfs import TensorBoardFS
from tboardfs.proto import compat_pb2

DT_FLOAT = 1

# Rows as the pr_curve plugin writes them: TP, FP, TN, FN, precision, recall.
PR = np.array(
    [
        [3, 2, 1],
        [3, 1, 0],
        [0, 2, 3],
        [0, 1, 2],
        [0.5, 0.75, 1.0],
        [1.0, 0.5, 0.25],
    ],
    dtype="<f4",
)


def _pr_log(tmp_path: Path, data: np.ndarray) -> TensorBoardFS:
    event = compat_pb2.Event(wall_time=1.0, step=2)
    value = event.summary.value.add(tag="pr")
    value.metadata.plugin_data.plugin_name = "pr_curves"
    value.tensor.dtype = DT_FLOAT
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in data.shape
    )
    value.tensor.tensor_content = data.tobytes()
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _read(fs: TensorBoardFS, name: str) -> bytes:
    return fs.read(f"/pr_curves/pr/000002.{name}", 1 << 20, 0)


def test_each_threshold_is_one_labeled_row(tmp_path: Path) -> None:
    """Thresholds are evenly spaced in [0, 1], one per column of the tensor."""
    rows = json.loads(_read(_pr_log(tmp_path, PR), "json"))

    assert rows == [
        {
            "threshold": 0.0,
            "true_positives": 3.0,
            "false_positives": 3.0,
            "true_negatives": 0.0,
            "false_negatives": 0.0,
            "precision": 0.5,
            "recall": 1.0,
        },
        {
            "threshold": 0.5,
            "true_positives": 2.0,
            "false_positives": 1.0,
            "true_negatives": 2.0,
            "false_negatives": 1.0,
            "precision": 0.75,
            "recall": 0.5,
        },
        {
            "threshold": 1.0,
            "true_positives": 1.0,
            "false_positives": 0.0,
            "true_negatives": 3.0,
            "false_negatives": 2.0,
            "precision": 1.0,
            "recall": 0.25,
        },
    ]


def test_the_tsv_has_the_same_columns(tmp_path: Path) -> None:
    """The TSV header names the columns in the plugin's row order."""
    header = _read(_pr_log(tmp_path, PR), "tsv").decode().splitlines()[0]

    assert header.split("\t") == [
        "threshold",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
    ]


def test_the_npy_keeps_the_raw_tensor(tmp_path: Path) -> None:
    """The .npy file is the tensor as written, without the threshold column."""
    data = np.load(io.BytesIO(_read(_pr_log(tmp_path, PR), "npy")))

    np.testing.assert_array_equal(data, PR)


def test_another_shape_keeps_generic_columns(tmp_path: Path) -> None:
    """A tensor that is not [6, N] is not labeled as a PR curve."""
    rows = json.loads(_read(_pr_log(tmp_path, PR.T.copy()), "json"))

    assert list(rows[0]) == [f"value_{i}" for i in range(6)]


def test_a_vector_of_six_values_keeps_a_generic_column(tmp_path: Path) -> None:
    """Six values without a threshold axis are one generic column."""
    rows = json.loads(_read(_pr_log(tmp_path, PR[:, 0].copy()), "json"))

    assert [list(row) for row in rows] == [["value_0"]] * 6
