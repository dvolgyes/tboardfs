"""Text summaries with many strings keep their table layout, as in TensorBoard."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from tboardfs import TensorBoardFS
from tboardfs.constants import MAX_REPEATED_TENSOR_BYTES
from tboardfs.proto import compat_pb2

from test_incremental_parse import _files, _frame

DT_STRING = 7


def _add_tensor(
    event: compat_pb2.Event,
    shape: list[int],
    *cells: str | bytes,
    tag: str = "notes",
    plugin: str = "text",
) -> None:
    value = event.summary.value.add(tag=tag)
    value.metadata.plugin_data.plugin_name = plugin
    value.tensor.dtype = DT_STRING
    value.tensor.tensor_shape.dim.extend(
        compat_pb2.TensorShapeProto.Dim(size=s) for s in shape
    )
    value.tensor.string_val.extend(
        cell.encode() if isinstance(cell, str) else cell for cell in cells
    )


def _text(
    tmp_path: Path, shape: list[int], *cells: str | bytes, plugin: str = "text"
) -> TensorBoardFS:
    event = compat_pb2.Event(wall_time=1.0, step=4)
    _add_tensor(event, shape, *cells, plugin=plugin)
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    return TensorBoardFS(tmp_path)


def _read(fs: TensorBoardFS, path: str) -> str:
    return fs.read(path, 1 << 20, 0).decode()


@pytest.mark.parametrize("shape", [[], [1], [1, 1]])
def test_one_string_is_plain_text(tmp_path: Path, shape: list[int]) -> None:
    """A single string, such as PyTorch add_text writes, is the file itself."""
    fs = _text(tmp_path, shape, "# Title\nbody")

    assert _read(fs, "/text/notes/000004.txt") == "# Title\nbody"
    assert "/text/notes/000004.json" not in _files(fs)


def test_a_vector_is_a_one_column_table(tmp_path: Path) -> None:
    """Each string of a vector is one row; a cell keeps its line breaks."""
    fs = _text(tmp_path, [3], "a", "b|c", "two\nlines")

    assert _read(fs, "/text/notes/000004.txt") == (
        "|   |\n| --- |\n| a |\n| b\\|c |\n| two<br>lines |\n"
    )


def test_a_matrix_is_a_table(tmp_path: Path) -> None:
    """Rows and columns follow the tensor."""
    fs = _text(tmp_path, [2, 3], "a", "b", "c", "d", "e", "f")

    assert _read(fs, "/text/notes/000004.txt") == (
        "|   |   |   |\n| --- | --- | --- |\n| a | b | c |\n| d | e | f |\n"
    )


def test_a_higher_rank_shows_the_first_slice_with_a_warning(tmp_path: Path) -> None:
    """Like TensorBoard, only the [0, ..., :, :] slice is a table."""
    fs = _text(tmp_path, [2, 1, 2], "a", "b", "c", "d")
    text = _read(fs, "/text/notes/000004.txt")

    assert text.startswith("**Warning:** this text summary has 3 dimensions")
    assert text.endswith("|   |   |\n| --- | --- |\n| a | b |\n")


def test_a_table_has_a_json_file_with_every_value(tmp_path: Path) -> None:
    """The JSON file keeps the shape and the exact strings."""
    fs = _text(tmp_path, [2, 1, 2], "a", "b|c", "x\ny", "d")
    data = json.loads(_read(fs, "/text/notes/000004.json"))

    assert data["shape"] == [2, 1, 2]
    assert data["values"] == [[["a", "b|c"]], [["x\ny", "d"]]]


def test_one_string_fills_a_larger_shape(tmp_path: Path) -> None:
    """TensorBoard repeats a single string to the shape size."""
    fs = _text(tmp_path, [3], "a")

    assert (
        _read(fs, "/text/notes/000004.txt") == "|   |\n| --- |\n| a |\n| a |\n| a |\n"
    )


def test_strings_that_do_not_fill_the_shape_have_no_text_file(tmp_path: Path) -> None:
    """TensorBoard cannot shape these strings, so only the raw record is left."""
    fs = _text(tmp_path, [3], "a", "b")
    record = json.loads(_read(fs, "/text/notes/000004.json"))

    assert not [path for path in _files(fs) if path.endswith(".txt")]
    assert record["tensor"]["string_val"] == ["a", "b"]


def test_text_previews_preserve_nuls_and_replace_invalid_utf8(tmp_path: Path) -> None:
    fs = _text(tmp_path, [3], b"ok\0", b"\0", b"\xff\0")

    assert _read(fs, "/text/notes/000004.txt") == (
        "|   |\n| --- |\n| ok\0 |\n| \0 |\n| �\0 |\n"
    )
    assert json.loads(_read(fs, "/text/notes/000004.json")) == {
        "shape": [3],
        "dtype": str(np.dtype("U3")),
        "values": ["ok\0", "\0", "�\0"],
    }


def test_single_text_preserves_a_trailing_nul(tmp_path: Path) -> None:
    fs = _text(tmp_path, [], b"ok\0")

    assert _read(fs, "/text/notes/000004.txt") == "ok\0"


def test_empty_leading_dimension_keeps_other_text_readable(tmp_path: Path) -> None:
    event = compat_pb2.Event(wall_time=1.0, step=4)
    _add_tensor(event, [2, 0, 1, 2], b"unused", tag="empty")
    _add_tensor(event, [], b"still here", tag="other")
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)

    assert _read(fs, "/text/empty/000004.txt") == (
        "**Warning:** this text summary has 4 dimensions and its first 2-D slice is empty.\n"
    )
    assert _read(fs, "/text/other/000004.txt") == "still here"


@pytest.mark.parametrize("cell", ["\0", "abc\0"])
def test_string_fill_is_bounded_by_fixed_width_unicode(
    tmp_path: Path, cell: str
) -> None:
    count = MAX_REPEATED_TENSOR_BYTES // (4 * len(cell))
    event = compat_pb2.Event(wall_time=1.0, step=4)
    _add_tensor(event, [count], cell, tag="at_limit")
    _add_tensor(event, [count + 1], cell, tag="over_limit")
    (tmp_path / "events.out.tfevents.host").write_bytes(_frame(event))
    fs = TensorBoardFS(tmp_path)

    assert (
        json.loads(_read(fs, "/text/at_limit/000004.json"))["values"] == [cell] * count
    )
    assert "/text/over_limit/000004.txt" not in _files(fs)


def test_explicit_string_tensor_json_is_exact_and_npy_has_no_pickle(
    tmp_path: Path,
) -> None:
    fs = _text(tmp_path, [1, 3], b"ok\0", b"\0", b"\xff\0z\0", plugin="tensors")
    assert json.loads(_read(fs, "/tensors/notes/000004.json")) == {
        "shape": [1, 3],
        "dtype": str(np.dtype("U4")),
        "values": [["ok\0", "\0", "�\0z\0"]],
    }
    blob = fs.read("/tensors/notes/000004.npy", 1 << 20, 0)
    array = np.load(io.BytesIO(blob), allow_pickle=False)
    assert array.shape == (1, 3)
    assert array.dtype == np.dtype("U4")
    assert array.tolist() == [["ok", "", "�\0z"]]
