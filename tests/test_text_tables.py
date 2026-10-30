"""Text summaries with many strings keep their table layout, as in TensorBoard."""

import json
from pathlib import Path

import pytest

from tboardfs import TensorBoardFS
from tboardfs.proto import event_pb2

from test_incremental_parse import _files, _frame

DT_STRING = 7


def _text(tmp_path: Path, shape: list[int], *cells: str) -> TensorBoardFS:
    event = event_pb2.Event(wall_time=1.0, step=4)
    value = event.summary.value.add(tag="notes")
    value.metadata.plugin_data.plugin_name = "text"
    value.tensor.dtype = DT_STRING
    value.tensor.tensor_shape.dim.extend(
        event_pb2.TensorShapeDim(size=s) for s in shape
    )
    value.tensor.string_val.extend(cell.encode() for cell in cells)
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
