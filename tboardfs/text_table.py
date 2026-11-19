"""Render text summary tensors as TensorBoard shows them."""

from typing import Any

import numpy as np

from tboardfs.array_json import array_to_json
from tboardfs.model import JsonEntry
from tboardfs.tensor import _TensorDType

WARNING = (
    "**Warning:** this text summary has {rank} dimensions, but only 2-D tables "
    "are supported. This table shows the [0, ..., 0, :, :] slice; the .json file "
    "has every value.\n\n"
)


class _TextTable:
    """Turn a string array into the text of one virtual file.

    One string is the text itself. Many strings form a Markdown table: a
    vector is one column, a matrix keeps its rows and columns, and a higher
    rank shows its first 2-D slice after a warning, as in TensorBoard.
    """

    @staticmethod
    def nodes(entry: JsonEntry) -> list[tuple[str, dict[str, Any]]]:
        """Return (file suffix, node) pairs for one text entry.

        The list is empty when the entry has no strings, or when they do not
        fill the tensor shape. A table also gets a JSON file that keeps the
        shape and the decoded text, not the original bytes.
        """
        tensor = entry.payload.get("tensor") or {}
        if not tensor.get("string_val"):
            return []
        cells = _TensorDType.shaped(_TensorDType.strings(tensor), tensor, fill=True)
        if cells is None:
            return []
        text = _TextTable.markdown(cells).encode()
        nodes = [("txt", {"type": "bytes", "data": text, "mtime": entry.wall_time})]
        if cells.size > 1:
            data = array_to_json(cells)
            nodes.append(
                ("json", {"type": "json", "data": data, "mtime": entry.wall_time})
            )
        return nodes

    @staticmethod
    def markdown(cells: np.ndarray) -> str:
        """Return one string as is, or many strings as a Markdown table."""
        if cells.size == 1:
            return str(cells.reshape(-1)[0])
        warning = ""
        if cells.ndim > 2:
            warning = WARNING.format(rank=cells.ndim)
            cells = cells[(0,) * (cells.ndim - 2)]
        rows = cells.reshape(-1, 1) if cells.ndim == 1 else cells
        columns = rows.shape[1]
        lines = [
            _TextTable.row([" "] * columns),
            _TextTable.row(["---"] * columns),
            *(_TextTable.row([_TextTable.cell(item) for item in row]) for row in rows),
        ]
        return warning + "".join(lines)

    @staticmethod
    def row(cells: list[str]) -> str:
        """Return one Markdown table row."""
        return "| " + " | ".join(cells) + " |\n"

    @staticmethod
    def cell(text: object) -> str:
        """Escape a string so that it stays inside one table cell."""
        return (
            str(text).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
        )
