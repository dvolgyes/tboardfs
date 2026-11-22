"""Render small per-step numeric tables with repeated observation metadata."""

from itertools import chain

import numpy as np
import orjson

from tboardfs._step_schema import DISTRIBUTION, HISTOGRAM, schema
from tboardfs.tables import _TableExport


def _json_row(names: tuple[str, ...]) -> str:
    """Make a row template from the fixed, unescaped schema names."""
    return "  {\n" + ",\n".join(f'    "{name}": %s' for name in names) + "\n  }"


_HISTOGRAM_JSON = _json_row(HISTOGRAM)
_DISTRIBUTION_JSON = _json_row(DISTRIBUTION)


def step_text(series: dict[str, np.ndarray], fmt: str) -> bytes | None:
    """Format repeated metadata once, preserving the sign and bits of each value."""
    if fmt not in ("json", "tsv", "csv"):
        return None
    names = schema(series)
    if names is None:
        return None
    values = tuple(series.values())
    rows = len(values[0])
    if any(column.tobytes() != column[:1].tobytes() * rows for column in values[:4]):
        return None

    raw = orjson.dumps(
        [[column[0].item() for column in values[:4]], *values[4:]],
        option=orjson.OPT_SERIALIZE_NUMPY,
    )
    metadata, *columns = (part.split(",") for part in raw[2:-2].decode().split("],["))
    if b"null" in raw:
        convert = (
            _TableExport.json_cells if fmt == "json" else _TableExport.number_cells
        )
        for i, cell in enumerate(metadata):
            if cell == "null":
                metadata[i] = convert(values[i][:1])[0]
        for i, column in enumerate(columns):
            if "null" in column:
                columns[i] = convert(values[i + 4])

    cells = tuple(chain.from_iterable(zip(*columns, strict=True)))
    fields = (*metadata, *(("%s",) * len(columns)))
    if fmt == "json":
        template = _HISTOGRAM_JSON if names == HISTOGRAM else _DISTRIBUTION_JSON
        row = template % fields
        body = ((row + ",\n") * (rows - 1) + row) % cells
        return ("[\n" + body + "\n]\n").encode()
    sep = "\t" if fmt == "tsv" else ","
    row = sep.join(fields) + "\n"
    return (sep.join(names) + "\n" + (row * rows) % cells).encode()
