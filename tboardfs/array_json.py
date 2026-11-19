from typing import Any

import numpy as np


def array_to_json(data: np.ndarray, *, dtype: np.dtype | None = None) -> dict[str, Any]:
    """Return an array description for json_document.

    Numbers stay one array, which json_document converts when the file is
    read: as Python lists they take about 8 times the memory. Strings become
    Python strings. A string preview can describe its corresponding Unicode
    storage dtype while keeping Python strings that retain trailing NULs.
    """
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype if dtype is None else dtype),
        "values": data.tolist() if data.dtype.kind in "OSU" else data,
    }
