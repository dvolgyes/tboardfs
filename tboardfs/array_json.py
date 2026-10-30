from typing import Any

import numpy as np


def array_to_json(data: np.ndarray) -> dict[str, Any]:
    """Return an array description for json_document.

    Numbers stay one array, which json_document converts when the file is
    read: as Python lists they take about 8 times the memory. Strings become
    Python strings, which are smaller than a fixed-width text array.
    """
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "values": data.tolist() if data.dtype.kind in "OSU" else data,
    }
