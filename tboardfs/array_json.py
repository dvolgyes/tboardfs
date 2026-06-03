from typing import Any

import numpy as np


def array_to_json(data: np.ndarray) -> dict[str, Any]:
    """Return a JSON-safe array description."""
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "values": data.tolist(),
    }
