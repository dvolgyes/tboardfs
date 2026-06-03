from io import BytesIO

import numpy as np


def array_to_npy(data: np.ndarray) -> bytes:
    """Serialize one array as a .npy payload."""
    handle = BytesIO()
    np.save(handle, data)
    return handle.getvalue()
