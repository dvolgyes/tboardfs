from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scalar:
    """Scalar value and its NumPy dtype.

    :ivar value: scalar value
    :ivar dtype: NumPy dtype for the scalar
    """

    value: Any
    dtype: np.dtype
