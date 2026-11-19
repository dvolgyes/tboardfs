"""Render heterogeneous table columns with pandas' CSV quoting and missing-value rules."""

from io import BytesIO

import numpy as np
import pandas as pd


def render(series: dict[str, np.ndarray], separator: str) -> bytes:
    """Encode columns that require a dataframe rather than the numeric fast path."""
    handle = BytesIO()
    pd.DataFrame(series).to_csv(handle, sep=separator, index=False, na_rep="nan")
    return handle.getvalue()
