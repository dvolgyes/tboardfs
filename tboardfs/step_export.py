"""Render per-step tables without changing the exported column mapping."""

import numpy as np

from tboardfs._step_npz import StepArchives
from tboardfs._step_text import step_text
from tboardfs.tables import export_table


class StepColumns(dict[str, np.ndarray]):
    """Share archive layouts across a projection, not exported payloads."""

    __slots__: tuple[str, ...] = ("_archives",)
    _archives: StepArchives

    def __init__(self, series: dict[str, np.ndarray], archives: StepArchives) -> None:
        super().__init__(series)
        self._archives = archives

    def render(self, fmt: str) -> bytes:
        """Use the generic writer for unsupported layouts and representations."""
        data = self._archives.encode(self) if fmt == "npz" else step_text(self, fmt)
        return export_table(self, fmt) if data is None else data


def _render_step(series: dict[str, np.ndarray], fmt: str) -> bytes:
    """Render the projection or a plain table supplied by another producer."""
    return (
        series.render(fmt)
        if isinstance(series, StepColumns)
        else export_table(series, fmt)
    )
