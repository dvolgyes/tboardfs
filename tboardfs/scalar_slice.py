"""Group canonical scalar rows by epoch and render selected spans lazily."""

from dataclasses import dataclass
from itertools import chain, pairwise

import numpy as np

from tboardfs.tables import export_table


@dataclass(frozen=True, slots=True)
class _ScalarSlice:
    """Select spans from a canonical scalar table without storing copied rows."""

    series: dict[str, np.ndarray]
    spans: tuple[slice, ...]

    def render(self, fmt: str) -> bytes:
        """Export the selected rows in canonical order."""
        if len(self.spans) == 1:
            span = self.spans[0]
            return export_table(
                {column: values[span] for column, values in self.series.items()}, fmt
            )
        return export_table(
            {
                column: np.concatenate([values[span] for span in self.spans])
                for column, values in self.series.items()
            },
            fmt,
        )

    @staticmethod
    def group(epochs: np.ndarray) -> dict[int, tuple[slice, ...]]:
        """Group contiguous spans by exact epoch ID, excluding unknown rows."""
        if not epochs.size:
            return {}
        boundaries = np.flatnonzero(epochs[1:] != epochs[:-1])
        groups: dict[int, list[slice]] = {}
        run_starts = chain((0,), (int(i) + 1 for i in boundaries), (epochs.size,))
        for start, stop in pairwise(run_starts):
            epoch = int(epochs[start])
            if epoch != -1:
                groups.setdefault(epoch, []).append(slice(start, stop))
        return {epoch: tuple(spans) for epoch, spans in groups.items()}
