"""Resolve epoch markers against exact producer steps."""

from collections.abc import Callable, Iterable

import numpy as np

from tboardfs.model import EventFileCache

type _Markers = tuple[np.ndarray, np.ndarray]


class _Epochs:
    """Resolve scoped and flat epoch markers for one run."""

    _markers: dict[str, _Markers]

    def __init__(self, markers: dict[str, _Markers]) -> None:
        self._markers = markers

    @classmethod
    def from_caches(cls, caches: Iterable[EventFileCache]) -> "_Epochs":
        """Collect the last marker per step in source and producer order."""
        series: dict[str, dict[int, int]] = {}
        for cache in caches:
            for tag, columns in cache.scalars.items():
                steps = columns["step"]
                if not cls.is_marker(tag) or steps.dtype != np.int64 or not steps.size:
                    continue
                rows = series.setdefault(tag, {})
                for step, value in zip(
                    steps.tolist(), columns["value"].tolist(), strict=True
                ):
                    rows[step] = cls._number(value)
        markers = {}
        for tag, rows in series.items():
            ordered = sorted(rows)
            markers[tag] = (
                np.asarray(ordered, dtype=np.int64),
                np.fromiter((rows[step] for step in ordered), np.int64, len(ordered)),
            )
        return cls(markers)

    @staticmethod
    def is_marker(tag: str) -> bool:
        """Return whether the raw tag is a flat or single-component marker."""
        return tag == "epoch" or (tag.endswith("/epoch") and tag.count("/") == 1)

    @property
    def has_markers(self) -> bool:
        """Return whether any marker rows were recorded."""
        return bool(self._markers)

    @staticmethod
    def _number(value: object) -> int:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return -1
        if not 0 <= value < 1 << 63:
            return -1
        number = int(value)
        return number if number == value else -1

    def _series(self, tag: str) -> tuple[_Markers | None, _Markers | None]:
        return self._markers.get(tag.partition("/")[0] + "/epoch"), self._markers.get(
            "epoch"
        )

    def at(self, tag: str, step: int) -> int | None:
        """Return the epoch at a producer step, preferring a scoped marker."""
        if not -(1 << 63) <= step < 1 << 63:
            return None
        for series in self._series(tag):
            if series is not None:
                known, values = series
                index = np.searchsorted(known, step, side="right") - 1
                if index >= 0:
                    value = int(values[index])
                    return value if value >= 0 else None
        return None

    def align(
        self,
        tables: dict[str, dict[str, np.ndarray]],
        *,
        record: Callable[[str, np.ndarray], None] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Replace carried epochs and optionally report exact IDs for known rows."""
        aligned = {}
        for tag, columns in tables.items():
            steps = columns["step"]
            scoped, flat = self._series(tag)
            if steps.dtype != np.int64 or (scoped is None and flat is None):
                aligned[tag] = {**columns, "epoch": np.full(steps.shape, np.nan)}
                continue
            ids = np.full(steps.shape, -1, dtype=np.int64)
            for series in (flat, scoped):
                if series is not None:
                    known, values = series
                    index = np.searchsorted(known, steps, side="right") - 1
                    applicable = index >= 0
                    ids[applicable] = values[index[applicable]]
            unknown = ids < 0
            epoch = ids.astype(np.float64)
            epoch[unknown] = np.nan
            aligned[tag] = {**columns, "epoch": epoch}
            if record is not None and not unknown.all():
                record(tag, ids)
        return aligned
