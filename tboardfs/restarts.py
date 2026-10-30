"""Drop the rows that a restarted training run writes again, as TensorBoard does."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import numpy as np

from tboardfs.model import EventFileCache, RestartMark

# A START step and the rows before it by (table, tag); None means all rows.
_Start = tuple[float, Mapping[tuple[str, str], int] | None]


class _Stepped(Protocol):
    """An indexed payload with a TensorBoard step."""

    @property
    def step(self) -> int | float:
        """Return the step; NaN when the event has none."""
        ...


class _RestartPurge:
    """Apply TensorBoard's SessionLog.START purge to the event files of a run.

    TensorBoard reads the files of a run in path order. When the file version
    is 2 or later, a SessionLog.START at step S drops each row read before it
    whose step is S or more; a missing step counts as 0. Rows after the START
    stay. Graphs are not dropped. A file without its own version uses the
    version of the previous file. Files without version 2 use another
    TensorBoard rule that drops rows when steps go backward; tboardfs does not
    apply that rule.
    """

    @staticmethod
    def parse_version(text: str) -> float:
        """Return the number of a ``brain.Event:N`` file version, or -1."""
        try:
            return float(text.split("brain.Event:")[-1])
        except ValueError:
            return -1.0

    @staticmethod
    def purge(caches: list[EventFileCache]) -> list[EventFileCache]:
        """Return the caches, in the same order, without the replaced rows."""
        starts = _RestartPurge.starts(caches)
        if not starts:
            return caches
        return [
            _RestartPurge.purge_file(
                cache,
                [
                    (mark.value, None if file > index else mark.counts)
                    for file, mark in starts
                    if file >= index
                ],
            )
            for index, cache in enumerate(caches)
        ]

    @staticmethod
    def starts(caches: list[EventFileCache]) -> list[tuple[int, RestartMark]]:
        """Return the START marks that purge, with the index of their file."""
        version: float | None = None
        found = []
        for index, cache in enumerate(caches):
            for mark in cache.restarts:
                if mark.kind == "version":
                    version = mark.value
                elif version is not None and version >= 2:
                    found.append((index, mark))
        return found

    @staticmethod
    def purge_file(cache: EventFileCache, starts: list[_Start]) -> EventFileCache:
        """Return a copy of one file cache without the rows the starts drop."""
        if not starts:
            return cache
        scalars = _RestartPurge.purge_series(cache.scalars, "scalars", starts)
        histograms = _RestartPurge.purge_series(cache.histograms, "histograms", starts)
        return replace(
            cache,
            scalars=scalars,
            scalar_mtimes={
                tag: mtime
                for tag, mtime in cache.scalar_mtimes.items()
                if tag in scalars
            },
            histograms=histograms,
            histogram_mtimes={
                tag: mtime
                for tag, mtime in cache.histogram_mtimes.items()
                if tag in histograms
            },
            binaries=_RestartPurge.purge_entries(cache.binaries, "binaries", starts),
            json_entries=_RestartPurge.purge_entries(
                cache.json_entries, "json_entries", starts
            ),
        )

    @staticmethod
    def purge_series(
        series_by_tag: dict[str, dict[str, np.ndarray]],
        table: str,
        starts: list[_Start],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Drop replaced rows from column tables; drop tags left empty."""
        purged = {}
        for tag, series in series_by_tag.items():
            keep = _RestartPurge.kept_rows(series["step"], (table, tag), starts)
            if keep.any():
                purged[tag] = {name: column[keep] for name, column in series.items()}
        return purged

    @staticmethod
    def purge_entries[Entry: _Stepped](
        entries: list[Entry], table: str, starts: list[_Start]
    ) -> list[Entry]:
        """Drop replaced entries from a list of indexed payloads."""
        steps = np.array([entry.step for entry in entries], dtype=np.float64)
        keep = _RestartPurge.kept_rows(steps, (table, ""), starts)
        return [entry for entry, kept in zip(entries, keep, strict=True) if kept]

    @staticmethod
    def kept_rows(
        steps: np.ndarray,
        key: tuple[str, str],
        starts: list[_Start],
    ) -> np.ndarray:
        """Return a mask of the rows that no START drops."""
        steps = np.nan_to_num(np.asarray(steps, dtype=np.float64), nan=0.0)
        index = np.arange(len(steps))
        keep = np.ones(len(steps), dtype=bool)
        for step, counts in starts:
            before = len(steps) if counts is None else counts.get(key, 0)
            keep &= ~((steps >= step) & (index < before))
        return keep
