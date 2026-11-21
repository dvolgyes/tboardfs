"""Drop the rows that a restarted training run writes again, as TensorBoard does."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tboardfs.model import EventFileCache, RestartMark
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.graph_summary_entry import GraphSummaryEntry
from tboardfs.model.packed_binaries import BinaryEntries, PackedBinaries
from tboardfs.model.packed_source import BinaryProjection, PackedSource

# A START step and the rows before it by (table, tag); None means all rows.
_Start = tuple[float, Mapping[tuple[str, str], int] | None]


def _offset_restarts(
    marks: list[RestartMark], prefix_counts: Mapping[tuple[str, str], int]
) -> Iterator[RestartMark]:
    """Add rows and entries from an earlier prefix to each START count."""
    return (
        _offset_restart(mark, prefix_counts) if mark.kind == "start" else mark
        for mark in marks
    )


def _offset_restart(
    mark: RestartMark, prefix_counts: Mapping[tuple[str, str], int]
) -> RestartMark:
    counts = dict(mark.counts)
    for key, length in prefix_counts.items():
        counts[key] = counts.get(key, 0) + length
    return replace(mark, counts=counts)


class _Stepped(Protocol):
    """An indexed payload with a TensorBoard step."""

    @property
    def step(self) -> int | float:
        """Return the step; 0 when the event has none."""
        ...


@dataclass(slots=True)
class _SelectedRows:
    """The surviving ordinals of a packed binary source."""

    source: PackedSource
    indices: NDArray[np.intp]

    def __len__(self) -> int:
        return len(self.indices)

    def entry_at(self, index: int) -> BinaryEntry:
        """Resolve a surviving row against its original source."""
        return self.source.entry_at(int(self.indices[index]))

    def project(self) -> Iterator[BinaryProjection]:
        """Translate original ordinals to the filtered sequence."""
        if not self.indices.size:
            return
        position = 0
        for key, step, wall_time, index in self.source.project():
            if index == self.indices[position]:
                yield key, step, wall_time, position
                position += 1
                if position == len(self.indices):
                    break

    def summaries(self) -> Iterator[GraphSummaryEntry]:
        """Preserve graph companions only for surviving summaries."""
        for (_, kind, _, _), _, _, index in self.project():
            if kind == "graphs":
                entry = self.entry_at(index)
                if isinstance(entry, GraphSummaryEntry):
                    yield entry


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
            binaries=_RestartPurge.purge_binaries(cache.binaries, starts),
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
        steps = np.asarray([entry.step for entry in entries])
        keep = _RestartPurge.kept_rows(steps, (table, ""), starts)
        return [entry for entry, kept in zip(entries, keep, strict=True) if kept]

    @staticmethod
    def purge_binaries(entries: BinaryEntries, starts: list[_Start]) -> BinaryEntries:
        """Retain packed rows by ordinal without reconstructing their payloads."""
        if isinstance(entries, list):
            return _RestartPurge.purge_entries(entries, "binaries", starts)
        key = ("binaries", "")
        if not entries or not any(
            counts is None or counts.get(key, 0) for _, counts in starts
        ):
            return entries
        steps: list[int | float] = []
        for chunk in entries.chunks():
            if isinstance(chunk, list):
                steps.extend(entry.step for entry in chunk)
            else:
                steps.extend(step for _, step, _, _ in chunk.project())
        keep = _RestartPurge.kept_rows(np.asarray(steps), key, starts)
        if keep.all():
            return entries
        result: BinaryEntries = []
        offset = 0
        for chunk in entries.chunks():
            end = offset + len(chunk)
            selection = keep[offset:end]
            if isinstance(chunk, list):
                result = PackedBinaries.merge(
                    result,
                    [
                        entry
                        for entry, kept in zip(chunk, selection, strict=True)
                        if kept
                    ],
                )
            elif selection.all():
                result = PackedBinaries.merge(result, PackedBinaries(chunk))
            elif selection.any():
                source = _SelectedRows(chunk, np.flatnonzero(selection))
                result = PackedBinaries.merge(result, PackedBinaries(source))
            offset = end
        return result

    @staticmethod
    def kept_rows(
        steps: np.ndarray,
        key: tuple[str, str],
        starts: list[_Start],
    ) -> np.ndarray:
        """Return a mask of the rows that no START drops."""
        keep = np.ones(len(steps), dtype=bool)
        for step, counts in starts:
            before = len(steps) if counts is None else counts.get(key, 0)
            keep[:before] &= ~(steps[:before] >= step)
        return keep
