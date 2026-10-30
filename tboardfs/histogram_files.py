"""Add the per-step histogram and distribution files of a run."""

from functools import partial
from typing import Any

from tboardfs.model import EventFileCache
from tboardfs.paths import _Paths
from tboardfs.run_files import RunFiles
from tboardfs.skip_failure import skip_failure
from tboardfs.tables import _TableMerge


def add_histogram_files(
    files: RunFiles,
    caches: list[EventFileCache],
    epochs: dict[str, Any] | None,
    scalar_formats: tuple[str, ...],
    step_digits: int,
) -> None:
    """Add the histogram and distribution tables of one run, with its epochs.

    A tag whose files fail to build is skipped with a warning.
    """
    histograms = _TableMerge.merge_series(cache.histograms for cache in caches)
    histograms = _TableMerge.step_epochs(histograms, epochs)
    mtimes = _TableMerge.merge_mtimes(cache.histogram_mtimes for cache in caches)
    for tag, series in histograms.items():
        skip_failure(
            f"histogram {tag!r}",
            partial(
                _HistogramFiles.add_tag_files,
                files,
                {"tag": tag, "series": series, "mtime": mtimes.get(tag, 0.0)},
                scalar_formats,
                step_digits,
            ),
        )


class _HistogramFiles:
    """Add per-step histogram and distribution files."""

    @staticmethod
    def add_tag_files(
        files: RunFiles,
        item: dict[str, Any],
        scalar_formats: tuple[str, ...],
        step_digits: int,
    ) -> None:
        """Add the histogram and distribution tables of one tag."""
        tag, series, mtime = item["tag"], item["series"], item["mtime"]
        steps = _TableMerge.unique_steps(series)
        for fmt in scalar_formats:
            for step in steps:
                step_series = _TableMerge.series_for_step(series, step)
                _HistogramFiles.add_step_table(
                    files,
                    {
                        "tab": "histograms",
                        "tag": tag,
                        "step": step,
                        "series": step_series,
                        "fmt": fmt,
                        "mtime": mtime,
                    },
                    step_digits,
                )
                distribution = _TableMerge.distribution_from_histogram(step_series)
                _HistogramFiles.add_step_table(
                    files,
                    {
                        "tab": "distributions",
                        "tag": tag,
                        "step": step,
                        "series": distribution,
                        "fmt": fmt,
                        "mtime": mtime,
                    },
                    step_digits,
                )

    @staticmethod
    def add_step_table(files: RunFiles, item: dict[str, Any], step_digits: int) -> None:
        """Add one per-step tabular file."""
        files.add(
            (item["tab"],),
            item["tag"],
            f"{_Paths.step_name(item['step'], step_digits)}.{item['fmt']}",
            {
                "type": "table",
                "series": item["series"],
                "format": item["fmt"],
                "mtime": item["mtime"],
            },
        )
