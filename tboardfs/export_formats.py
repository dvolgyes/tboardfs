"""Select the formats of typed TensorBoard exports."""

from dataclasses import dataclass

from tboardfs.constants import (
    DEFAULT_HISTOGRAM_FORMATS,
    DEFAULT_SCALAR_FORMATS,
    TABLE_FORMATS,
)

type FormatList = str | list[str] | tuple[str, ...]


def normalize_formats(formats: FormatList, kind: str) -> tuple[str, ...]:
    """Normalize format names and reject unsupported formats."""
    items = formats.split(",") if isinstance(formats, str) else formats
    normalized = []
    for item in items:
        fmt = item.strip().lower().lstrip(".")
        if not fmt:
            continue
        if fmt not in TABLE_FORMATS:
            raise ValueError(
                f"unsupported {kind} format {fmt!r}; "
                f"use one or more of {', '.join(TABLE_FORMATS)}"
            )
        if fmt not in normalized:
            normalized.append(fmt)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class _ExportFormats:
    """Hold the normalized formats for one run projection."""

    scalars: tuple[str, ...] = DEFAULT_SCALAR_FORMATS
    histograms: tuple[str, ...] = DEFAULT_HISTOGRAM_FORMATS
    pr_curve_npy: bool = False

    @classmethod
    def parse(
        cls, scalars: FormatList, histograms: FormatList, pr_curve_npy: bool
    ) -> "_ExportFormats":
        """Select table formats, using defaults for empty inputs."""
        return cls(
            normalize_formats(scalars, "scalar") or DEFAULT_SCALAR_FORMATS,
            normalize_formats(histograms, "histogram") or DEFAULT_HISTOGRAM_FORMATS,
            pr_curve_npy,
        )
