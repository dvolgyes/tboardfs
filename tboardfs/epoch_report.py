"""Summarize step coverage by epoch and producer tag scope."""

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from types import MappingProxyType
from typing import Final

import numpy as np

from tboardfs.epochs import _Epochs
from tboardfs.run_inventory import _Inventory
from tboardfs.scalar_slice import _ScalarSlice

_CORE = ("epoch", "scope", "step_min", "step_max", "epoch_dir")
_STEP_COLUMNS: Final = MappingProxyType(
    {
        "audio": "audio_steps",
        "distributions": "distribution_steps",
        "graphs": "graph_steps",
        "histograms": "histogram_steps",
        "images": "image_steps",
        "meshes": "mesh_steps",
        "plugins": "plugin_steps",
        "pr_curves": "pr_curve_steps",
        "tensors": "tensor_steps",
        "text": "text_steps",
        "videos": "video_steps",
    }
)
type _Rows = dict[tuple[int, str], _Coverage]


@dataclass(slots=True)
class _Coverage:
    """Accumulate exact step bounds and counts without counting file formats."""

    low: int
    high: int
    metrics: int = 0
    steps: dict[str, set[int]] = field(default_factory=dict)

    @staticmethod
    def at(rows: _Rows, epoch: int, tag: str, low: int, high: int) -> "_Coverage":
        """Extend the bounds of an epoch and scope."""
        key = epoch, tag.partition("/")[0]
        row = rows.get(key)
        if row is None:
            rows[key] = row = _Coverage(low, high)
        else:
            row.low = min(row.low, low)
            row.high = max(row.high, high)
        return row

    def count(self, column: str) -> int:
        """Return zero when this scope lacks a column present elsewhere."""
        return (
            self.metrics if column == "metric_rows" else len(self.steps.get(column, ()))
        )


@dataclass(frozen=True, slots=True)
class _EpochReport:
    """Use one set of ordered rows for both the JSON and TSV exports."""

    columns: tuple[str, ...]
    rows: list[dict[str, int | str]]

    @classmethod
    def collect(
        cls, inventory: _Inventory, epochs: _Epochs, dirs: dict[int, str]
    ) -> "_EpochReport":
        """Collect coverage from projected tags with a visible epoch directory."""
        rows = cls._scan(inventory, epochs, dirs)
        optional = {column for row in rows.values() for column in row.steps}
        if any(row.metrics for row in rows.values()):
            optional.add("metric_rows")
        extra = tuple(sorted(optional))
        return cls(
            (*_CORE, *extra),
            [
                {
                    "epoch": epoch,
                    "scope": scope,
                    "step_min": row.low,
                    "step_max": row.high,
                    "epoch_dir": dirs[epoch],
                    **{column: row.count(column) for column in extra},
                }
                for (epoch, scope), row in sorted(rows.items())
            ],
        )

    @classmethod
    def _scan(
        cls, inventory: _Inventory, epochs: _Epochs, dirs: dict[int, str]
    ) -> _Rows:
        rows: _Rows = {}
        markers: _Rows = {}
        seen: set[str] = set()
        for item in inventory.items():
            tag = item.tag
            if tag is None:
                continue
            if item.series is not None:
                if tag not in seen:
                    seen.add(tag)
                    marker = epochs.is_marker(tag)
                    cls._scalars(
                        markers if marker else rows,
                        item.series["step"],
                        tag,
                        epochs,
                        dirs,
                        marker=marker,
                    )
            elif item.step is not None and (column := _STEP_COLUMNS.get(item.kind)):
                epoch = epochs.at(tag, item.step)
                if epoch is not None and epoch in dirs:
                    row = _Coverage.at(rows, epoch, tag, item.step, item.step)
                    row.steps.setdefault(column, set()).add(item.step)
        for key, row in markers.items():
            rows.setdefault(key, row)
        return rows

    @staticmethod
    def _scalars(
        rows: _Rows,
        steps: np.ndarray,
        tag: str,
        epochs: _Epochs,
        dirs: dict[int, str],
        *,
        marker: bool,
    ) -> None:
        for epoch, spans in _ScalarSlice.group(epochs.ids(tag, steps)).items():
            if epoch not in dirs:
                continue
            row = _Coverage.at(
                rows,
                epoch,
                tag,
                int(steps[spans[0].start]),
                int(steps[spans[-1].stop - 1]),
            )
            if not marker:
                row.metrics += sum(span.stop - span.start for span in spans)

    def render(self, fmt: str) -> bytes:
        """Encode the table with the same column order and counts in both formats."""
        if fmt == "json":
            return (json.dumps(self.rows, ensure_ascii=True, indent=2) + "\n").encode()
        with StringIO(newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
            return stream.getvalue().encode()
