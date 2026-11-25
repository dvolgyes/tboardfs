"""Summarize the canonical typed files and observed steps of one run."""

from dataclasses import dataclass, field

import numpy as np

from tboardfs.epochs import _Epochs
from tboardfs.run_inventory import _Inventory, _Item
from tboardfs.tables import _TableExport


def _epoch_ranges(ids: np.ndarray) -> list[list[int]]:
    """Collapse consecutive observed IDs without filling gaps or unknowns."""
    ranges: list[list[int]] = []
    for epoch in map(int, np.unique(ids)):
        if epoch < 0:
            continue
        if ranges and epoch == ranges[-1][1] + 1:
            ranges[-1][1] = epoch
        else:
            ranges.append([epoch, epoch])
    return ranges


@dataclass(slots=True)
class _TagFiles:
    """Accumulate physical catalog metadata for one canonical tag group."""

    file_count: int = 0
    formats: set[str] = field(default_factory=set)
    paths: dict[str, str] = field(default_factory=dict)
    directories: set[tuple[str, ...]] = field(default_factory=set)
    steps: set[int] = field(default_factory=set)
    scalar_steps: np.ndarray | None = None

    def add(self, item: "_Item") -> None:
        """Record one canonical file, retaining a shared scalar column only once."""
        self.file_count += 1
        if item.format is not None:
            self.formats.add(item.format)
        if item.kind == "scalars" and item.series is not None:
            if item.format is not None:
                self.paths[item.format] = "/".join(item.path)
            if self.scalar_steps is None:
                self.scalar_steps = item.series["step"]
        else:
            self.directories.add(item.path[:-1])
            if item.step is not None:
                self.steps.add(item.step)

    def entry(self, kind: str, tag: str, epochs: "_Epochs") -> dict[str, object]:
        """Describe the files and their producer steps without rendering them."""
        entry: dict[str, object] = {
            "tag": tag,
            "kind": kind,
            "formats": sorted(self.formats),
            "file_count": self.file_count,
        }
        if (steps := self.scalar_steps) is not None:
            entry["paths"] = dict(sorted(self.paths.items()))
            entry["observations"] = len(steps)
            if len(steps):
                ids = epochs.ids(tag, steps)
                entry.update(
                    distinct_steps=1 + int(np.count_nonzero(steps[1:] != steps[:-1])),
                    step_min=int(steps[0]),
                    step_max=int(steps[-1]),
                    known_epoch_ranges=_epoch_ranges(ids),
                    observations_without_epoch=int(np.count_nonzero(ids == -1)),
                )
        else:
            entry["directories"] = sorted("/".join(path) for path in self.directories)
            if self.steps:
                steps = np.fromiter(self.steps, dtype=np.int64, count=len(self.steps))
                entry.update(
                    distinct_steps=len(self.steps),
                    step_min=int(steps.min()),
                    step_max=int(steps.max()),
                    known_epoch_ranges=_epoch_ranges(epochs.ids(tag, steps)),
                )
        return entry


class _Catalog:
    """Render a deterministic catalog for a single run."""

    @staticmethod
    def render(inventory: "_Inventory", epochs: "_Epochs") -> bytes:
        """Serialize catalog metadata from canonical typed leaves only."""
        groups: dict[tuple[str, str, tuple[str, ...]], _TagFiles] = {}
        files: list[str] = []
        for item in inventory.items():
            if item.tag is None:
                files.append("/".join(item.path))
                continue
            namespace = item.path[1:2] if item.kind == "plugins" else ()
            key = (item.kind, item.tag, namespace)
            group = groups.get(key)
            if group is None:
                groups[key] = group = _TagFiles()
            group.add(item)
        tags = [groups[key].entry(key[0], key[1], epochs) for key in sorted(groups)]
        return _TableExport.json_document(
            {"version": 1, "tags": tags, "files": sorted(files)}
        )
