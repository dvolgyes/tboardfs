"""Project canonical files and scalar rows into nonempty epoch directories."""

from collections.abc import Iterator
from typing import Any, cast

import numpy as np

from tboardfs.epochs import _Epochs
from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.paths import _DirectoryNode, _Paths
from tboardfs.run_listing import RunListing
from tboardfs.scalar_slice import _ScalarSlice

type _Path = tuple[str, ...]
type _Spans = dict[int, tuple[slice, ...]]


class _EpochBreakdown:
    """Publish epoch views from the winning canonical leaves."""

    _epochs: _Epochs
    _leaves: dict[int, tuple[str, int | None]]
    _scalars: dict[str, _Spans]

    def __init__(self, epochs: _Epochs) -> None:
        self._epochs = epochs
        self._leaves = {}
        self._scalars = {}

    def record_scalars(self, tag: str, ids: np.ndarray) -> None:
        """Keep the row spans of each known epoch in canonical order."""
        self._scalars[tag] = _ScalarSlice.group(ids)

    def track(self, value: LeafValue, tag: str, step: float | None) -> None:
        """Retain raw tag and step provenance until names are resolved."""
        if type(step) is int:
            self._leaves[id(value)] = (tag, step)
        elif tag in self._scalars and LeafSpec.kind_of(value) == "scalar":
            self._leaves[id(value)] = (tag, None)

    def project(self, run: dict[str, Any]) -> None:
        """Add nonempty epoch directories and preserve each canonical path."""
        breakdown = _DirectoryNode()
        groups: dict[int, dict[str, Any]] = {}
        for path, value in _canonical_leaves(run):
            recorded = self._leaves.get(id(value))
            if recorded is None:
                continue
            tag, step = recorded
            if step is None:
                for group, leaf in self._scalar_leaves(tag, cast(LeafSpec, value)):
                    self._insert(breakdown, groups, group, path, leaf)
            else:
                epoch = self._epochs.at(tag, step)
                if epoch is not None:
                    self._insert(breakdown, groups, epoch, path, value)
        if groups:
            run["children"]["epoch-breakdown"] = breakdown

    def _scalar_leaves(
        self, tag: str, leaf: LeafSpec
    ) -> Iterator[tuple[int, LeafSpec]]:
        series = cast(dict[str, np.ndarray], leaf.payload)
        for epoch, spans in self._scalars[tag].items():
            yield (
                epoch,
                LeafSpec(
                    "scalar_slice",
                    leaf.mtime,
                    "slice",
                    _ScalarSlice(series, spans),
                    format=leaf.format,
                ),
            )

    @staticmethod
    def _insert(
        breakdown: dict[str, Any],
        groups: dict[int, dict[str, Any]],
        epoch: int,
        path: _Path,
        leaf: LeafValue,
    ) -> None:
        group = groups.get(epoch)
        if group is None:
            groups[epoch] = group = _Paths.mkdirs(breakdown, (f"epoch-{epoch}",))
        parent = _Paths.mkdirs(group, path[:-1])
        children = parent["children"]
        if not isinstance(children, RunListing):
            children = parent["children"] = RunListing(children)
        children.set_leaf(path[-1], leaf)


def _canonical_leaves(run: dict[str, Any]) -> Iterator[tuple[_Path, LeafValue]]:
    pending: list[tuple[_Path, dict[str, Any]]] = [((), run)]
    while pending:
        path, directory = pending.pop()
        children = directory["children"]
        if isinstance(children, RunListing):
            children = children.entries
        for name, value in children.items():
            if not path and name in {
                "raw",
                "hparams",
                "custom_scalars",
                "epoch-breakdown",
                "name_map.json",
            }:
                continue
            child = (*path, name)
            if LeafSpec.kind_of(value) == "dir":
                pending.append((child, value))
            else:
                yield child, value
