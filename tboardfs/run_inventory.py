"""Enumerate typed files from a published run tree without rendering them."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from tboardfs.leaf_spec import LeafSpec, LeafValue
from tboardfs.model.binary_entry import BinaryEntry
from tboardfs.model.packed_leaf import PackedLeaf
from tboardfs.paths import NAME_MAP
from tboardfs.run_listing import RunListing, _LogicalFile

_OMIT_ROOT = frozenset(
    (
        "raw",
        "epoch-breakdown",
        "llms.txt",
        NAME_MAP,
        "catalog.json",
        "epochs.json",
        "epochs.tsv",
    )
)


@dataclass(frozen=True, slots=True)
class _Item:
    """One canonical typed leaf and its indexed producer metadata."""

    path: tuple[str, ...]
    kind: str
    tag: str | None
    format: str | None
    step: int | None
    series: dict[str, np.ndarray] | None

    @classmethod
    def from_leaf(
        cls, path: tuple[str, ...], value: LeafValue, logical: _LogicalFile | None
    ) -> "_Item":
        """Read provenance from the tree or the packed binary index."""
        tag = None if logical is None else logical[1]
        step = None if logical is None else logical[5]
        entry = _binary(value.get("entry") if isinstance(value, dict) else value)
        fmt: str | None
        if entry is not None:
            if logical is None:
                tag, step = entry.tag, entry.step
            fmt = entry.extension
        elif logical is not None:
            name, named = logical[2], logical[3]
            fmt = name if named else (name.rsplit(".", 1)[-1] if "." in name else None)
        else:
            fmt = value.format if isinstance(value, LeafSpec) else None
        return cls(
            path,
            path[0],
            tag,
            fmt,
            step if type(step) is int else None,
            cls._series(value, logical),
        )

    @staticmethod
    def _series(
        value: LeafValue, logical: _LogicalFile | None
    ) -> dict[str, np.ndarray] | None:
        if logical is None or logical[-1] != "scalar":
            return None
        if (
            isinstance(value, LeafSpec)
            and value.kind == "scalar"
            and value.key == "series"
        ):
            return cast(dict[str, np.ndarray], value.payload)
        if isinstance(value, dict) and value.get("type") == "scalar":
            series = value.get("series")
            return series if isinstance(series, dict) else None
        return None


@dataclass(frozen=True, slots=True)
class _Inventory:
    """Hold a shallow typed-root snapshot and enumerate leaves on demand."""

    _root: tuple[tuple[str, LeafValue, _LogicalFile | None], ...]

    def __init__(self, tree: dict[str, Any]) -> None:
        object.__setattr__(
            self,
            "_root",
            tuple(item for item in _children(tree) if item[0] not in _OMIT_ROOT),
        )

    def items(self) -> Iterator[_Item]:
        """Yield typed leaves without expanding or sorting entire directories."""
        pending: list[
            tuple[tuple[str, ...], Iterator[tuple[str, LeafValue, _LogicalFile | None]]]
        ] = [((), iter(self._root))]
        while pending:
            parent, children = pending[-1]
            for name, value, logical in children:
                path = (*parent, name)
                if isinstance(value, dict) and value["type"] == "dir":
                    pending.append((path, _children(value)))
                    break
                yield _Item.from_leaf(path, value, logical)
            else:
                pending.pop()


def _children(
    tree: dict[str, Any],
) -> Iterator[tuple[str, LeafValue, _LogicalFile | None]]:
    children = tree["children"]
    if isinstance(children, RunListing):
        for name, value in children.entries.items():
            yield name, value, children.logical.get(name)
    else:
        for name, value in children.items():
            yield name, value, None


def _binary(value: object) -> BinaryEntry | None:
    if isinstance(value, PackedLeaf):
        source = value.source
        return (
            source if isinstance(source, BinaryEntry) else source.entry_at(value.index)
        )
    return value if isinstance(value, BinaryEntry) else None
