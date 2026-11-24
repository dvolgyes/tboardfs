"""Keep discovery, source checks, publications, and open handles local to one mount."""

from dataclasses import dataclass, field
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

from tboardfs._directory import _GuardPlans
from tboardfs.constants import DEFAULT_HISTOGRAM_FORMATS
from tboardfs.export_formats import FormatList, _ExportFormats
from tboardfs.leaf_cache import _MountTree
from tboardfs.model import EventFileCache, RunCache, SidecarEntry
from tboardfs.mount_index import MountIndexKeywords, _IndexOptions, _SiblingSnapshot
from tboardfs.run_tree import RunTree
from tboardfs.verified_span import _FrameValidations, _VerifiedSpan


class _MountKeywords(MountIndexKeywords, total=False):
    """Optional mount settings supplied as keywords."""

    epoch_breakdown: bool
    raw: bool
    histogram_formats: FormatList
    pr_curve_npy: bool


class _RunView:
    """Keep the mount's run projection and append policy together."""

    raw: bool
    epochs: bool
    formats: _ExportFormats
    _guide: dict[str, Any]

    def __init__(self, scalars: FormatList, options: _MountKeywords) -> None:
        self.raw = options.pop("raw", False)
        self.epochs = options.pop("epoch_breakdown", False)
        self.formats = _ExportFormats.parse(
            scalars,
            options.pop("histogram_formats", DEFAULT_HISTOGRAM_FORMATS),
            options.pop("pr_curve_npy", False),
        )
        self._guide = {
            "type": "bytes",
            "data": package_files("tboardfs").joinpath("llms.txt").read_bytes(),
            "mtime": 0.0,
        }

    def build(self, run: RunCache, step_digits: int) -> dict[str, Any]:
        """Build the run view with the per-run help file."""
        tree = RunTree.build(
            run, self.formats, step_digits, raw=self.raw, epoch_breakdown=self.epochs
        )
        tree["children"]["llms.txt"] = self._guide
        return tree

    def append(
        self,
        tree: dict[str, Any],
        run: RunCache,
        before: EventFileCache,
        step_digits: int,
    ) -> dict[str, Any] | None:
        """Rebuild lazily when a new epoch marker could move existing entries."""
        if self.epochs or before.source_path != max(run.files, default=None):
            return None
        return RunTree.append(
            tree,
            run,
            before,
            run.files[before.source_path],
            self.formats,
            step_digits,
            raw=self.raw,
        )


@dataclass
class _Directory:
    """Keep independent deadlines for directory discovery and known event sources."""

    scanned_at: float | None = None
    checked_at: float | None = None
    siblings: _SiblingSnapshot | None = None
    children: set[tuple[str, ...]] = field(default_factory=set)
    owned: set[Path] = field(default_factory=set)
    owner: tuple[str, ...] | None = None
    pieces: dict[tuple[str, ...], tuple[SidecarEntry, ...]] = field(
        default_factory=dict
    )

    def reusable(self, now: float, interval: float) -> bool:
        """Only a completed source check starts a reuse window."""
        return (
            interval > 0
            and self.checked_at is not None
            and now - self.checked_at < interval
        )

    def match(self, checked: float) -> frozenset[Path] | None:
        """Return only paths validated during this lookup, never on a timer hit."""
        siblings = self.siblings
        if siblings is None or not siblings.matches():
            return None
        self.checked_at = checked
        return siblings.paths


@dataclass
class _FsState:
    """The mount lock protects run publications, source clocks and handle ownership."""

    index: _IndexOptions
    guard_plans: _GuardPlans | None
    runs: dict[tuple[str, ...], RunCache] = field(default_factory=dict)
    directories: dict[tuple[str, ...], _Directory] = field(default_factory=dict)
    tree: _MountTree = field(default_factory=_MountTree)
    run_trees: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    next_version: int = 1
    frame_validations: _FrameValidations = field(default_factory=_FrameValidations)
    open_files: dict[int, tuple[bytes | _VerifiedSpan, float]] = field(
        default_factory=dict
    )
    next_handle: int = 1

    def visible_dirs(self) -> list[tuple[str, ...]]:
        """Keep existing files and still-cold branches proven by discovery."""
        visible = []
        parents: set[tuple[str, ...]] = {()}
        for parts in reversed(self.directories):
            if parts in parents or self._has_files(parts):
                visible.append(parts)
                parents.add(parts[:-1])
        return visible

    def restore_branches(self, branches: set[tuple[str, ...]]) -> bool:
        """Reopen hidden branches when a fresh search finds a logfile again."""
        restored = False
        for parts in branches:
            if not self._has_subtree(parts):
                self.directories[parts].scanned_at = None
                restored = True
        return restored

    def _has_subtree(self, parts: tuple[str, ...]) -> bool:
        pending = [parts]
        while pending:
            current = pending.pop()
            if self._has_files(current):
                return True
            pending.extend(self.directories[current].children)
        return False

    def _has_files(self, parts: tuple[str, ...]) -> bool:
        run = self.runs.get(parts)
        return self.directories[parts].scanned_at is None or (
            run is not None and bool(run.files)
        )
