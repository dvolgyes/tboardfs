"""Keep discovery, source checks, publications, and open handles local to one mount."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tboardfs._directory import _GuardPlans
from tboardfs.leaf_cache import _MountTree
from tboardfs.model import RunCache, SidecarEntry
from tboardfs.mount_index import _IndexOptions, _SiblingSnapshot
from tboardfs.verified_span import _FrameValidations, _VerifiedSpan


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
