"""Discover sidecars at one visited directory without searching other runs."""

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from tboardfs.constants import TENSORBOARD_EVENT_GLOB
from tboardfs.model import SidecarEntry
from tboardfs.open3d_asset import OPEN3D_PLUGIN, Open3DAsset
from tboardfs.paths import _Paths
from tboardfs.projector import CONFIG_NAME, _ProjectorConfig


@dataclass
class SidecarScan:
    """Collect only sidecar subtrees and projector-config targets of one listing."""

    directory: Path
    run_dir: Path
    allowed: Callable[[Path], bool]
    owned: set[Path] = field(default_factory=set)
    boundaries: set[Path] = field(default_factory=set)
    needed: set[Path] = field(default_factory=set)
    named: set[Path] = field(default_factory=set)
    visited: set[Path] = field(default_factory=set)

    def collect(self, entries: list[os.DirEntry[str]]) -> tuple[SidecarEntry, ...]:
        """Return the selected files and retain the roots used only for sidecars."""
        result = []
        pending = [(self.directory, entries)]
        while pending:
            directory, listing = pending.pop()
            if self._separate_run(directory, listing):
                continue
            self._read_configs(listing)
            for entry in listing:
                path = Path(entry.path)
                if _Paths.is_file(entry):
                    item = self._entry(path)
                    if item is not None:
                        result.append(item)
                elif _Paths.is_dir(entry) and self._descend(path):
                    pending.append((path, self._listing(path)))
        self.owned.difference_update(self.boundaries)
        return tuple(result)

    def _separate_run(self, path: Path, entries: list[os.DirEntry[str]]) -> bool:
        if path == self.directory:
            return False
        top = self.directory / path.relative_to(self.directory).parts[0]
        if _Paths.event_files(path, entries):
            self.boundaries.add(top)
            return True
        self.owned.add(top)
        return False

    def _listing(self, path: Path) -> list[os.DirEntry[str]]:
        return [
            entry
            for entry in _Paths.entries(path)
            if not entry.is_symlink() or self.allowed(Path(entry.path))
        ]

    def _read_configs(self, entries: list[os.DirEntry[str]]) -> None:
        for entry in entries:
            if entry.name != CONFIG_NAME or not _Paths.is_file(entry):
                continue
            path = Path(entry.path)
            for names in _ProjectorConfig.named_parts(path):
                if ".." in names or fnmatchcase(names[-1], TENSORBOARD_EVENT_GLOB):
                    continue
                self.named.add(path.parent.joinpath(*names))
                self.needed.update(
                    path.parent.joinpath(*names[:i]) for i in range(1, len(names))
                )

    def _descend(self, path: Path) -> bool:
        top = path.relative_to(self.directory).parts[0].lower()
        if path not in self.needed and top not in {"plugins", "profile", "projector"}:
            return False
        real = path.resolve()
        if real in self.visited or self.directory.resolve().is_relative_to(real):
            return False
        self.visited.add(real)
        return True

    def _entry(self, path: Path) -> SidecarEntry | None:
        if fnmatchcase(path.name, TENSORBOARD_EVENT_GLOB):
            return None
        parts = path.relative_to(self.run_dir).parts
        if Open3DAsset.matches(parts):
            return Open3DAsset.stat(path)
        lowered = tuple(part.lower() for part in parts)
        if parts == ("hparams.yaml",):
            tab = "hparams"
        elif "profile" in lowered:
            tab = "profile"
        elif "projector" in lowered or path.name == CONFIG_NAME or path in self.named:
            tab = "projector"
        else:
            return None
        return _stat_entry(path, tab, _relative_name(parts, tab))

    @staticmethod
    def restat(
        entries: tuple[SidecarEntry, ...], allowed: Callable[[Path], bool]
    ) -> tuple[SidecarEntry, ...]:
        """Retain the known regular files with their current size and timestamp."""
        return tuple(
            item
            for entry in entries
            if allowed(entry.source_path)
            if (
                item := (
                    Open3DAsset.stat(entry.source_path)
                    if entry.tab == OPEN3D_PLUGIN
                    else _stat_entry(entry.source_path, entry.tab, entry.rel_parts)
                )
            )
            is not None
        )

    @staticmethod
    def rebase(
        entries: tuple[SidecarEntry, ...], run_dir: Path
    ) -> tuple[SidecarEntry, ...]:
        """Assign already discovered files to a changed enclosing run."""
        result = []
        for entry in entries:
            parts = entry.source_path.relative_to(run_dir).parts
            if entry.tab == "hparams" and parts != ("hparams.yaml",):
                continue
            if entry.tab == OPEN3D_PLUGIN:
                if Open3DAsset.matches(parts):
                    result.append(entry)
                continue
            result.append(
                SidecarEntry(
                    entry.source_path,
                    entry.tab,
                    _relative_name(parts, entry.tab),
                    entry.size,
                    entry.mtime_ns,
                )
            )
        return tuple(result)


def _relative_name(parts: tuple[str, ...], tab: str) -> tuple[str, ...]:
    lowered = tuple(part.lower() for part in parts)
    if tab in lowered:
        return parts[lowered.index(tab) + 1 :] or parts
    return parts


def _stat_entry(path: Path, tab: str, parts: tuple[str, ...]) -> SidecarEntry | None:
    try:
        status = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return SidecarEntry(path, tab, parts, status.st_size, status.st_mtime_ns)
