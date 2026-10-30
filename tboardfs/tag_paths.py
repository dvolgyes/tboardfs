"""Give each tag of one run its own virtual path."""

from collections.abc import Iterable, Mapping

from loguru import logger

from tboardfs.model import RunCache
from tboardfs.paths import _Paths


class TagPaths:
    """Give each tag of one run its own virtual path.

    Different tags can map to the same path, such as "" and "untagged", "a/b"
    and "a//b", or "\\" and "%5C". A tag that its path spells exactly keeps
    that path. Each other tag whose path is in use gets "~2", "~3", ... on its
    last part, in sorted tag order, so every tag keeps its own files.

    A blocked directory is a tag path prefix that is also the path of a file,
    such as a/000001.npy when the tag "a" has that step file. Every tag below
    it moves to the first free "~N" of that part, the same in every tab.

    Attributes:
        renamed: by tag, the path with a suffix and the reason for it.
    """

    renamed: dict[str, tuple[tuple[str, ...], str]]

    def __init__(
        self, tags: Iterable[str], blocked: Mapping[tuple[str, ...], str]
    ) -> None:
        """Resolve the path of every tag.

        Args:
            tags: the tags of the run.
            blocked: the directories that are files, with the path of the file.
        """
        self.renamed = {}
        natural = {tag: _Paths.tag_dir_parts(tag) for tag in set(tags)}
        taken = {
            parts[:end]
            for parts in natural.values()
            for end in range(1, len(parts) + 1)
        }
        moves: dict[tuple[str, ...], tuple[str, ...]] = {}
        holders: dict[tuple[str, ...], str] = {}
        for tag in sorted(natural, key=lambda tag: (not TagPaths.spelled(tag), tag)):
            parts, reason = natural[tag], ""
            for end in range(1, len(parts) + 1):
                if parts[:end] in blocked:
                    reason = f"{blocked[parts[:end]]!r} is a file"
                    if parts[:end] not in moves:
                        moves[parts[:end]] = TagPaths.free(parts[:end], taken, blocked)
                        taken.add(moves[parts[:end]])
                    parts = (*moves[parts[:end]], *parts[end:])
            path, number = parts, 1
            while path in holders or path in blocked:
                number += 1
                path = (*parts[:-1], f"{parts[-1]}~{number}")
            if parts in holders:
                reason = f"tag {holders[parts]!r} has the same path"
            holders[path] = tag
            if path != natural[tag]:
                self.renamed[tag] = (path, reason)

    @staticmethod
    def free(
        prefix: tuple[str, ...],
        taken: set[tuple[str, ...]],
        blocked: Mapping[tuple[str, ...], str],
    ) -> tuple[str, ...]:
        """Return the prefix with the first "~N" on its last part that is free."""
        number = 2
        while (moved := (*prefix[:-1], f"{prefix[-1]}~{number}")) in taken or (
            moved in blocked
        ):
            number += 1
        return moved

    def log(self, run: RunCache) -> None:
        """Log each tag that gets a suffix, and keep the suffixes in the run.

        A run that grows logs a suffix once, and again only when it changes.
        """
        for tag, (path, reason) in self.renamed.items():
            if run.renamed_tags.get(tag) != path:
                logger.warning(
                    "the files of tag {!r} in run {!r} are under {!r}: {}",
                    tag,
                    "/".join(run.run_parts),
                    "/".join(path),
                    reason,
                )
        run.renamed_tags = {tag: path for tag, (path, _) in self.renamed.items()}

    @staticmethod
    def spelled(tag: str) -> bool:
        """Return true when the virtual path of a tag spells the tag exactly."""
        return "/".join(_Paths.tag_dir_parts(tag)) == tag

    def dir_parts(self, tag: str) -> tuple[str, ...]:
        """Return the virtual directory parts of a tag."""
        if tag in self.renamed:
            return self.renamed[tag][0]
        return _Paths.tag_dir_parts(tag)
