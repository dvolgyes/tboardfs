"""Give each tag of one run its own virtual path."""

from collections.abc import Iterable, Mapping

from loguru import logger

from tboardfs.model import RunCache
from tboardfs.paths import _Paths


class TagPaths:
    """Give each tag of one run its own virtual path.

    A tag's natural path reversibly encodes each slash-separated component.

    A blocked directory is a tag path prefix that is also the path of a file,
    such as a/000001.npy when the tag "a" has that step file. Every tag below
    it moves to the first free "~N" of that part, the same in every tab.

    Attributes:
        renamed: by tag, the path with a suffix and the reason for it.
    """

    renamed: dict[str, tuple[tuple[str, ...], str]]
    paths: dict[str, tuple[str, ...]]

    def __init__(
        self,
        tags: Iterable[str],
        blocked: Mapping[tuple[str, ...], str],
        name_map: dict[str, str],
    ) -> None:
        """Resolve the path of every tag.

        Args:
            tags: the tags of the run.
            blocked: the directories that are files, with the path of the file.
        """
        self.renamed = {}
        self.paths = {tag: _Paths.tag_dir_parts(tag, name_map=name_map) for tag in tags}
        if not blocked:
            return
        taken = {
            parts[:end]
            for parts in self.paths.values()
            for end in range(1, len(parts) + 1)
        }
        moves: dict[tuple[str, ...], tuple[str, ...]] = {}
        for tag, natural in self.paths.items():
            parts, reason = natural, ""
            for end in range(1, len(parts) + 1):
                if parts[:end] in blocked:
                    reason = f"{blocked[parts[:end]]!r} is a file"
                    if parts[:end] not in moves:
                        moves[parts[:end]] = TagPaths.free(
                            parts[:end], taken, blocked, name_map
                        )
                        taken.add(moves[parts[:end]])
                    parts = (*moves[parts[:end]], *parts[end:])
            if parts != natural:
                self.renamed[tag] = (parts, reason)
                self.paths[tag] = parts

    @staticmethod
    def free(
        prefix: tuple[str, ...],
        taken: set[tuple[str, ...]],
        blocked: Mapping[tuple[str, ...], str],
        name_map: dict[str, str],
    ) -> tuple[str, ...]:
        """Return the prefix with the first "~N" on its last part that is free."""
        number = 2
        while True:
            suffix = f"~{number}"
            name = _Paths.fit(
                prefix[-1] + suffix,
                _Paths.decode_component(prefix[-1], name_map=name_map),
                name_map,
            )
            moved = (*prefix[:-1], name)
            if moved not in taken and moved not in blocked:
                return moved
            number += 1

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

    def dir_parts(self, tag: str) -> tuple[str, ...]:
        """Return the virtual directory parts of a tag."""
        return self.paths[tag]
