"""Give each tag of one run its own virtual path."""

from collections.abc import Iterable
from typing import Self

from loguru import logger

from tboardfs.mesh import _MeshFiles
from tboardfs.model import JsonEntry, RunCache
from tboardfs.paths import _Paths


class TagPaths:
    """Give each tag of one run its own virtual path.

    Different tags can map to the same path, such as "" and "untagged", "a/b"
    and "a//b", or "\\" and "%5C". A tag that its path spells exactly keeps
    that path. Each other tag whose path is in use gets "~2", "~3", ... on its
    last part, in sorted tag order, so every tag keeps its own files.

    Attributes:
        renamed: by tag, the path with a suffix and the tag that has its path.
    """

    renamed: dict[str, tuple[tuple[str, ...], str]]

    def __init__(self, tags: Iterable[str]) -> None:
        self.renamed = {}
        holders: dict[tuple[str, ...], str] = {}
        for tag in sorted(set(tags), key=lambda tag: (not TagPaths.spelled(tag), tag)):
            parts = path = _Paths.tag_dir_parts(tag)
            number = 1
            while path in holders:
                number += 1
                path = (*parts[:-1], f"{parts[-1]}~{number}")
            holders[path] = tag
            if path != parts:
                self.renamed[tag] = (path, holders[parts])

    @classmethod
    def for_run(cls, run: RunCache, scalar_tags: Iterable[str]) -> Self:
        """Return the tag paths of one run, and log each tag that gets a suffix.

        The run keeps its suffixed tags, so a run that grows logs a collision
        once, and again only when its path changes.
        """
        caches = list(run.files.values())
        tags = cls(
            [
                *scalar_tags,
                *(tag for cache in caches for tag in cache.histograms),
                *(entry.tag for cache in caches for entry in cache.binaries),
                *(
                    tag
                    for cache in caches
                    for entry in cache.json_entries
                    if (tag := cls.entry_tag(entry)) is not None
                ),
            ]
        )
        for tag, (path, holder) in tags.renamed.items():
            if run.renamed_tags.get(tag) != path:
                logger.warning(
                    "tags {!r} and {!r} have the same virtual path in run {!r}; "
                    "the files of {!r} are under {!r}",
                    holder,
                    tag,
                    "/".join(run.run_parts),
                    tag,
                    "/".join(path),
                )
        run.renamed_tags = {tag: path for tag, (path, _) in tags.renamed.items()}
        return tags

    @staticmethod
    def entry_tag(entry: JsonEntry) -> str | None:
        """Return the tag whose path holds the files of a plugin record.

        A mesh component has the path of its mesh tag, and another record of
        the meshes tab has no path.
        """
        if entry.tab == "meshes":
            return _MeshFiles.tag(entry)
        return entry.tag

    @staticmethod
    def spelled(tag: str) -> bool:
        """Return true when the virtual path of a tag spells the tag exactly."""
        return "/".join(_Paths.tag_dir_parts(tag)) == tag

    def dir_parts(self, tag: str) -> tuple[str, ...]:
        """Return the virtual directory parts of a tag."""
        if tag in self.renamed:
            return self.renamed[tag][0]
        return _Paths.tag_dir_parts(tag)

    def file_parts(self, tag: str, fmt: str) -> tuple[str, ...]:
        """Return virtual path parts for a tagged file with an extension."""
        stem_parts = self.dir_parts(tag)
        return (*stem_parts[:-1], f"{stem_parts[-1]}.{fmt}")
