"""Find the embedding files that a projector_config.pbtxt names."""

import os
import re
from pathlib import Path, PurePosixPath

CONFIG_NAME = "projector_config.pbtxt"
MAX_CONFIG_BYTES = 1 << 20
_PATH_FIELD = re.compile(
    rb'\b(?:tensor_path|metadata_path|image_path|bookmarks_path)\s*:\s*"([^"\\\n]*)"'
)


class _ProjectorConfig:
    """Map projector config paths to sidecar files of the same run.

    PyTorch and tensorboardX add_embedding write tensors, metadata, and
    sprites to <step>/<tag>/ and name them in projector_config.pbtxt,
    relative to the config, as TensorBoard reads them. A name selects only
    among the candidate files; it never opens a path of its own.
    """

    @staticmethod
    def references(candidates: list[Path]) -> dict[Path, tuple[Path, tuple[str, ...]]]:
        """Return named file -> (config, path parts relative to the config)."""
        known = set(candidates)
        found: dict[Path, tuple[Path, tuple[str, ...]]] = {}
        for config in candidates:
            if config.name != CONFIG_NAME:
                continue
            for parts in _ProjectorConfig.named_parts(config):
                path = config.parent.joinpath(*parts)
                if path in known and path.name != CONFIG_NAME:
                    found.setdefault(path, (config, parts))
        return found

    @staticmethod
    def named_parts(config: Path) -> list[tuple[str, ...]]:
        """Return the relative paths of one config.

        Absolute and empty paths are left out, and escaped strings are left
        out instead of decoded. A ".." part never matches a candidate,
        because joined paths are compared without normalization.
        """
        try:
            with config.open("rb") as handle:
                text = handle.read(MAX_CONFIG_BYTES)
        except FileNotFoundError:
            return []  # removed since the directory walk
        named = []
        for match in _PATH_FIELD.finditer(text):
            path = PurePosixPath(os.fsdecode(match.group(1)))
            if path.parts and not path.is_absolute():
                named.append(path.parts)
        return named
