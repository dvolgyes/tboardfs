import importlib

from tboardfs.constants import FIXED_TABS
from tboardfs.filesystem import TensorBoardFS
from tboardfs.indexer import parse_file
from tboardfs.paths import find_tensorboard_files

__all__ = (
    "FIXED_TABS",
    "TensorBoardFS",
    "find_tensorboard_files",
    "main",
    "parse_file",
)


def __getattr__(name: str) -> object:
    """Import the mount command when it is used: mfusepy needs libfuse."""
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    main: object = importlib.import_module("tboardfs.cli").main
    return main
