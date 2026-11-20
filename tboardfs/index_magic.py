"""Observe the libmagic rule generation used by the existing classifier."""

import ctypes
from dataclasses import dataclass, field
from importlib import import_module
from itertools import islice
import locale
import os
from pathlib import Path
import stat
import sys
import time
from typing import ClassVar, cast

_MAX_PATHS = 128
_FILE_LOAD = 0

OBSERVATION_ERRORS = (
    AttributeError,
    locale.Error,
    NotImplementedError,
    OSError,
    RuntimeError,
    ValueError,
)


class _DlInfo(ctypes.Structure):
    """Shared-object information returned by dladdr."""

    _fields_: ClassVar[list[tuple[str, type]]] = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


class _Loader:
    """Query the loaded library instead of assuming file(1)'s configuration."""

    @staticmethod
    def path() -> Path:
        """Return the path of the mapped libmagic object."""
        magic = import_module("magic")
        symbol = ctypes.cast(magic.libmagic["magic_getpath"], ctypes.c_void_p)
        dladdr = ctypes.CDLL(None)["dladdr"]
        dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
        dladdr.restype = ctypes.c_int
        info = _DlInfo()
        if not dladdr(symbol, ctypes.byref(info)) or not info.filename:
            raise ValueError("unknown libmagic shared object")
        path = Path(os.fsdecode(info.filename))
        if not path.is_absolute() or symbol.value is None:
            raise ValueError("unobservable libmagic shared object")
        if sys.platform.startswith("linux") and not _Loader._same_mapping(
            path, symbol.value
        ):
            raise ValueError("libmagic shared object has changed since loading")
        return path

    @staticmethod
    def _same_mapping(path: Path, address: int) -> bool:
        """Check that an atomically replaced file is not mistaken for the mapped inode."""
        status = path.stat()
        with Path("/proc/self/maps").open(
            encoding="ascii", errors="surrogateescape"
        ) as maps:
            for line in maps:
                span, _, _, _, inode, *mapped_path = line.split(maxsplit=5)
                low, high = (int(value, 16) for value in span.split("-"))
                if low <= address < high:
                    return _Loader._same_inode(
                        path, status.st_ino, int(inode), mapped_path
                    )
        return False

    @staticmethod
    def _same_inode(
        path: Path, inode: int, mapped_inode: int, mapped_path: list[str]
    ) -> bool:
        """Compare canonical paths and inodes; btrfs stat and maps devices can differ."""
        return (
            bool(mapped_path)
            and inode == mapped_inode
            and str(path.resolve()) == mapped_path[0].rstrip("\n")
        )

    @staticmethod
    def rules() -> tuple[Path, ...]:
        """Ask the loaded library for its ordered POSIX rule path list."""
        if os.name != "posix":
            raise ValueError("unsupported libmagic lookup platform")
        magic = import_module("magic")
        call = magic.libmagic["magic_getpath"]
        call.argtypes = (ctypes.c_char_p, ctypes.c_int)
        call.restype = ctypes.c_char_p
        raw = call(None, _FILE_LOAD)
        if not raw:
            raise ValueError("no libmagic rules path")
        parts = raw.split(b":", _MAX_PATHS)
        paths = tuple(Path(os.fsdecode(part)) for part in parts)
        if len(paths) > _MAX_PATHS or not all(
            path.is_absolute() and os.fsencode(path) == part
            for path, part in zip(paths, parts, strict=True)
        ):
            raise ValueError("unobservable libmagic rules path")
        return paths


@dataclass(slots=True)
class _RuleFiles:
    """Collect bounded file-backed inputs, including missing lookup candidates."""

    files: dict[Path, tuple[int, ...]] = field(default_factory=dict)

    @staticmethod
    def signature(status: os.stat_result | None) -> tuple[int, ...]:
        """Describe the file generation independently of its access time."""
        if status is None:
            return ()
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @staticmethod
    def status(path: Path) -> os.stat_result | None:
        """Follow native lookup semantics for ordinary symlinks and missing files."""
        try:
            return path.stat()
        except FileNotFoundError:
            return None

    def record(self, path: Path) -> os.stat_result | None:
        """Watch one path only if its contents or absence are observable."""
        if len(self.files) >= _MAX_PATHS:
            raise ValueError("too many libmagic rules paths")
        status = self.status(path)
        if status is not None and not (
            stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
        ):
            raise ValueError("libmagic rules are not regular files")
        self.files[path] = self.signature(status)
        return status

    def record_home(self) -> None:
        """Watch .magic.mgc, .magic and the directory form of its database."""
        home = os.environ.get("HOME")
        if home is None or "MAGIC" in os.environ:
            return
        path = Path(home)
        if not path.is_absolute():
            raise ValueError("unobservable libmagic home directory")
        self.record(path / ".magic.mgc")
        status = self.record(path / ".magic")
        if status is not None and stat.S_ISDIR(status.st_mode):
            self.record(path / ".magic" / "magic.mgc")

    def record_source(self, path: Path) -> bool:
        """Watch a text fallback or all bounded entries of its rule directory."""
        status = self.record(path)
        if status is None:
            return False
        if stat.S_ISREG(status.st_mode):
            return True
        with os.scandir(path) as entries:
            children = sorted(entry.name for entry in islice(entries, _MAX_PATHS + 1))
        present = False
        for child in children:
            present = self.record_regular(path / child) or present
        return present

    def record_regular(self, path: Path) -> bool:
        """Watch a potential compiled database or individual text rule file."""
        status = self.record(path)
        if status is None:
            return False
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("unobservable libmagic rules file")
        return True

    def record_database(self, path: Path) -> bool:
        """Watch MIME-specific and generic compiled rules and their text fallback."""
        stem = str(path).removesuffix(".mgc")
        mime = self.record_regular(Path(f"{stem}.mime.mgc"))
        compiled = self.record_regular(Path(f"{stem}.mgc"))
        text = self.record_source(path)
        return mime or compiled or text


@dataclass(frozen=True, slots=True)
class MagicHost:
    """Describe file-backed native rules and the environment under which they load."""

    environment: tuple[object, ...]
    files: tuple[tuple[Path, tuple[int, ...]], ...]

    @staticmethod
    def _environment() -> tuple[object, ...]:
        """Capture settings that affect classifier lookup and output."""
        keys = (
            "HOME",
            "MAGIC",
            "LANG",
            "LANGUAGE",
            "LOCPATH",
            "TZ",
            "TZDIR",
            "POSIXLY_CORRECT",
            "NPY_DISABLE_CPU_FEATURES",
        )
        return (
            tuple((key, os.environ.get(key)) for key in keys),
            tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("LC_"))),
            locale.setlocale(locale.LC_ALL),
            time.tzname,
            time.timezone,
            time.altzone,
        )

    @classmethod
    def observe(cls) -> "MagicHost":
        """Observe native lookup only during opt-in, before the mount starts."""
        magic = import_module("magic")
        environment = cls._environment()
        rules = _RuleFiles()
        rules.record(_Loader.path())
        rules.record(Path(cast(str, magic.__file__)))
        rules.record_home()
        present = False
        for path in _Loader.rules():
            present = rules.record_database(path) or present
        if not present:
            raise ValueError("no file-backed libmagic rules")
        return cls(environment, tuple(rules.files.items()))

    def unchanged(self) -> bool:
        """Check live process compatibility without touching rule files."""
        magic = import_module("magic")
        try:
            return self.environment == self._environment() and all(
                getattr(instance, "_tboardfs_magic_environment", None)
                == self.environment
                for instance in tuple(magic._instances.values())
            )
        except OBSERVATION_ERRORS:
            return False

    def files_unchanged(self) -> bool:
        """Check files during pre-mount capture, not on FUSE or worker paths."""
        if not self.unchanged():
            return False
        magic = import_module("magic")
        cookie_times = [
            int(getattr(instance, "_tboardfs_magic_loaded_after", 0))
            for instance in tuple(magic._instances.values())
        ]
        cutoff = min(cookie_times, default=time.time_ns())
        return all(
            saved == _RuleFiles.signature(_RuleFiles.status(path))
            and (not saved or saved[-1] < cutoff)
            for path, saved in self.files
        )

    @staticmethod
    def signature(status: os.stat_result) -> tuple[int, ...]:
        """Use the same generation identity during a bounded policy digest pass."""
        return _RuleFiles.signature(status)

    @classmethod
    def classify(cls, blob: bytes, *, mime: bool = False) -> str:
        """Use python-magic normally, recording its first load without filesystem I/O."""
        magic = import_module("magic")
        if not hasattr(magic, "_instances") or mime in magic._instances:
            return cast(str, magic.from_buffer(blob, mime=mime))
        environment = cls._environment()
        started = time.time_ns()
        try:
            return cast(str, magic.from_buffer(blob, mime=mime))
        finally:
            instance = magic._instances.get(mime)
            if instance is not None and environment == cls._environment():
                instance._tboardfs_magic_environment = environment
                instance._tboardfs_magic_loaded_after = started
