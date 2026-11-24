"""Time a Linux FUSE mount through the first indexed run listing."""

import argparse
import ctypes
import json
import mmap
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter_ns, sleep
from types import MappingProxyType

_DEFAULT_CORPUS = Path("~/workspace/benchmarks/tboardfs/cold-mount")
_RUNS = MappingProxyType({"mixed": "run", "lightning": "lightning-4g"})
_PARALLEL_MIN_BYTES = 8 << 20


class _Source:
    """Validate the fixed workflow layout, source identity, and cache state."""

    @staticmethod
    def event(log: Path, run: str) -> Path:
        """Accept only a single direct run and event file."""
        folder = log / run
        if log.is_symlink() or list(log.iterdir()) != [folder] or folder.is_symlink():
            raise ValueError(f"{log}: expected only the run directory {run}")
        files = list(folder.iterdir())
        if (
            len(files) != 1
            or not files[0].name.startswith("events.out.tfevents.")
            or files[0].is_symlink()
            or not files[0].is_file()
        ):
            raise ValueError(
                f"{folder}: expected only one regular events.out.tfevents.* file"
            )
        return files[0]

    @staticmethod
    def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
        """Identify the file and detect changes other than access time."""
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @classmethod
    def check(cls, event: Path, expected: os.stat_result) -> None:
        """Check the layout and file identity against the initial snapshot."""
        if cls.event(event.parent.parent, event.parent.name) != event or cls._identity(
            event.stat()
        ) != cls._identity(expected):
            raise RuntimeError(f"{event}: benchmark source changed")

    @classmethod
    def evict(cls, event: Path, expected: os.stat_result) -> None:
        """Evict only the source and inspect its mapping without faulting pages."""
        size = expected.st_size
        libc = ctypes.CDLL(None, use_errno=True)
        mincore = libc.mincore
        mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
        page_size = mmap.PAGESIZE
        pages = (size + page_size - 1) // page_size
        resident = (ctypes.c_ubyte * pages)()
        with event.open("rb") as stream:
            fd = stream.fileno()
            if cls._identity(os.fstat(fd)) != cls._identity(expected):
                raise RuntimeError(f"{event}: benchmark source changed before eviction")
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                if mincore(address, size, resident) != 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error), event)
        count = sum(page & 1 for page in resident)
        if count:
            raise RuntimeError(f"{event}: {count}/{pages} source pages still resident")


class _Mount:
    """Own a native foreground FUSE mount and observe its first listing."""

    work: Path
    mount: Path
    fusermount: str

    _interrupted: int

    def __init__(self, work: Path, fusermount: str) -> None:
        self.work = work
        self.mount = work / "mount"
        self.fusermount = fusermount
        self._interrupted = 0

    def _defer(self, signum: int, _frame: object) -> None:
        """Remember a signal received before the child process handle is available."""
        self._interrupted = signum

    def _fuse_mounted(self) -> bool:
        """Check the kernel mount table without issuing a filesystem callback."""
        mount = self.mount
        encoded = os.fsencode(mount)
        escapes = (
            (b"\\", b"\\134"),
            (b" ", b"\\040"),
            (b"\t", b"\\011"),
            (b"\n", b"\\012"),
        )
        for char, escaped in escapes:
            encoded = encoded.replace(char, escaped)
        for row in Path("/proc/self/mountinfo").read_bytes().splitlines():
            if row.split()[4] == encoded:
                kind = row.split(b" - ", 1)[1].split()[0]
                if not kind.startswith(b"fuse"):
                    raise RuntimeError(f"{mount}: not a FUSE mount ({kind!r})")
                return True
        return False

    @staticmethod
    def _deadline(_signum: int, _frame: object) -> None:
        """Abort a timed mount or listing."""
        raise TimeoutError("FUSE startup or first indexed run listing timed out")

    @staticmethod
    def _interrupt(signum: int, _frame: object) -> None:
        """Interrupt a foreground trial."""
        raise KeyboardInterrupt(signal.Signals(signum).name)

    def _stop(self, process: subprocess.Popen[bytes]) -> None:
        """Detach and reap the mount, including a mount that races interruption."""
        try:
            if self._fuse_mounted():
                unmount = self._unmount(lazy=False)
                if unmount.returncode:
                    self._unmount(lazy=True).check_returncode()
                process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if self._fuse_mounted():
                self._unmount(lazy=True).check_returncode()

    def _unmount(self, *, lazy: bool) -> subprocess.CompletedProcess[bytes]:
        """Invoke the system FUSE helper with fixed flags and our private mount path."""
        return subprocess.run(  # noqa: S603
            [self.fusermount, "-uz" if lazy else "-u", str(self.mount)],
            check=False,
            timeout=10,
        )

    @contextmanager
    def foreground(
        self, command: list[str]
    ) -> Iterator[tuple[subprocess.Popen[bytes], int]]:
        """Launch the selected executable with fixed flags and a private mount path."""
        env = {**os.environ, "XDG_CACHE_HOME": str(self.work / "cache")}
        signals = (signal.SIGINT, signal.SIGTERM)
        process = None

        with (self.work / "mount.log").open("wb") as output:
            handlers = {sig: signal.signal(sig, self._defer) for sig in signals}
            try:
                start = perf_counter_ns()
                process = subprocess.Popen(  # noqa: S603
                    command,
                    env=env,
                    stdout=output,
                    stderr=output,
                    start_new_session=True,
                )
                for sig in signals:
                    signal.signal(sig, self._interrupt)
                if self._interrupted:
                    self._interrupt(self._interrupted, None)
                yield process, start
            finally:
                for sig in signals:
                    signal.signal(sig, signal.SIG_IGN)
                try:
                    if process is not None:
                        self._stop(process)
                    if self._fuse_mounted():
                        raise RuntimeError(f"FUSE mount not detached: {self.mount}")
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)

    def timing(
        self, process: subprocess.Popen[bytes], start: int, run: str, timeout: int
    ) -> dict[str, int]:
        """Measure mount readiness and the first root and indexed run listings."""
        previous = signal.signal(signal.SIGALRM, self._deadline)
        signal.alarm(timeout)
        try:
            while not self._fuse_mounted():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"FUSE exited before mounting ({process.returncode})"
                    )
                sleep(0.003)
            ready = perf_counter_ns()
            roots = [path.name for path in self.mount.iterdir()]
            before_list = perf_counter_ns()
            tabs = [path.name for path in (self.mount / run).iterdir()]
            end = perf_counter_ns()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        if (
            roots != [run]
            or "scalars" not in tabs
            or "raw" in tabs
            or process.poll() is not None
        ):
            raise RuntimeError(
                f"unexpected typed-only FUSE listing: root={roots!r}, run={tabs!r}"
            )
        return {
            "elapsed_ns": end - start,
            "launch_to_ready_ns": ready - start,
            "root_listing_ns": before_list - ready,
            "list_run_ns": end - before_list,
        }


class _Trial:
    """Measure repeatable cold mounts against one unchanged source snapshot."""

    event: Path
    expected: os.stat_result
    launcher: Path
    fusermount: str

    def __init__(
        self, event: Path, expected: os.stat_result, launcher: Path, fusermount: str
    ) -> None:
        self.event = event
        self.expected = expected
        self.launcher = launcher
        self.fusermount = fusermount

    def _check_index(self, work: Path, workers: int) -> int:
        """Require a published index and return the observed parallel range count."""
        event = self.event
        indexes = [
            path
            for path in (work / "cache" / "tboardfs" / "indexes-v1").glob("index-*.idx")
            if re.fullmatch(r"index-[0-9a-f]{32}\.idx", path.name)
        ]
        if (
            len(indexes) != 1
            or not indexes[0].is_file()
            or not indexes[0].stat().st_size
        ):
            raise RuntimeError(f"{event}: persistent index was not written")
        diagnostic = (
            rf"parallel cold index of {re.escape(str(event))} used (\d+) ranges"
        )
        matches = re.findall(
            diagnostic, (work / "mount.log").read_text(errors="replace")
        )
        ranges = int(matches[0]) if len(matches) == 1 else 0
        if workers > 1 and self.expected.st_size >= _PARALLEL_MIN_BYTES and ranges < 2:
            raise RuntimeError(
                f"{event}: no successful parallel index with {workers} workers"
            )
        return ranges

    def measure(self, workers: int, timeout: int) -> dict[str, int]:
        """Measure one cold mount and validate its outputs before reporting."""
        event, expected = self.event, self.expected
        workflow = event.parent.parent.parent
        if workflow.stat().st_dev != expected.st_dev:
            raise ValueError(
                f"{event}: run and trial workspace must be on the same filesystem"
            )
        _Source.check(event, expected)
        work = Path(mkdtemp(prefix=".tboardfs-cold-mount-", dir=workflow))
        fuse = _Mount(work, self.fusermount)
        command = [
            str(self.launcher),
            str(event.parent.parent),
            str(fuse.mount),
            "--foreground",
            "--disk-index",
            "--index-workers",
            str(workers),
            "--loglevel",
            "DEBUG",
        ]
        try:
            fuse.mount.mkdir()
            _Source.evict(event, expected)
            with fuse.foreground(command) as (process, start):
                timings = fuse.timing(process, start, event.parent.name, timeout)
            _Source.check(event, expected)
            timings["parallel_ranges"] = self._check_index(work, workers)
        except (Exception, KeyboardInterrupt):
            sys.stderr.write(f"Trial artifacts retained in {work} (see mount.log).\n")
            raise
        shutil.rmtree(work)
        return timings


def main() -> None:
    """Emit one JSON line per cold mount, labeled by workflow and worker count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=_RUNS)
    parser.add_argument("-d", "--corpus-root", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("-j", "--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("-r", "--repeat", type=int, default=4)
    parser.add_argument(
        "-t", "--timeout", type=int, default=240, help="trial timeout in seconds"
    )
    parser.add_argument(
        "-l",
        "--launcher",
        type=Path,
        help="tboardfs executable (default: beside Python)",
    )
    args = parser.parse_args()
    if min(*args.workers, args.repeat, args.timeout) < 1:
        parser.error("workers, repeat and timeout must be positive")
    fusermount = shutil.which("fusermount3", path=os.defpath) or shutil.which(
        "fusermount", path=os.defpath
    )
    if (
        sys.platform != "linux"
        or fusermount is None
        or not os.access("/dev/fuse", os.R_OK | os.W_OK)
    ):
        parser.error("requires Linux with accessible /dev/fuse and fusermount")
    launcher = args.launcher or Path(sys.executable).with_name("tboardfs")
    launcher = launcher.expanduser().absolute()
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        parser.error(f"not an executable file: {launcher}")
    workflow_dir = (args.corpus_root.expanduser() / args.workflow).resolve()
    log = workflow_dir / "log"
    try:
        event = _Source.event(log, _RUNS[args.workflow])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    source_stat = event.stat()
    size = source_stat.st_size
    if not size:
        parser.error(f"empty event file: {event}")
    if size >= _PARALLEL_MIN_BYTES and max(args.workers) > len(os.sched_getaffinity(0)):
        parser.error(
            "worker count exceeds available CPUs for a parallel-eligible source"
        )
    benchmark = _Trial(event, source_stat, launcher, fusermount)
    for trial in range(1, args.repeat + 1):
        order = args.workers if trial % 2 else reversed(args.workers)
        for workers in order:
            timing = benchmark.measure(workers, args.timeout)
            row = {
                "workflow": args.workflow,
                "source": str(event),
                "source_bytes": size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_ctime_ns": source_stat.st_ctime_ns,
                "launcher": str(launcher),
                "trial": trial,
                "workers": workers,
                **timing,
            }
            sys.stdout.write(json.dumps(row) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
