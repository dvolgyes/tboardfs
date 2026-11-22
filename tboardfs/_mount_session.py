"""Run the standard TensorBoardFS mount with native FUSE 2 stat entry points."""

import errno
import logging
from collections.abc import Callable
from ctypes import c_char_p, pointer, sizeof
from functools import partial
from signal import SIG_DFL, SIGINT, getsignal, signal
from threading import current_thread, main_thread

import mfusepy

from tboardfs._mount_callbacks import StatCallbacks
from tboardfs.mount_binding import MountBinding
from tboardfs.mounted import MountedTensorBoardFS

_LOGGER = logging.getLogger("fuse")
_MAX_KERNEL_ERRNO = 4095

_CALLBACKS = (
    ("open", "open"),
    ("read", "read"),
    ("release", "release"),
    ("readdir", "readdir_fuse_2"),
    ("mknod", "mknod"),
    ("mkdir", "mkdir"),
    ("unlink", "unlink"),
    ("rmdir", "rmdir"),
    ("symlink", "symlink"),
    ("rename", "rename_fuse_2"),
    ("chmod", "chmod_fuse_2"),
    ("chown", "chown_fuse_2"),
    ("truncate", "truncate_fuse_2"),
    ("ftruncate", "ftruncate"),
    ("create", "create"),
    ("write", "write"),
)


class _Fuse2Session(MountBinding):
    """Reuse mfusepy's argument marshaling with a mount-local error boundary."""

    operations: MountedTensorBoardFS
    raw_fi: bool = True
    encoding: str = "utf-8"
    errors: str = "surrogateescape"
    use_ns: bool = False
    _fatal: tuple[BaseException, list[BaseException]] | None

    def __init__(self, operations: MountedTensorBoardFS) -> None:
        self.operations = operations
        self._prepare_stat()
        self._fatal = None

    def _invoke(self, method: Callable[..., int | None], *args: object) -> int:
        """Keep exceptions from escaping through a native callback."""
        try:
            return method(*args) or 0
        except Exception as error:  # noqa: BLE001 - native callback boundary
            try:
                return self._errno(method.__name__, error)
            except BaseException as logging_error:  # noqa: BLE001 - defer to mount thread
                return self._abort(method.__name__, logging_error)
        except BaseException as error:  # noqa: BLE001 - defer to mount thread
            return self._abort(method.__name__, error)

    @staticmethod
    def _errno(operation: str, error: Exception) -> int:
        """Translate a recoverable failure to a Linux kernel error reply."""
        if (
            isinstance(error, OSError)
            and isinstance(code := error.errno, int)
            and 0 < code <= _MAX_KERNEL_ERRNO
        ):
            _LOGGER.debug(
                "FUSE operation %s returned errno %s.",
                operation,
                code,
                exc_info=not (operation == "fgetattr" and code == errno.ENOENT),
            )
            return -code
        _LOGGER.exception("Uncaught exception from FUSE operation %s.", operation)
        return -errno.EINVAL

    def _abort(self, operation: str, error: BaseException) -> int:
        """Request shutdown and keep secondary failures for the mounting thread."""
        if self._fatal is None:
            self._fatal = (error, [])
        _, failures = self._fatal
        try:
            _LOGGER.critical(
                "Uncaught critical exception from FUSE operation %s, aborting.",
                operation,
                exc_info=error,
            )
        except BaseException as log_error:  # noqa: BLE001 - defer to mount thread
            failures.append(log_error)
        try:
            mfusepy.fuse_exit()
        except BaseException as exit_error:  # noqa: BLE001 - defer to mount thread
            failures.append(exit_error)
        return -errno.EFAULT

    def _run(
        self, callbacks: StatCallbacks, mountpoint: str, foreground: bool, options: str
    ) -> None:
        """Pin the callback owner until libfuse returns; init returns the same handle."""
        fuse_ops = mfusepy.fuse_operations()
        prototypes = {name: prototype for name, prototype, *_ in fuse_ops._fields_}
        for name, method in _CALLBACKS:
            callback = partial(self._invoke, getattr(self, method))
            setattr(fuse_ops, name, prototypes[name](callback))
        fuse_ops.getattr = callbacks.getattr
        fuse_ops.fgetattr = callbacks.fgetattr
        private_data = callbacks.user_data
        fuse_ops.init = prototypes["init"](lambda _conn: private_data.value)

        args = ["fuse"]
        if foreground:
            args.append("-f")
        args.extend(("-o", options, mountpoint))
        encoded = [arg.encode(self.encoding, self.errors) for arg in args]
        argv = (c_char_p * len(encoded))(*encoded)
        old_handler = signal(SIGINT, SIG_DFL)
        try:
            status = mfusepy.fuse_main_real(
                len(encoded), argv, pointer(fuse_ops), sizeof(fuse_ops), private_data
            )
        finally:
            signal(SIGINT, old_handler)
        if self._fatal is not None:
            error, failures = self._fatal
            if failures:
                raise error from BaseExceptionGroup("FUSE abort failures", failures)
            raise error
        if status:
            raise RuntimeError(status)


def mount(
    operations: MountedTensorBoardFS,
    mountpoint: str,
    *,
    foreground: bool,
    ro: bool,
    raw_fi: bool,
    attr_timeout: int,
) -> None:
    """Use native stat slots only for the audited CLI mount."""
    options = {"ro": ro, "attr_timeout": attr_timeout}
    if (
        ro is True
        and raw_fi is True
        and type(attr_timeout) is int
        and attr_timeout == 0
        and type(foreground) is bool
        and type(mountpoint) is str
        and type(operations) is MountedTensorBoardFS
        and _try_mount(operations, mountpoint, foreground, options)
    ):
        return
    MountBinding(
        operations, mountpoint, foreground=foreground, raw_fi=raw_fi, **options
    )


def _try_mount(
    operations: MountedTensorBoardFS,
    mountpoint: str,
    foreground: bool,
    options: dict[str, bool | int],
) -> bool:
    """Decline custom operations and modes that cannot preserve the SIGINT handler."""
    if current_thread() is not main_thread() or getsignal(SIGINT) is None:
        return False
    custom_fields = (
        "use_ns",
        "init_with_config",
        "readdir_with_offset",
        *(name for name, *_ in mfusepy.fuse_operations._fields_),
    )
    attributes = vars(operations)
    if any(name in attributes for name in custom_fields):
        return False
    session = _Fuse2Session(operations)
    callbacks = StatCallbacks.try_create(partial(session._invoke, session.fgetattr))
    if callbacks is None:
        return False
    fuse_options = session._normalize_fuse_options(
        **options, fsname=type(operations).__name__
    )
    session._run(callbacks, mountpoint, foreground, ",".join(fuse_options))
    return True
