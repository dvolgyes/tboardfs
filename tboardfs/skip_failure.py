"""Skip one item that fails to convert, and keep the items beside it."""

from collections.abc import Callable
from contextvars import ContextVar

from loguru import logger

_speculative_record = ContextVar("speculative_record", default=False)


class _ReplayRecordError(Exception):
    """A speculative conversion needs the serial diagnostic path."""


def _warn_skipped(message: str, *args: object) -> None:
    """Defer skip diagnostics during speculative record conversion."""
    if _speculative_record.get():
        raise _ReplayRecordError(message.format(*args))
    logger.warning(message, *args)


def skip_failure[Result](
    what: str, convert: Callable[[], Result], *, strict: bool = False
) -> Result | None:
    """Return convert(), or None and a warning when it raises.

    Each converter guards the malformed input that it knows, and its warning
    names the cause. This boundary catches the failures that those guards
    miss, so that one record or tag hides only its own files. Without it, a
    failure in the tree that each file system call builds fails every call
    of the mount. MemoryError, and errors that are not an Exception, pass.

    Args:
        what: the item, as the warning names it, such as "tensor t at step 5"
        convert: the conversion of that item
        strict: let the error pass, as a strict parse does
    """
    try:
        return convert()
    except MemoryError:
        raise
    except Exception as error:
        if _speculative_record.get():
            if isinstance(error, _ReplayRecordError):
                raise
            raise _ReplayRecordError(
                f"{what} is skipped: {type(error).__name__}: {error}"
            ) from error
        if strict:
            raise
        logger.warning("{} is skipped: {}: {}", what, type(error).__name__, error)
        logger.opt(exception=error).debug("traceback of the skipped {}", what)
        return None
