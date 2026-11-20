"""Skip one item that fails to convert, and keep the items beside it."""

from collections.abc import Callable
from contextvars import ContextVar

from loguru import logger

from tboardfs.model.parse_progress import _ParseProvenance

_speculative_record = ContextVar("speculative_record", default=False)
_diagnostic_provenance: ContextVar[_ParseProvenance | None] = ContextVar(
    "diagnostic_provenance", default=None
)


class _ReplayRecordError(Exception):
    """A speculative conversion needs the serial diagnostic path."""


def _warn_skipped(message: str, *args: object) -> None:
    """Request serial replay or mark the index non-cacheable and emit the warning."""
    if _speculative_record.get():
        raise _ReplayRecordError(message.format(*args))
    provenance = _diagnostic_provenance.get()
    if provenance is not None:
        provenance.cacheable = False
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
    except Exception as error:
        if not _skip_error(error, what, strict=strict):
            raise
        return None


def _skip_error(error: Exception, what: str, *, strict: bool) -> bool:
    """Log a skipped item, request serial replay, or let the caller propagate the error."""
    if isinstance(error, MemoryError):
        return False
    if _speculative_record.get():
        if isinstance(error, _ReplayRecordError):
            return False
        raise _ReplayRecordError(
            f"{what} is skipped: {type(error).__name__}: {error}"
        ) from error
    if strict:
        return False
    _warn_skipped("{} is skipped: {}: {}", what, type(error).__name__, error)
    logger.opt(exception=error).debug("traceback of the skipped {}", what)
    return True
