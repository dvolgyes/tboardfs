"""Skip one item that fails to convert, and keep the items beside it."""

from collections.abc import Callable

from loguru import logger


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
        if strict:
            raise
        logger.warning("{} is skipped: {}: {}", what, type(error).__name__, error)
        logger.opt(exception=error).debug("traceback of the skipped {}", what)
        return None
