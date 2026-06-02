from mfusepy import FuseOSError


def fuse_error(errno: int) -> Exception:
    """Create the FUSE-compatible exception for an errno value."""
    error: Exception = FuseOSError(errno)
    return error
