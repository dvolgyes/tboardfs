import os


def fuse_error(errno: int) -> OSError:
    """Create the exception for an errno value; mfusepy returns it as -errno."""
    return OSError(errno, os.strerror(errno))
