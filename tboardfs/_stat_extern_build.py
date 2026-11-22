"""Generate the optional CFFI extern-Python entrypoints for FUSE 2 stat calls."""

from cffi import FFI

ffi = FFI()
ffi.cdef(
    """
    struct stat;
    struct fuse_file_info;
    extern "Python" int stat_getattr(const char *, struct stat *);
    extern "Python" int stat_fgetattr(const char *, struct stat *, struct fuse_file_info *);
    """
)
ffi.set_source(
    "tboardfs._stat_extern",
    """
    struct stat;
    struct fuse_file_info;
    """,
    optional=True,
)
