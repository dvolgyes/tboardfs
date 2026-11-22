/* Fresh no-follow checks for the mount's guarded sibling snapshot. */
#include <Python.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>

#if !defined(__linux__) || !defined(HAVE_FSTATAT) || !defined(__SIZEOF_INT128__) || \
    PY_VERSION_HEX < 0x030C0000 || defined(PYPY_VERSION) || defined(Py_GIL_DISABLED)
#error "The optional sibling stat accelerator requires supported Linux CPython headers"
#endif

static int
long_long_at(PyObject *tuple, Py_ssize_t index, long long *value)
{
    PyObject *item = PyTuple_GET_ITEM(tuple, index);
    if (!PyLong_CheckExact(item)) {
        return 0;
    }
    int overflow;
    *value = PyLong_AsLongLongAndOverflow(item, &overflow);
    return !overflow;
}

static int
unsigned_long_long_at(PyObject *tuple, Py_ssize_t index, unsigned long long *value)
{
    PyObject *item = PyTuple_GET_ITEM(tuple, index);
    if (!PyLong_CheckExact(item)) {
        return 0;
    }
    *value = PyLong_AsUnsignedLongLong(item);
    if (*value == ULLONG_MAX && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static __int128
nanoseconds(struct timespec timestamp)
{
    return (__int128)timestamp.tv_sec * 1000000000 + timestamp.tv_nsec;
}

static PyObject *
matches(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    (void)self;
    if (nargs != 2 || !PyLong_CheckExact(args[0]) || !PyTuple_CheckExact(args[1])) {
        Py_RETURN_FALSE;
    }
    int overflow;
    long fd = PyLong_AsLongAndOverflow(args[0], &overflow);
    if (overflow || fd < 0 || fd > INT_MAX || PyTuple_GET_SIZE(args[1]) == 0) {
        Py_RETURN_FALSE;
    }
    PyObject *files = args[1];
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(files); i++) {
        PyObject *row = PyTuple_GET_ITEM(files, i);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
            Py_RETURN_FALSE;
        }
        PyObject *name = PyTuple_GET_ITEM(row, 0);
        PyObject *fingerprint = PyTuple_GET_ITEM(row, 1);
        if (!PyBytes_CheckExact(name) || !PyTuple_CheckExact(fingerprint) ||
            PyTuple_GET_SIZE(fingerprint) != 6 || PyTuple_GET_ITEM(fingerprint, 2) != Py_None) {
            Py_RETURN_FALSE;
        }
        const char *path = PyBytes_AS_STRING(name);
        Py_ssize_t len = PyBytes_GET_SIZE(name);
        if (!len || memchr(path, '/', len) || memchr(path, '\0', len) ||
            (path[0] == '.' && (len == 1 || (len == 2 && path[1] == '.')))) {
            Py_RETURN_FALSE;
        }
        unsigned long long dev, ino;
        long long size, mtime_ns, ctime_ns;
        if (!unsigned_long_long_at(fingerprint, 0, &dev) ||
            !unsigned_long_long_at(fingerprint, 1, &ino) ||
            !long_long_at(fingerprint, 3, &size) ||
            !long_long_at(fingerprint, 4, &mtime_ns) ||
            !long_long_at(fingerprint, 5, &ctime_ns)) {
            Py_RETURN_FALSE;
        }
        struct stat status;
        int result;
        Py_BEGIN_ALLOW_THREADS
        result = fstatat((int)fd, path, &status, AT_SYMLINK_NOFOLLOW);
        Py_END_ALLOW_THREADS
        if (result != 0 || !S_ISREG(status.st_mode) || status.st_dev != dev ||
            status.st_ino != ino || status.st_size != size ||
            nanoseconds(status.st_mtim) != mtime_ns || nanoseconds(status.st_ctim) != ctime_ns) {
            Py_RETURN_FALSE;
        }
    }
    Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {
    {"matches", _PyCFunction_CAST(matches), METH_FASTCALL,
     "Compare live regular-file fingerprints through a caller-owned guarded directory fd."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_sibling_stat",
    .m_size = 0,
    .m_methods = methods,
};

PyMODINIT_FUNC
PyInit__sibling_stat(void)
{
    return PyModule_Create(&module);
}
