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

#ifdef __has_include
#if __has_include(<linux/openat2.h>) && __has_include(<sys/syscall.h>)
#include <linux/openat2.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(SYS_openat2) && defined(RESOLVE_NO_SYMLINKS) && defined(O_PATH)
#define TBOARDFS_HAVE_OPENAT2 1
#endif
#endif
#endif

typedef struct {
    const char *path;
    unsigned long long dev, ino;
    long long size, mtime_ns, ctime_ns;
} SiblingRow;

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
        if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
            PyErr_Clear();
        }
        return 0;
    }
    return 1;
}

static __int128
nanoseconds(struct timespec timestamp)
{
    return (__int128)timestamp.tv_sec * 1000000000 + timestamp.tv_nsec;
}

static int
parse_row(PyObject *row, SiblingRow *expected)
{
    if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
        return 0;
    }
    PyObject *name = PyTuple_GET_ITEM(row, 0);
    PyObject *fingerprint = PyTuple_GET_ITEM(row, 1);
    if (!PyBytes_CheckExact(name) || !PyTuple_CheckExact(fingerprint) ||
        PyTuple_GET_SIZE(fingerprint) != 6 || PyTuple_GET_ITEM(fingerprint, 2) != Py_None) {
        return 0;
    }
    const char *path = PyBytes_AS_STRING(name);
    Py_ssize_t len = PyBytes_GET_SIZE(name);
    if (!len || memchr(path, '/', len) || memchr(path, '\0', len) ||
        (path[0] == '.' && (len == 1 || (len == 2 && path[1] == '.')))) {
        return 0;
    }
    expected->path = path;
    return unsigned_long_long_at(fingerprint, 0, &expected->dev) &&
           unsigned_long_long_at(fingerprint, 1, &expected->ino) &&
           long_long_at(fingerprint, 3, &expected->size) &&
           long_long_at(fingerprint, 4, &expected->mtime_ns) &&
           long_long_at(fingerprint, 5, &expected->ctime_ns);
}

static int
match_row(int fd, const SiblingRow *expected)
{
    struct stat status;
    return fstatat(fd, expected->path, &status, AT_SYMLINK_NOFOLLOW) == 0 &&
           S_ISREG(status.st_mode) && status.st_dev == expected->dev &&
           status.st_ino == expected->ino && status.st_size == expected->size &&
           nanoseconds(status.st_mtim) == expected->mtime_ns &&
           nanoseconds(status.st_ctim) == expected->ctime_ns;
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
        SiblingRow expected;
        if (!parse_row(PyTuple_GET_ITEM(files, i), &expected)) {
            if (PyErr_Occurred()) {
                return NULL;
            }
            Py_RETURN_FALSE;
        }
        int result;
        Py_BEGIN_ALLOW_THREADS
        result = match_row((int)fd, &expected);
        Py_END_ALLOW_THREADS
        if (!result) {
            Py_RETURN_FALSE;
        }
    }
    Py_RETURN_TRUE;
}

#ifdef TBOARDFS_HAVE_OPENAT2
static const char *
absolute_path(PyObject *name)
{
    if (!PyBytes_CheckExact(name)) {
        return NULL;
    }
    const char *path = PyBytes_AS_STRING(name);
    Py_ssize_t len = PyBytes_GET_SIZE(name);
    if (!len || path[0] != '/' || memchr(path, '\0', len)) {
        return NULL;
    }
    return path;
}

static int
guarded_open(const char *path, int directory)
{
    const struct open_how how = {
        .flags = O_PATH | O_CLOEXEC | (directory ? O_DIRECTORY : 0),
        .resolve = RESOLVE_NO_SYMLINKS,
    };
    return (int)syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof(how));
}

static PyObject *
open_directory(PyObject *self, PyObject *name)
{
    (void)self;
    const char *path = absolute_path(name);
    int fd = -1;
    if (path != NULL) {
        Py_BEGIN_ALLOW_THREADS
        fd = guarded_open(path, 1);
        Py_END_ALLOW_THREADS
    }
    PyObject *result = PyLong_FromLong(fd);
    if (result == NULL && fd >= 0) {
        Py_BEGIN_ALLOW_THREADS
        close(fd);
        Py_END_ALLOW_THREADS
    }
    return result;
}

static PyObject *
plain_path(PyObject *self, PyObject *name)
{
    (void)self;
    const char *path = absolute_path(name);
    if (path == NULL) {
        Py_RETURN_FALSE;
    }
    int fd;
    Py_BEGIN_ALLOW_THREADS
    fd = guarded_open(path, 0);
    if (fd >= 0) {
        close(fd);
    }
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(fd >= 0);
}

typedef struct {
    PyObject_VAR_HEAD
    PyObject *parent;
    PyObject *files;
    const char *path;
    SiblingRow rows[];
} PreparedSiblings;

static void
prepared_siblings_dealloc(PyObject *self)
{
    PreparedSiblings *prepared = (PreparedSiblings *)self;
    Py_DECREF(prepared->files);
    Py_DECREF(prepared->parent);
    PyObject_Del(self);
}

static PyObject *
prepared_siblings_call(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwargs != NULL && PyDict_GET_SIZE(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "prepared sibling checker takes no arguments");
        return NULL;
    }
    const PreparedSiblings *prepared = (PreparedSiblings *)self;
    Py_ssize_t count = Py_SIZE(self);
    int matches_all = 0;
    Py_BEGIN_ALLOW_THREADS
    int fd = guarded_open(prepared->path, 1);
    if (fd >= 0) {
        matches_all = 1;
        for (Py_ssize_t i = 0; i < count; i++) {
            if (!match_row(fd, &prepared->rows[i])) {
                matches_all = 0;
                break;
            }
        }
        close(fd);
    }
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(matches_all);
}

static PyTypeObject prepared_siblings_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tboardfs._sibling_stat._PreparedSiblings",
    .tp_basicsize = sizeof(PreparedSiblings),
    .tp_itemsize = sizeof(SiblingRow),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .tp_dealloc = prepared_siblings_dealloc,
    .tp_call = prepared_siblings_call,
};

static PyObject *
prepare_siblings(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    (void)self;
    if (nargs != 2 || !PyTuple_CheckExact(args[1])) {
        Py_RETURN_NONE;
    }
    const char *path = absolute_path(args[0]);
    Py_ssize_t count = PyTuple_GET_SIZE(args[1]);
    if (path == NULL || count == 0 ||
        (size_t)count > (PY_SSIZE_T_MAX - sizeof(PreparedSiblings)) / sizeof(SiblingRow)) {
        Py_RETURN_NONE;
    }
    PreparedSiblings *prepared = PyObject_NewVar(PreparedSiblings, &prepared_siblings_type, count);
    if (prepared == NULL) {
        return NULL;
    }
    prepared->parent = Py_NewRef(args[0]);
    prepared->files = Py_NewRef(args[1]);
    prepared->path = path;
    for (Py_ssize_t i = 0; i < count; i++) {
        if (!parse_row(PyTuple_GET_ITEM(prepared->files, i), &prepared->rows[i])) {
            Py_DECREF(prepared);
            if (PyErr_Occurred()) {
                return NULL;
            }
            Py_RETURN_NONE;
        }
    }
    return (PyObject *)prepared;
}
#endif

static PyMethodDef methods[] = {
    {"matches", _PyCFunction_CAST(matches), METH_FASTCALL,
     "Compare live regular-file fingerprints through a caller-owned guarded directory fd."},
#ifdef TBOARDFS_HAVE_OPENAT2
    {"open_directory", open_directory, METH_O, NULL},
    {"plain_path", plain_path, METH_O, NULL},
    {"prepare_siblings", _PyCFunction_CAST(prepare_siblings), METH_FASTCALL, NULL},
#endif
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
#ifdef TBOARDFS_HAVE_OPENAT2
    if (PyType_Ready(&prepared_siblings_type) < 0) {
        return NULL;
    }
#endif
    return PyModule_Create(&module);
}
