"""Configure the optional Linux CPython accelerators at build time."""

import os
import platform
import sys
import sysconfig
from typing import Any

from setuptools import Extension, setup

if __name__ == "__main__":
    extensions = []
    options: dict[str, Any] = {}
    native_build = (
        os.environ.get("TBOARDFS_NATIVE") == "1"
        and sys.platform == "linux"
        and sys.implementation.name == "cpython"
        and os.environ.get("TBOARDFS_PURE_PYTHON") != "1"
    )
    if native_build:
        extensions.append(
            Extension(
                "tboardfs._sibling_stat",
                ["tboardfs/_sibling_stat.c"],
                optional=True,
            )
        )
    if (
        native_build
        and platform.machine() == "x86_64"
        and sys.maxsize == 2**63 - 1
        and not sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        options["cffi_modules"] = ["tboardfs/_stat_extern_build.py:ffi"]
    setup(ext_modules=extensions, **options)
