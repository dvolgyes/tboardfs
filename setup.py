"""Configure the optional Linux CPython accelerator at build time."""

import os
import sys

from setuptools import Extension, setup

if __name__ == "__main__":
    extensions = []
    if (
        sys.platform == "linux"
        and sys.implementation.name == "cpython"
        and os.environ.get("TBOARDFS_PURE_PYTHON") != "1"
    ):
        extensions.append(
            Extension(
                "tboardfs._sibling_stat",
                ["tboardfs/_sibling_stat.c"],
                optional=True,
            )
        )
    setup(ext_modules=extensions)
