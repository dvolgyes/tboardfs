#!/usr/bin/env python3
"""Simple test runner to verify the tests work."""

import subprocess
import sys


def run_tests():
    """Run pytest with appropriate options."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",  # verbose
        "--tb=short",  # short traceback format
        "tests/",
    ]

    print("Running tests...")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
