#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["puremagic==2.3.0", "tboardfs"]
# [tool.uv.sources]
# tboardfs = { path = "..", editable = true }
# ///
"""Compare tboardfs suffixes with puremagic guesses for one event file, as JSON Lines."""

import argparse
import json
import sys
from pathlib import Path

import puremagic

from tboardfs.extract import extract_binary_blob
from tboardfs.indexer import parse_file
from tboardfs.model import BinaryEntry


def _report(entry: BinaryEntry) -> str:
    """Compare the indexed suffix against ranked content matches without a filename hint."""
    data = extract_binary_blob(entry)
    matches = puremagic.magic_string(data) if data else []
    return json.dumps(
        {
            "tag": entry.tag,
            "step": entry.step,
            "part": entry.part,
            "tab": entry.kind,
            "tboardfs": f".{entry.extension}",
            "puremagic": list(
                dict.fromkeys(m.extension for m in matches if m.extension)
            ),
        }
    )


def main() -> None:
    """Print one comparison per indexed payload in the supplied fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_file", type=Path)
    path = parser.parse_args().event_file
    if not path.is_file():
        parser.error(f"not a regular event file: {path}")
    for entry in parse_file(path)["binary_entries"]:
        if entry.kind != "graphs":
            sys.stdout.write(_report(entry) + "\n")


if __name__ == "__main__":
    main()
