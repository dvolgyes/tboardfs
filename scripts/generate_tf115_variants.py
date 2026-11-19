#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = [
#   "tensorflow-cpu==1.15.0", "tensorboard==1.15.0", "numpy==1.18.5",
#   "protobuf==3.20.3", "setuptools==67.8.0", "werkzeug==1.0.1",
# ]
# ///
"""Generate the native TF/TB 1.15 historical variant corpus."""

from historical_variants import main

if __name__ == "__main__":
    main()
