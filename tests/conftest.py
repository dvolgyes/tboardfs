"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_event_file():
    """Return path to a test TensorBoard event file."""
    # Use pre-generated test file
    return "tests/example-data/full_log/events.out.tfevents.1748727850.FG-OSL-WS122.7152.0.v2"


@pytest.fixture
def minimal_event_file():
    """Return path to a minimal test TensorBoard event file."""
    # Use pre-generated minimal test file
    return "tests/example-data/minimal_log/events.out.tfevents.1748727851.FG-OSL-WS122.7152.1.v2"
