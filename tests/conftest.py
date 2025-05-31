"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from tboardfs.test_generator import (
    generate_test_tensorboard_log,
    generate_minimal_test_log,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_event_file(temp_dir):
    """Generate a test TensorBoard event file with all data types."""
    log_dir = temp_dir / "test_log"
    event_file = generate_test_tensorboard_log(str(log_dir), num_iterations=11)
    yield event_file


@pytest.fixture
def minimal_event_file(temp_dir):
    """Generate a minimal test TensorBoard event file."""
    log_dir = temp_dir / "minimal_log"
    event_file = generate_minimal_test_log(str(log_dir))
    yield event_file
