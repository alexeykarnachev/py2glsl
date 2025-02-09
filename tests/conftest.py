import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test files."""
    return tmp_path
