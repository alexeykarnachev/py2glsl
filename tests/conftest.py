"""Fixtures and configuration for pytest."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring a GPU")
    config.addinivalue_line("markers", "backend: mark test as a backend test")
