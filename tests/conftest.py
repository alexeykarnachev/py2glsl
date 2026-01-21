"""Fixtures and configuration for pytest."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--generate",
        action="store_true",
        default=False,
        help="Generate/update gold outputs and validate GLSL compiles",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring a GPU")
    config.addinivalue_line("markers", "backend: mark test as a backend test")


def pytest_collection_modifyitems(config, items):
    """Skip gold tests when generating gold outputs."""
    if config.getoption("--generate"):
        skip_marker = pytest.mark.skip(reason="Generating gold outputs")
        for item in items:
            if "test_gold" in item.nodeid and "test_shader" in item.name:
                item.add_marker(skip_marker)


def pytest_sessionfinish(session, exitstatus):
    """Generate gold outputs and validate GLSL if --generate flag is set."""
    if session.config.getoption("--generate", default=False):
        from tests.test_gold import generate_gold_outputs, validate_gold_glsl

        generate_gold_outputs()
        success = validate_gold_glsl()
        if not success:
            session.exitstatus = 1
