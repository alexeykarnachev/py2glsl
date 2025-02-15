import sys

import pytest
from loguru import logger


def pytest_addoption(parser):
    parser.addoption(
        "--log-debug",
        action="store_true",
        default=False,
        help="Set logging level to DEBUG",
    )


@pytest.fixture(autouse=True)
def setup_logging(request):
    log_level = "DEBUG" if request.config.getoption("--log-debug") else "INFO"

    logger.remove()
    logger.add(sys.stderr, level=log_level)
