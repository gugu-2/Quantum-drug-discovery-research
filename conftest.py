"""
conftest.py — shared pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run with --slow flag, deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "hardware: marks tests that require IBM Quantum hardware access"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks end-to-end integration tests"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests (full VQE benchmarks, multi-molecule screening)",
    )
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run tests on real IBM Quantum hardware (requires IBM token in config.py)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Slow test — add --slow flag to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--hardware"):
        skip_hw = pytest.mark.skip(reason="Hardware test — add --hardware flag and set IBM token")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hw)
