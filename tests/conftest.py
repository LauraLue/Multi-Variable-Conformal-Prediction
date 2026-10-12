"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest

PLOT_DIR = Path(__file__).parent / "plots"


@pytest.fixture
def plot_dir(tmp_path: Path) -> Path:
    """Directory for diagnostic plots written by the tests.

    A temporary directory by default, so test runs leave no files behind.
    Set MCP_TEST_PLOTS=1 to keep the plots in tests/plots/ for inspection.
    """
    if os.environ.get("MCP_TEST_PLOTS"):
        PLOT_DIR.mkdir(exist_ok=True)
        return PLOT_DIR
    return tmp_path
