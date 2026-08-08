"""Shared paths and deterministic helpers for layout/geometry tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root independently of the process working directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def reticle_dir(project_root: Path) -> Path:
    """Return the checked-in GDS regression fixture directory."""
    return project_root / "TestReticle"
