# ./tests/test_version.py
"""
Tests for package version resolution and pyproject.toml consistency.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

from ollamatoolkit import __version__
from ollamatoolkit._version import resolve_version
from ollamatoolkit.client_api.common import PACKAGE_VERSION


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


class TestVersionResolution:
    """Ensure version strings stay aligned with pyproject.toml."""

    def test_public_version_matches_pyproject(self) -> None:
        assert __version__ == _pyproject_version()

    def test_resolve_version_matches_pyproject(self) -> None:
        assert resolve_version() == _pyproject_version()

    def test_user_agent_version_matches_pyproject(self) -> None:
        assert PACKAGE_VERSION == _pyproject_version()
