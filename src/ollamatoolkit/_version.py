# ./src/ollamatoolkit/_version.py
"""
Resolve the package version from installed distribution metadata or pyproject.toml.
Run: imported by `ollamatoolkit.__init__` and client User-Agent helpers.
Inputs: optional installed `roy-ollama-toolkit` / `ollamatoolkit` distribution.
Outputs: semantic version string used as the single source of truth.
Side effects: none.
Operational notes: pyproject.toml `project.version` is authoritative for editable installs.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def resolve_version() -> str:
    """Return the package version from installed metadata or local pyproject.toml."""
    for distribution_name in ("roy-ollama-toolkit", "ollamatoolkit"):
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            version = project.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
        except Exception:
            pass

    return "0.0.0"


__version__ = resolve_version()
