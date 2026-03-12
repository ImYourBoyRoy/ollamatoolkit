# ./tests/test_cli_models.py
"""
CLI tests for lightweight OllamaToolkit model inspection behavior.
Run with `python -m pytest -q tests/test_cli_models.py`.
Inputs: monkeypatched inventory helpers and isolated CLI imports.
Outputs: assertions proving `models` works without run-only config or vision tool imports.
Side effects: manipulates temporary module import state only within the test process.
Operational notes: protects the lazy-import contract for operator model inspection commands.
"""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from types import ModuleType
from typing import Any


def test_models_command_returns_json_without_run_config(monkeypatch, capsys) -> None:
    from ollamatoolkit import cli

    def _fake_collect(**_: Any) -> dict[str, Any]:
        return {
            "base_url": "http://localhost:11434",
            "total_models": 1,
            "models": [
                {
                    "name": "qwen3:8b",
                    "family": "qwen3",
                    "size": "8B",
                    "capabilities": ["completion", "reasoning"],
                    "running": True,
                }
            ],
            "running_models": [{"name": "qwen3:8b"}],
            "recommended": {"chat": "qwen3:8b"},
            "benchmark_summary": {},
            "summary": {"by_capability": {"completion": ["qwen3:8b"]}},
        }

    monkeypatch.setattr(
        "ollamatoolkit.models.inventory.collect_model_inventory",
        _fake_collect,
    )
    exit_code = cli.main(["models", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0]["name"] == "qwen3:8b"


def test_importing_cli_does_not_require_vision_module(monkeypatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("ollamatoolkit.tools.vision"):
            raise AssertionError("vision module should not be imported for CLI import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    sys.modules.pop("ollamatoolkit.cli", None)
    module = importlib.import_module("ollamatoolkit.cli")
    assert isinstance(module, ModuleType)
    assert hasattr(module, "main")
