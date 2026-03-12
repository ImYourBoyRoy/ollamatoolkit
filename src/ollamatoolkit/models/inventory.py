# ./src/ollamatoolkit/models/inventory.py
"""
Inventory and benchmark helpers for OllamaToolkit model inspection.
Run via imports from the CLI or other operator tooling; no standalone CLI entrypoint.
Inputs: Ollama base URL, optional capability filters, and optional benchmark report files.
Outputs: normalized installed/running model inventory plus capability and benchmark summaries.
Side effects: performs lightweight Ollama API reads and optional local benchmark file reads.
Operational notes: designed to avoid optional vision/system tool imports so inventory commands stay lightweight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import ollama

from .selector import ModelSelector, TaskType


def _load_benchmark_summary(benchmarks_file: str | Path | None) -> Dict[str, Any]:
    if not benchmarks_file:
        return {}
    path = Path(benchmarks_file)
    if not path.exists():
        return {"path": str(path), "available": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "available": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"path": str(path), "available": False, "error": "invalid payload"}

    live_rows = []
    live_section = payload.get("live_model_benchmarks", {})
    if isinstance(live_section, Mapping):
        live_rows = [
            dict(row)
            for row in live_section.get("results", [])
            if isinstance(row, Mapping)
        ]

    by_role: Dict[str, Dict[str, Any]] = {}
    for row in live_rows:
        role = str(row.get("role", "") or "")
        if not role:
            continue
        by_role[role] = {
            "model": str(row.get("model", "") or ""),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "error": row.get("error"),
        }
    return {
        "path": str(path),
        "available": True,
        "roles": by_role,
        "artifact_summary": payload.get("artifact_report", {}).get("summary", {})
        if isinstance(payload.get("artifact_report", {}), Mapping)
        else {},
        "decision_summary": payload.get("decision_report", {}).get("summary", {})
        if isinstance(payload.get("decision_report", {}), Mapping)
        else {},
    }


def _running_models(client: ollama.Client) -> List[Dict[str, Any]]:
    try:
        response = client.ps()
    except Exception:
        return []
    models = response.models if hasattr(response, "models") else []
    rows: List[Dict[str, Any]] = []
    for item in models:
        name = getattr(item, "model", None) or getattr(item, "name", None) or ""
        if not name:
            continue
        rows.append(
            {
                "name": str(name),
                "size_vram": getattr(item, "size_vram", 0),
                "expires_at": getattr(item, "expires_at", None),
                "details": getattr(item, "details", None),
            }
        )
    return rows


def collect_model_inventory(
    *,
    base_url: str,
    capability: Optional[str] = None,
    benchmarks_file: str | Path | None = None,
) -> Dict[str, Any]:
    """Collect installed/running model info plus optional benchmark summaries."""
    selector = ModelSelector(base_url=base_url)
    running_rows = _running_models(selector.client)
    running_names = {str(row.get("name", "") or "") for row in running_rows}

    if capability:
        model_infos = selector.get_models_by_capability(capability)
    else:
        model_infos = [
            info
            for name in selector.model_names
            if (info := selector.get_model(name)) is not None
        ]

    models: List[Dict[str, Any]] = []
    for info in model_infos:
        models.append(
            {
                "name": info.name,
                "family": info.family,
                "size": info.parameter_size,
                "parameter_count": info.parameter_count,
                "capabilities": list(info.capabilities),
                "quantization": info.quantization,
                "context_length": info.context_length,
                "running": info.name in running_names,
            }
        )

    summary = selector.summary()
    benchmark_summary = _load_benchmark_summary(benchmarks_file)
    return {
        "base_url": base_url,
        "capability_filter": capability,
        "total_models": len(models),
        "models": models,
        "running_models": running_rows,
        "summary": summary,
        "recommended": {
            "chat": selector.get_best_chat_model(),
            "embedding": selector.get_best_embedding_model(),
            "vision": selector.get_best_vision_model(),
            "reasoning": selector.get_for_task(TaskType.REASONING, prefer_small=False),
        },
        "benchmark_summary": benchmark_summary,
    }
