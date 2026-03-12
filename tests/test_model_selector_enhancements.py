# ./tests/test_model_selector_enhancements.py
"""
Regression tests for OllamaToolkit model selector capability normalization and chat suitability.
Run with `python -m pytest -q tests/test_model_selector_enhancements.py`.
Inputs: synthetic in-memory model catalogs only.
Outputs: assertions for reasoning alias handling and chat-model recommendation quality.
Side effects: none.
Operational notes: avoids live Ollama calls by constructing selector state directly.
"""

from __future__ import annotations

from ollamatoolkit.models.selector import ModelInfo, ModelSelector


def _selector_with_models(*models: ModelInfo) -> ModelSelector:
    selector = ModelSelector.__new__(ModelSelector)
    selector.base_url = "http://localhost:11434"
    selector.client = None
    selector._models = {model.name: model for model in models}
    selector._last_refresh = None
    return selector


def test_reasoning_alias_matches_thinking_capability() -> None:
    selector = _selector_with_models(
        ModelInfo(
            name="qwen3:14b",
            family="qwen3",
            parameter_size="14.8B",
            parameter_count=14_768_307_200,
            capabilities=["completion", "tools", "thinking", "reasoning"],
            quantization="Q4_K_M",
            context_length=40960,
        )
    )
    models = selector.get_models_by_capability("reasoning")
    assert [model.name for model in models] == ["qwen3:14b"]


def test_get_best_chat_model_skips_reranker_like_models() -> None:
    selector = _selector_with_models(
        ModelInfo(
            name="B-A-M-N/qwen3-reranker-0.6b-fp16:latest",
            family="qwen3",
            parameter_size="595.78M",
            parameter_count=595_776_512,
            capabilities=["completion", "tools"],
            quantization="F16",
            context_length=40960,
        ),
        ModelInfo(
            name="qwen3:8b",
            family="qwen3",
            parameter_size="8.2B",
            parameter_count=8_190_735_360,
            capabilities=["completion", "tools", "thinking", "reasoning"],
            quantization="Q4_K_M",
            context_length=40960,
        ),
    )
    assert selector.get_best_chat_model() == "qwen3:8b"
