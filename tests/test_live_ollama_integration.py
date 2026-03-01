# ./tests/test_live_ollama_integration.py
"""
Opt-in live integration tests against a real Ollama server.
Run: `OLLAMA_TEST_BASE_URL=http://host:11434 python -m pytest tests/test_live_ollama_integration.py -m integration`.
Inputs: environment variable `OLLAMA_TEST_BASE_URL` and installed remote models.
Outputs: validates version/tags plus lightweight generate/embed calls.
Side effects: performs live network/API calls to the configured Ollama server.
Operational notes: skipped automatically when `OLLAMA_TEST_BASE_URL` is not set.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterable, List

import pytest

from ollamatoolkit.client import AsyncOllamaClient, OllamaClient

pytestmark = pytest.mark.integration

_GENERATION_CANDIDATES = [
    "tinyllama:latest",
    "qwen3:8b",
    "llama3.1:latest",
    "ministral-3:latest",
    "mistral-small3.2:latest",
]
_EMBEDDING_CANDIDATES = [
    "qwen3-embedding:4b",
    "nomic-embed-text-v2-moe:latest",
    "mxbai-embed-large:latest",
]
_TOOLS_CANDIDATES = [
    "qwen3:8b",
    "qwen3:14b",
    "gpt-oss:20b",
    "llama3.1:latest",
]
_VISION_CANDIDATES = [
    "qwen3-vl:latest",
    "deepseek-ocr:latest",
    "mistral-small3.2:latest",
]

_VISION_SAMPLE_PATH = Path(__file__).parents[1] / "test_input_samples" / "image.jpeg"


def _require_live_base_url() -> str:
    base_url = os.getenv("OLLAMA_TEST_BASE_URL", "").strip()
    if not base_url:
        pytest.skip("Set OLLAMA_TEST_BASE_URL to run live Ollama integration tests.")
    return base_url


def _pick_model(available_models: List[str], preferred: Iterable[str]) -> str | None:
    available_set = set(available_models)
    for model in preferred:
        if model in available_set:
            return model
    return available_models[0] if available_models else None


def _get_model_names(client: OllamaClient) -> List[str]:
    return [name for name in (m.name or m.model for m in client.list().models) if name]


def test_live_server_version_and_list_models() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=30.0) as client:
        version = client.version()
        models = client.list()

    assert version.version, "Version response should include a semantic version string."
    assert len(models.models) > 0, (
        "Live server should have at least one installed model."
    )


def test_live_generate_roundtrip() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=120.0) as client:
        model_names = _get_model_names(client)
        chat_model = _pick_model(model_names, _GENERATION_CANDIDATES)
        if not chat_model:
            pytest.skip("No generation model available on live server.")

        response = client.generate(
            chat_model,
            "Reply with exactly one word: ACK",
            options={"temperature": 0.0, "num_predict": 8},
        )

    assert response.response
    assert len(response.response.strip()) > 0


def test_live_embed_roundtrip() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=120.0) as client:
        model_names = _get_model_names(client)
        embed_model = _pick_model(model_names, _EMBEDDING_CANDIDATES)
        if not embed_model:
            pytest.skip("No embedding model available on live server.")

        result = client.embed(embed_model, "B2X lead scoring context")

    assert len(result.embeddings) >= 1
    assert len(result.embeddings[0]) > 0


def test_live_generate_streaming_roundtrip() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=180.0) as client:
        model_names = _get_model_names(client)
        chat_model = _pick_model(model_names, _GENERATION_CANDIDATES)
        if not chat_model:
            pytest.skip("No generation model available on live server.")

        chunks = list(
            client.generate(
                chat_model,
                "Reply with exactly: STREAM_OK",
                stream=True,
                options={"temperature": 0.0, "num_predict": 16},
            )
        )

    assert len(chunks) > 0
    joined = "".join(chunk.response for chunk in chunks if chunk.response)
    assert joined.strip()


def test_live_model_management_read_endpoints() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=120.0) as client:
        version = client.version()
        model_names = _get_model_names(client)
        if not model_names:
            pytest.skip("No models available on live server.")

        show_response = client.show(model_names[0])
        running_response = client.ps()

    assert version.version
    assert show_response is not None
    assert running_response is not None


def test_live_tool_call_roundtrip_openai_compat() -> None:
    base_url = _require_live_base_url()
    with OllamaClient(host=base_url, timeout=180.0) as client:
        model_names = _get_model_names(client)
        tools_model = _pick_model(model_names, _TOOLS_CANDIDATES)
        if not tools_model:
            pytest.skip("No likely tool-capable model available on live server.")

        response = client.openai_chat_completions(
            model=tools_model,
            messages=[
                {
                    "role": "user",
                    "content": "Use the lookup_company tool for company 'B2X' and nothing else.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_company",
                        "description": "Lookup company details.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                            },
                            "required": ["company"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "lookup_company"}},
            parallel_tool_calls=True,
            temperature=0.0,
            max_tokens=64,
        )

        tool_calls = (
            response.choices[0].message.tool_calls
            if response.choices and response.choices[0].message
            else None
        )
        if not tool_calls:
            fallback = client.chat(
                tools_model,
                messages=[
                    {
                        "role": "user",
                        "content": "Call the lookup_company tool for B2X.",
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_company",
                            "description": "Lookup company details.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string"},
                                },
                                "required": ["company"],
                            },
                        },
                    }
                ],
                options={"temperature": 0.0, "num_predict": 96},
            )
            if not fallback.message.tool_calls:
                pytest.skip(
                    "Selected live model did not emit tool calls in this environment."
                )
            tool_calls = fallback.message.tool_calls

    assert response.choices
    message = response.choices[0].message
    assert message is not None
    assert tool_calls is not None and len(tool_calls) > 0
    assert tool_calls[0].function.name == "lookup_company"


def test_live_vision_chat_roundtrip() -> None:
    base_url = _require_live_base_url()
    if not _VISION_SAMPLE_PATH.exists():
        pytest.skip(f"Vision sample not found: {_VISION_SAMPLE_PATH}")

    encoded_image = base64.b64encode(_VISION_SAMPLE_PATH.read_bytes()).decode()
    with OllamaClient(host=base_url, timeout=180.0) as client:
        model_names = _get_model_names(client)
        vision_model = _pick_model(model_names, _VISION_CANDIDATES)
        if not vision_model:
            pytest.skip("No likely vision model available on live server.")

        response = client.chat(
            vision_model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe the image in one short sentence.",
                    "images": [encoded_image],
                }
            ],
            options={"temperature": 0.0, "num_predict": 64},
        )

    content = (response.message.content or "").strip()
    thinking = (response.message.thinking or "").strip()
    assert content or thinking


@pytest.mark.asyncio
async def test_live_async_chat_streaming_roundtrip() -> None:
    base_url = _require_live_base_url()
    async with AsyncOllamaClient(host=base_url, timeout=180.0) as client:
        model_names = [
            name
            for name in (m.name or m.model for m in (await client.list()).models)
            if name
        ]
        chat_model = _pick_model(model_names, _GENERATION_CANDIDATES)
        if not chat_model:
            pytest.skip("No generation model available on live server.")

        stream = await client.chat(
            chat_model,
            messages=[{"role": "user", "content": "Reply with ASYNC_OK only"}],
            stream=True,
            options={"temperature": 0.0, "num_predict": 32},
        )
        chunks = [chunk async for chunk in stream]

    assert len(chunks) > 0
    content = "".join(chunk.message.content or "" for chunk in chunks)
    assert content.strip()
