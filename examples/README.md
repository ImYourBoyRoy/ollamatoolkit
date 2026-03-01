# OllamaToolkit Examples

This directory contains working examples demonstrating key features of OllamaToolkit.

## Quick Start

```bash
# Install
pip install ollamatoolkit

# Run any example
python examples/01_basic_chat.py
```

## Examples Index

| File | Description |
|------|-------------|
| `01_basic_chat.py` | Simple agent chat |
| `02_with_tools.py` | Agent with function calling |
| `03_vision_ocr.py` | PDF/image extraction |
| `04_rag_pipeline.py` | Vector search + retrieval |
| `05_multi_agent_team.py` | AgentTeam orchestration |
| `06_streaming_ui.py` | Real-time token streaming |
| `07_schema_cache.py` | Schema validation + caching |
| `08_email_tools.py` | Email validation + extraction |
| `09_tool_registry.py` | LLM-first tool integration |

## Prerequisites

- Python 3.10+
- Ollama running locally (`ollama serve`)
- At least one model installed (`ollama pull llama3.1`)
