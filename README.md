# OllamaToolkit

## Use Case Synopsis
`ollamatoolkit` is a modular, pip-installable toolkit for building reliable local-AI and hybrid-AI systems on top of Ollama. It is designed for:
- **agentic workflows** (single agent + multi-agent teams),
- **programmatic automation** (sync/async clients, typed responses, streaming events),
- **tool orchestration** (web, files, vector/RAG, DB, system, email, schema, vision), and
- **integration into larger systems** (research pipelines, MCP-connected tools, telemetry-enabled apps).

If your goal is “make smaller local models perform consistently through good scaffolding,” this package is built for exactly that.

---

## Installation

```bash
# Core
pip install roy-ollama-toolkit

# Full extras (system + file/image + email tooling)
pip install "roy-ollama-toolkit[full]"
```

Supported Python versions: **3.10–3.13**.
Import path remains `ollamatoolkit` after installation.

---

## Integration Guide

### 1) Standalone usage (direct Python)

```python
from ollamatoolkit.client import OllamaClient

with OllamaClient(base_url="http://localhost:11434") as client:
    reply = client.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Give me a 1-line business summary."}],
    )
    print(reply.message.content)
```

### 2) Agentic usage (SimpleAgent + tools)

```python
from ollamatoolkit.agents.simple import SimpleAgent

agent = SimpleAgent(
    name="research-assistant",
    system_message="You are concise, evidence-focused, and tool-using.",
    model_config={"model": "ollama/qwen3:8b", "base_url": "http://localhost:11434"},
)

@agent.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print(agent.run("What is 19 + 23?"))
```

### 3) Agentic + MCP tool servers (stdio bridge)

`ollamatoolkit.tools.mcp` connects to MCP servers via **command + args** (stdio JSON-RPC).

```python
from ollamatoolkit.tools.mcp import MCPToolManager
from ollamatoolkit.agents.simple import SimpleAgent

mcp = MCPToolManager(
    {
        "web": {
            "command": "python",
            "args": ["-m", "web_scraper_toolkit.server.mcp_server"],
        }
    }
)
mcp.start_all()

agent = SimpleAgent(
    name="mcp-agent",
    system_message="Use tools when helpful.",
    model_config={"model": "ollama/qwen3:8b", "base_url": "http://localhost:11434"},
    tools=mcp.get_tool_schemas(),
    function_map=mcp.get_proxy_functions(),
)

print(agent.run("Find contact info for example.com and summarize it."))
mcp.shutdown()
```

> For remote MCP hosting, run the server on the remote machine and use a transport command that exposes stdio locally (for example via SSH command invocation).

---

## Quick API Starts

### Sync + Async parity

```python
import asyncio
from ollamatoolkit.client import OllamaClient, AsyncOllamaClient

with OllamaClient() as client:
    print(client.version())

async def main() -> None:
    async with AsyncOllamaClient() as client:
        print(await client.version())

asyncio.run(main())
```

### Stream structured events

```python
from ollamatoolkit.client import OllamaClient

with OllamaClient() as client:
    for event in client.stream_chat_events(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Explain caching in 2 bullets."}],
    ):
        if event.event == "token":
            print(event.text, end="", flush=True)
```

### OpenAI-compatible `/v1/*` helpers

```python
from ollamatoolkit.client import OllamaClient

with OllamaClient() as client:
    models = client.openai_list_models()
    chat = client.openai_chat_completions(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "One sentence about B2B lead qualification."}],
    )
    print(chat.choices[0].message.content)
```

### Structured extraction

```python
from pydantic import BaseModel
from ollamatoolkit.agents.simple import SimpleAgent

class CompanyFacts(BaseModel):
    website: str
    ceo_name: str

agent = SimpleAgent(
    name="extractor",
    system_message="Return valid JSON only.",
    model_config={"model": "ollama/qwen3:8b", "base_url": "http://localhost:11434"},
)

facts = agent.run_structured("Extract website and CEO from: ...", CompanyFacts)
print(facts.model_dump())
```

---

## CLI Usage

Entry point:

```bash
ollamatoolkit --help
# or
python -m ollamatoolkit.cli --help
```

### Commands

#### `config`
- `--init` create sample config JSON.

#### `run`
- positional: `role_file`
- `--task/-t`
- `--interactive/-i`
- `--dashboard/-d`
- `--model/-m`
- `--tools` (`all files math db server system web vision`)
- `--db-path`
- `--config/-c`
- `--batch-file/-b`
- `--output-dir/-o`
- `--streaming/-s`

#### `models`
- `--capability/-c` (`vision embedding tools completion reasoning`)
- `--json`
- `--base-url`

#### `chat`
- `--model/-m`
- `--base-url`
- `--system/-s`

### Example commands

```bash
# Generate sample config
python -m ollamatoolkit.cli config --init

# Quick chat
python -m ollamatoolkit.cli chat --model qwen3:8b

# Run role-based agent with streaming
python -m ollamatoolkit.cli run .\roles\researcher.json -t "Research Acme Corp" --streaming

# List only embedding-capable models as JSON
python -m ollamatoolkit.cli models --capability embedding --json
```

---

## Configuration Hierarchy

When running through CLI/runtime patterns, prefer this precedence:
1. **Explicit runtime arguments** (CLI flags / constructor args)
2. **Config file values** (`config.json` or generated sample config)
3. **Dataclass defaults** in `ToolkitConfig`

Primary config sections:
- `agent`
- `tools` (including `mcp_servers`)
- `web`
- `vision`
- `vector`
- `models` (per-capability slot config)
- `memory`
- `document`
- `benchmark`
- `telemetry`
- `logging`
- `dashboard`

Load and save:

```python
from ollamatoolkit.config import ToolkitConfig

cfg = ToolkitConfig.load("config.json")
cfg.save("config.generated.json")
```

---

## Architecture (high level)

- **Client layer**: `ollamatoolkit.client` + `client_api/*`
  - sync + async clients
  - transport + retries/backoff/circuit breaker
  - chat/generate/embed/model-management/web
  - OpenAI-compat wrappers
- **Agent layer**: `agents/simple.py`, `agents/team.py`, `agents/role.py`, `agents/memory.py`
  - tool calling
  - streaming token loops
  - structured Pydantic responses
  - multi-agent orchestration
- **Tool layer**: `tools/*`
  - vector, document, web, files, DB, math, email, schema, system, server
  - vision subpackage for OCR/analysis/video/spatial tasks
  - MCP client for external tool servers
- **Model intelligence**: `models/selector.py`
  - choose best model by capability

---

## Source File Map (what each file does)

| File | Responsibility |
|---|---|
| `ollamatoolkit/__init__.py` | Ollama Toolkit - A Professional-Grade Agentic Framework |
| `ollamatoolkit/agent.py` | Ollama Toolkit - Agent Shim |
| `ollamatoolkit/agents/__init__.py` | Ollama Toolkit - Agent Components |
| `ollamatoolkit/agents/memory.py` | Ollama Toolkit - Agent Memory Management |
| `ollamatoolkit/agents/role.py` | Ollama Toolkit - Role Agent |
| `ollamatoolkit/agents/simple.py` | Simple agent runtime for synchronous/async LiteLLM conversations with tool execution. |
| `ollamatoolkit/agents/team.py` | Ollama Toolkit - Multi-Agent Team Orchestration |
| `ollamatoolkit/callbacks.py` | OllamaToolkit Callbacks |
| `ollamatoolkit/cli.py` | Ollama Toolkit - CLI Runner |
| `ollamatoolkit/client.py` | Public client module for OllamaToolkit. |
| `ollamatoolkit/client_api/__init__.py` | Composable client-domain adapters used by OllamaToolkit public client façades. |
| `ollamatoolkit/client_api/async_client.py` | Public asynchronous Ollama client composed from endpoint-domain adapters. |
| `ollamatoolkit/client_api/common.py` | Shared helpers for OllamaToolkit API clients. |
| `ollamatoolkit/client_api/inference.py` | Inference endpoint adapters for OllamaToolkit clients. |
| `ollamatoolkit/client_api/models.py` | Model-management endpoint adapters for OllamaToolkit clients. |
| `ollamatoolkit/client_api/openai_compat.py` | OpenAI-compatible (`/v1/*`) endpoint adapters for OllamaToolkit clients. |
| `ollamatoolkit/client_api/sync_client.py` | Public synchronous Ollama client composed from endpoint-domain adapters. |
| `ollamatoolkit/client_api/transport.py` | HTTP transport layer used by OllamaToolkit client domain modules. |
| `ollamatoolkit/client_api/web.py` | Web endpoint adapters for OllamaToolkit clients. |
| `ollamatoolkit/common/utils.py` | Ollama Toolkit - Utilities |
| `ollamatoolkit/config/__init__.py` | Ollama Toolkit - Configuration Package |
| `ollamatoolkit/config/core.py` | Ollama Toolkit - Core Configuration |
| `ollamatoolkit/config/presets.py` | Ollama Toolkit - Model Presets |
| `ollamatoolkit/config.py` | Ollama Toolkit - Configuration |
| `ollamatoolkit/connector.py` | High-level Ollama connector façade used by toolkit consumers and dependent apps. |
| `ollamatoolkit/dashboard.py` | Ollama Toolkit - Mission Control Dashboard |
| `ollamatoolkit/exceptions.py` | OllamaToolkit Exception Hierarchy |
| `ollamatoolkit/extractor.py` | Schema-first field extraction helper built on top of ``SimpleAgent``. |
| `ollamatoolkit/models/__init__.py` | Ollama Toolkit - Model Utilities |
| `ollamatoolkit/models/selector.py` | Ollama Toolkit - Smart Model Selector |
| `ollamatoolkit/openai_types.py` | Typed models for Ollama OpenAI-compatible (`/v1/*`) endpoints. |
| `ollamatoolkit/telemetry.py` | Ollama Toolkit - Telemetry Integration |
| `ollamatoolkit/tool_registry.py` | OllamaToolkit - LLM Tool Registry |
| `ollamatoolkit/tools/__init__.py` | Ollama Toolkit - Tools Package |
| `ollamatoolkit/tools/benchmark.py` | Ollama Toolkit - Model Benchmarking (GPU-Aware) |
| `ollamatoolkit/tools/cache.py` | OllamaToolkit - Cache Tools |
| `ollamatoolkit/tools/db.py` | Ollama Toolkit - Database Tools |
| `ollamatoolkit/tools/document.py` | Ollama Toolkit - Document Processor |
| `ollamatoolkit/tools/email.py` | OllamaToolkit - Email Tools |
| `ollamatoolkit/tools/files.py` | Ollama Toolkit - File Tools |
| `ollamatoolkit/tools/math.py` | Ollama Toolkit - Math Tools |
| `ollamatoolkit/tools/mcp.py` | Ollama Toolkit - MCP Client |
| `ollamatoolkit/tools/models.py` | Ollama Toolkit - Model Inspector |
| `ollamatoolkit/tools/pdf.py` | PDF helper utilities shared by vision/document workflows. |
| `ollamatoolkit/tools/schema.py` | OllamaToolkit - JSON Schema Tools |
| `ollamatoolkit/tools/server.py` | Ollama Toolkit - Server Tools |
| `ollamatoolkit/tools/system.py` | Ollama Toolkit - System Tools |
| `ollamatoolkit/tools/system_health.py` | Ollama Toolkit - System Health Tool |
| `ollamatoolkit/tools/vector.py` | Ollama Toolkit - Vector Intelligence (RAG) |
| `ollamatoolkit/tools/vision/__init__.py` | Ollama Toolkit - Vision Package |
| `ollamatoolkit/tools/vision/analysis.py` | Ollama Toolkit - Vision Analysis |
| `ollamatoolkit/tools/vision/metadata.py` | Ollama Toolkit - Vision Metadata |
| `ollamatoolkit/tools/vision/ocr.py` | OCR helper that routes image/PDF inputs through a vision-capable model. |
| `ollamatoolkit/tools/vision/spatial.py` | Ollama Toolkit - Vision Spatial |
| `ollamatoolkit/tools/vision/tiling.py` | Ollama Toolkit - Vision Tiling |
| `ollamatoolkit/tools/vision/video.py` | Ollama Toolkit - Smart Video Processor |
| `ollamatoolkit/tools/web.py` | Ollama Toolkit - Web Tools |
| `ollamatoolkit/types.py` | Ollama Toolkit - Type Definitions |
| `ollamatoolkit/utils.py` | Ollama Toolkit - Utilities |

---

## Examples Directory

The `examples/` folder includes runnable patterns:
- `01_basic_chat.py`
- `02_with_tools.py`
- `03_vision_ocr.py`
- `04_rag_pipeline.py`
- `05_multi_agent_team.py`
- `06_streaming_ui.py`
- `07_schema_cache.py`
- `08_email_tools.py`
- `09_tool_registry.py`

---

## Testing and Quality

```bash
# Lint
python -m ruff check .

# Format
python -m ruff format .

# Type-check
python -m mypy src/ollamatoolkit

# Tests
python -B -m pytest -q
```

Optional live Ollama integration tests (env-gated):

```bash
# PowerShell example (do not hardcode private hosts in tracked files)
$env:OLLAMA_TEST_BASE_URL="http://YOUR_HOST:11434"
python -m pytest tests/test_live_ollama_integration.py -m integration -v
```

Workspace gateway (run from workspace root):

```bash
python .\tools\run_quality_gate.py --project .\OllamaToolkit
```

---

## Publishing Checklist (GitHub + PyPI)

From workspace root:

```bash
python .\tools\run_quality_gate.py --project .\OllamaToolkit
python .\tools\publish_release.py --project .\OllamaToolkit
```

---

## Security & Privacy Notes

- Do not commit private server IPs, API keys, or machine-specific paths.
- Use environment variables for private runtime endpoints.
- Keep local-only testing settings in ignored files (e.g., local config files excluded by `.gitignore`).

---

## Author

**Created by**: Roy Dawson IV  
**GitHub**: [https://github.com/imyourboyroy](https://github.com/imyourboyroy)  
**PyPI**: [https://pypi.org/user/ImYourBoyRoy/](https://pypi.org/user/ImYourBoyRoy/)

---

## Lightweight Model Inventory CLI

`ollamatoolkit models` now stays lightweight by lazily importing optional tool families. That means model inspection no longer requires vision extras just to list inventory.

### Inspect installed + running models

```bash
ollamatoolkit models --base-url http://192.168.1.21:11434 --json
```

### Merge benchmark context into inventory output

```bash
ollamatoolkit models --base-url http://192.168.1.21:11434 --benchmarks-file output/_benchmarks/benchmark_report.json --json
```

The JSON payload now includes:
- installed inventory
- running models
- capability/family summaries
- recommended models by use-case
- optional benchmark-role summaries when a benchmark report is provided
