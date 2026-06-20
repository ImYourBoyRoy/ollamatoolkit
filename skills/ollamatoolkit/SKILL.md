---
name: ollamatoolkit
description: Builds Python agent workflows on Ollama via roy-ollama-toolkit — OllamaClient, SimpleAgent, MCPToolManager, streaming, structured extraction, and CLI. Use when creating local LLM agents, tool loops, RAG, or MCP bridges in Python. Do not use for Tauri/Rust apps that already have native Ollama integration unless building a Python sidecar.
---

# ollamatoolkit

Modular pip package (`roy-ollama-toolkit`) for reliable local/hybrid AI on Ollama + LiteLLM.

## When to use

- Python script or service needs chat, tools, streaming, or structured JSON from Ollama
- Wiring MCP stdio servers into an agent (`MCPToolManager`)
- Multi-agent teams, RAG (`tools/vector`), vision/OCR pipelines
- CLI batch: `ollamatoolkit run`, `models`, `chat`

**When NOT to use:** DNA_Tools / Genomics Caddy in-app chat (use Tauri `stream_ollama_chat`); non-Python stacks.

## Process

### 1. Install and verify server

```bash
pip install roy-ollama-toolkit
# or: pip install "roy-ollama-toolkit[full]"
curl http://localhost:11434/api/version   # or user's OLLAMA_HOST
```

### 2. Pick layer

```
Simple HTTP        → OllamaClient(base_url=...).chat(...)
Tool agent         → SimpleAgent(..., tools=[...]) + @agent.tool()
MCP tools          → MCPToolManager({...}).start_all() → pass schemas to SimpleAgent
Structured output  → agent.run_structured(prompt, MyPydanticModel)
Config-driven CLI  → ollamatoolkit run roles/foo.json -t "task" --streaming
```

### 3. Configuration

Precedence: **constructor/CLI args → config.json → ToolkitConfig defaults**

```python
from ollamatoolkit.config import ToolkitConfig
cfg = ToolkitConfig.load("config.json")
```

Sample: `sample_config.json` in repo root.

### 4. MCP bridge pattern

```python
from ollamatoolkit.tools.mcp import MCPToolManager
from ollamatoolkit.agents.simple import SimpleAgent

mcp = MCPToolManager({"web": {"command": "python", "args": ["-m", "some_mcp_server"]}})
mcp.start_all()
agent = SimpleAgent(..., tools=mcp.get_tool_schemas(), function_map=mcp.get_proxy_functions())
```

Shutdown: `mcp.shutdown()`.

### 5. Model selection

```bash
ollamatoolkit models --capability tools --json
ollamatoolkit models --base-url http://HOST:11434
```

Or `ollamatoolkit.models.selector` in code.

## Key docs (repo cache after install)

| Topic | Path |
|-------|------|
| Overview | `README.md` |
| Examples | `examples/01_basic_chat.py` … `09_tool_registry.py` |
| Agent rules | `AGENTS.md` |
| Tests | `pytest -q` |

Repo cache: `%USERPROFILE%\.cursor\ollamatoolkit\`

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "Raw requests are simpler" | Client handles retries, streaming events, OpenAI-compat |
| "Skip pytest" | Library change needs unit tests |
| "Use ollamatoolkit inside Tauri UI path" | Use app-native Ollama unless Python sidecar is explicit |
| "Hardcode model name" | Use selector or `models` CLI for capability fit |

## Verification

- [ ] Ollama reachable at configured `base_url`
- [ ] Chosen layer matches task (client vs agent vs MCP)
- [ ] `pytest -q` passes
- [ ] No secrets in code or committed config

## Install this skill

```powershell
pwsh -File "$env:USERPROFILE\.cursor\scripts\install-cursor-skills.ps1" `
  -RepoUrl "https://github.com/imyourboyroy/ollamatoolkit"
```

See `docs/cursor-setup.md`.
