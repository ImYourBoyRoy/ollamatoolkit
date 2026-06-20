# ollamatoolkit — Agent Instructions

Repo-local rules for AI agents working on **ollamatoolkit** or building Python agent systems on Ollama/LiteLLM.

## Read first (in order)

1. `README.md` — install, API layers, CLI, architecture map
2. `examples/README.md` — runnable patterns (chat, tools, RAG, MCP, teams)
3. `sample_config.json` — config shape
4. `docs/cursor-setup.md` — install the Cursor skill
5. `pyproject.toml` — package name vs import path

## Package identity

| Install (PyPI) | Import |
|----------------|--------|
| `pip install roy-ollama-toolkit` | `import ollamatoolkit` |
| `pip install "roy-ollama-toolkit[full]"` | optional tools (system, email, vision, etc.) |

Python **3.10–3.13**. Supported local server: Ollama (`OLLAMA_HOST` / `base_url`).

## Instruction precedence

1. Explicit user request
2. This `AGENTS.md`
3. `README.md` + `examples/`
4. Existing project integration (e.g. Tauri/Rust apps may use native Ollama — do not force this library)

## Choose the right layer

| Need | Use |
|------|-----|
| Raw HTTP to Ollama | `OllamaClient` / `AsyncOllamaClient` (`ollamatoolkit.client`) |
| Tool-calling agent loop | `SimpleAgent` (`ollamatoolkit.agents.simple`) |
| Multi-agent | `Team` (`ollamatoolkit.agents.team`) |
| Role JSON + CLI batch | `RoleAgent`, `ollamatoolkit run` |
| External MCP tool servers | `MCPToolManager` (`ollamatoolkit.tools.mcp`) |
| Structured Pydantic output | `agent.run_structured(...)` |
| Model pick by capability | `models/selector.py`, CLI `ollamatoolkit models` |

## When NOT to use ollamatoolkit

- **DNA_Tools / Genomics Caddy** and other Tauri apps that already stream via Rust `stream_ollama_chat` — use the app's native integration unless building a standalone Python sidecar.
- Production cloud inference without a local/remote Ollama endpoint configured.

## Development workflow

```powershell
# From repo root
python -m ruff check .
python -m ruff format .
python -m mypy src/ollamatoolkit
python -B -m pytest -q
```

Live integration (env-gated): set `OLLAMA_TEST_BASE_URL`, run `tests/test_live_ollama_integration.py`.

Config hierarchy: **CLI args → config.json → ToolkitConfig defaults**.

## Verification before completion

- [ ] Correct layer chosen (client vs agent vs MCP)
- [ ] `pytest -q` passes for touched modules
- [ ] No private IPs, tokens, or hostnames committed
- [ ] Example or CLI path documented if behavior changed

## Cursor skill

See [docs/agent-skills/README.md](docs/agent-skills/README.md)

```text
Install the agent skills from https://github.com/imyourboyroy/ollamatoolkit
```

Skill name: **ollamatoolkit**
