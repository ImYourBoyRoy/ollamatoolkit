# OpenCode

OpenCode uses **agent-driven** skill execution via `AGENTS.md` and the built-in `skill` tool.

## Install

1. Clone or open the repo as your workspace:

```bash
git clone https://github.com/imyourboyroy/ollamatoolkit.git
```

2. Ensure present:

- `AGENTS.md` (root)
- `skills/ollamatoolkit/SKILL.md`

No separate install step — the agent discovers skills from the workspace.

## Agent prompt

```text
Use the ollamatoolkit workspace skills from https://github.com/imyourboyroy/ollamatoolkit — read AGENTS.md and invoke the ollamatoolkit skill when building Ollama agent workflows
```

## Expected behavior

- Local LLM agent, tool loop, or Ollama client tasks → load `ollamatoolkit` skill
- Pick client vs SimpleAgent vs MCPToolManager before raw requests

## MCP

Configure Ollama in OpenCode MCP config when available.

