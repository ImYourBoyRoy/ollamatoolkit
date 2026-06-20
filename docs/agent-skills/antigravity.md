# Antigravity CLI (agy)

## Install

```bash
agy plugin install https://github.com/imyourboyroy/ollamatoolkit.git
```

Local clone:

```bash
git clone https://github.com/imyourboyroy/ollamatoolkit.git
agy plugin install /path/to/ollamatoolkit
```

Or:

```bash
./scripts/install-agent-skills.sh --agent antigravity
```

## Validate

```bash
agy plugin validate /path/to/ollamatoolkit
agy plugin list
```

## Agent prompt

```text
Install the agent skills from https://github.com/imyourboyroy/ollamatoolkit as an Antigravity plugin (agy plugin install)
```

## Workspace rules

Copy or symlink `AGENTS.md` into project roots where strict Ollama agent discipline is required.

## MCP

Use `OllamaClient` alongside the plugin for structured runtime management.

