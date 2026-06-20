# Cursor

## Install

```powershell
# Windows
./scripts/install-agent-skills.ps1 -Agent cursor
```

```bash
# macOS / Linux
./scripts/install-agent-skills.sh --agent cursor
```

Skills copy to:

- **User (default):** `~/.cursor/skills/<skill-name>/SKILL.md`
- **Project:** `.cursor/skills/` in the current directory (`-Scope project`)

## Agent prompt

```text
Install the agent skills from https://github.com/imyourboyroy/ollamatoolkit
```

## MCP (recommended)

Configure Ollama in Cursor MCP settings:

```powershell
ollamatoolkit --help
```

Then say: **Follow the ollamatoolkit skill** when building or debugging Ollama agent workflows.

## Update

Re-run the install script after `git pull`.

