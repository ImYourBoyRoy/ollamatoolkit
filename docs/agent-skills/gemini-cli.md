# Gemini CLI

## Install (recommended)

```bash
gemini skills install https://github.com/imyourboyroy/ollamatoolkit.git --path skills
```

From a local clone:

```bash
git clone https://github.com/imyourboyroy/ollamatoolkit.git
gemini skills install /path/to/ollamatoolkit/skills/
```

Workspace-only (project `.gemini/skills/`):

```bash
gemini skills install /path/to/ollamatoolkit/skills/ --scope workspace
```

Or use the installer:

```bash
./scripts/install-agent-skills.sh --agent gemini
```

## Verify

```
/skills list
```

## Agent prompt

```text
Install the agent skills from https://github.com/imyourboyroy/ollamatoolkit using gemini skills install
```

## Persistent context (optional)

For always-on rules, add `@skills/ollamatoolkit/SKILL.md` to project `GEMINI.md`. Prefer on-demand skills for most workflows.

## MCP

Configure `OllamaClient` in `~/.gemini/config.json` when building Ollama agent workflows from Gemini.

