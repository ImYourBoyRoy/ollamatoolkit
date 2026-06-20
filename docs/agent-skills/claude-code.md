# Claude Code

## Marketplace install (recommended)

In Claude Code:

```text
/plugin marketplace add https://github.com/imyourboyroy/ollamatoolkit.git
/plugin install ollamatoolkit@ollamatoolkit
```

If SSH clone fails, use the HTTPS marketplace URL above.

## Local / development

```bash
git clone https://github.com/imyourboyroy/ollamatoolkit.git
claude --plugin-dir /path/to/ollamatoolkit
```

## Agent prompt

```text
Install the agent skills from https://github.com/imyourboyroy/ollamatoolkit using the Claude Code plugin marketplace or --plugin-dir
```

## Skills location

Plugin metadata: `.claude-plugin/plugin.json`  
Skills: `skills/ollamatoolkit/SKILL.md`

Also read repo `AGENTS.md` when editing the Rust codebase.

## MCP

Set OLLAMA_HOST from `ollamatoolkit --help` to Claude MCP config for structured Python environment tools.

