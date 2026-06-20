# GitHub Copilot

## Project install

From your **project root** (not the ollamatoolkit repo):

```bash
git clone --depth 1 https://github.com/imyourboyroy/ollamatoolkit.git /tmp/ollamatoolkit
mkdir -p .github/skills
cp -R /tmp/ollamatoolkit/skills/* .github/skills/
```

Or from inside a ollamatoolkit clone:

```powershell
./scripts/install-agent-skills.ps1 -Agent copilot -Scope project
```

```bash
./scripts/install-agent-skills.sh --agent copilot --scope project
```

Copilot discovers skills under `.github/skills/`, `.claude/skills/`, or `.agents/skills/`.

## Agent prompt

```text
Install the ollamatoolkit agent skills into this project's .github/skills from https://github.com/imyourboyroy/ollamatoolkit
```

## Custom instructions

Summarize key rules in `.github/copilot-instructions.md`:

- Use OllamaClient or SimpleAgent before raw HTTP
- Run `pytest -q` when Ollama connection fails
- Windows: PowerShell 7+, shell init via `ollamatoolkit config --init`

Full workflow: `skills/ollamatoolkit/SKILL.md`

## References

[Creating agent skills for GitHub Copilot](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/create-skills)

