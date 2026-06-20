# Windsurf

Windsurf uses `.windsurfrules` or global rules — not a native skills folder.

## Project rules

```bash
cat skills/ollamatoolkit/SKILL.md > .windsurfrules
```

Add `AGENTS.md` summaries if you need Rust development rules in the same project.

## Global rules

Windsurf → Settings → AI → Global Rules → paste `skills/ollamatoolkit/SKILL.md` (keep concise).

## Agent prompt

```text
Add the ollamatoolkit skill from https://github.com/imyourboyroy/ollamatoolkit to .windsurfrules for this project
```

## Tip

Keep 1–2 skills in `.windsurfrules`; paste README integration examples when debugging agents.

