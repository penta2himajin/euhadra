# Vendored skills

Third-party Claude Code skills copied into this repository verbatim.

## `grill-me` / `grilling`

- Source: <https://github.com/mattpocock/skills> (`skills/productivity/grill-me`, `skills/productivity/grilling`)
- Author: Matt Pocock
- Licence: MIT — Copyright (c) 2026 Matt Pocock

Vendored rather than installed as a plugin because this repository is worked on
from ephemeral remote containers, where a user-level `~/.claude/skills`
install does not survive the session.

`grill-me` is a thin wrapper that invokes `grilling`; both files are needed.
Neither contains executable code — they are prompt instructions only.

Upstream install methods, if you would rather track the source:

```bash
/plugin marketplace add mattpocock/skills
/plugin install mattpocock-skills
# or
npx skills@latest add mattpocock/skills
```
