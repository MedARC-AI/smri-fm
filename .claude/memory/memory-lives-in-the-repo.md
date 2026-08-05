---
name: memory-lives-in-the-repo
description: Assistant memory lives at .claude/memory/ inside the repo, not in the harness's external directory — Connor must be able to read, review and delete it.
metadata:
  type: project
---

The harness's own memory directory (`~/.claude/projects/-data-connor-nanobrain-1/memory/`) was
deleted on 2026-08-01. It had grown to 18 files and 45KB in ten days with no pruning mechanism, and
none of it was state Connor could read, review or delete.

The replacement — a single append-only `.claude/HISTORY.md` — was itself dropped on 2026-08-05: too
monolithic, and its entries describing other files in `.claude/` went stale and contradicted them.
Memory is now one fact per file under `.claude/memory/`, indexed by `MEMORY.md`.

Also rejected during that design: a `.claude/rules/` directory — no content yet, and its boundary
against CLAUDE.md was undefined.

**Why:** memory outside the repo is unreviewable and unportable; memory in one big file is
unprunable.
**How to apply:** write new memories to `.claude/memory/`, one fact per file, and add the index
line. Record observations and arguments, not descriptions of what another file currently contains —
that is the specific thing that rotted. The trade-off: project-local memory is **not** auto-recalled
by the harness, so it only gets read when CLAUDE.md points at it.
