---
name: no-explanatory-comments
description: Connor deletes explanatory comments outright — code should be obviously correct rather than explain the history that produced it.
metadata:
  type: feedback
---

Comments and docstrings were trimmed twice in one session, then deleted outright in the fork
(`2690d9a`) — including the one-line invariant they had already been trimmed to, which annotated a
change from `np.where(...)` to `np.where(...)[0][0]`.

His framing: *"make sure the code is obviously correct and not bother explaining the history that
got us here."*

**Why:** a comment explaining why the code is the way it is records a past state, which is exactly
what the code should no longer need.
**How to apply:** if a line needs explaining, try making it obvious first. Rationale goes in memory,
not in the source. See [[over-producing-the-artifact]] and rule 6 in `CODING_STANDARDS.md`.
