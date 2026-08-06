---
name: no-memory-references-in-code
description: Don't cite `.claude/memory/*.md` paths from source files — state the fact in the code and let memory hold the rationale.
metadata:
  type: feedback
---

Connor, 2026-08-06, on a `smri_mae.py` docstring that ended "-- see
`.claude/memory/smri-mae-preprocessing-gap.md`": **best not to refer to memories in the codebase
directly.** Replaced with the fact itself ("the brain mask is a mean threshold, not the SynthSeg
mask used in pretraining, so it keeps skull and neck").

**Why:** the two have different audiences and lifetimes. Source should state what is true of the
code; memory holds rationale, measurements and rejected alternatives for whoever picks the thread
up. A path reference couples them, goes stale silently when memory is pruned or renamed, and sends
a reader out of the file for something the sentence could just say.

**How to apply:** put the fact in the docstring or comment, in one line, and keep the argument and
evidence in `.claude/memory/`. Cross-links between memory files (`[[name]]`) are still fine — this
is only about source citing memory. Same spirit as CODING_STANDARDS rule 6.
