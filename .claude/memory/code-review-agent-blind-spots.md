---
name: code-review-agent-blind-spots
description: Three blind spots found by dry-running a code-quality review agent — locality, re-litigating settled decisions, and disproportionate fixes. Do not re-derive.
metadata:
  type: feedback
---

Found by dry-running a review agent against commit `50cb2a8`:

1. **Locality.** It only found defects visible inside a single function. It missed a probe signature
   asymmetry visible only in `main.py`, where all three probes meet, and fixed one `try`/`except`
   test antipattern without grepping for the identical one in a sibling file. Needs a peer-symmetry
   check, a rule to read wiring/entry-point files in full regardless of diff size, and a
   sweep-for-siblings pass after each fix.
2. **Re-litigating settled decisions.** It reported deliberate deferrals as fresh findings. Needs a
   triage step that checks TODOs, commit bodies and notes before treating an observation as a
   finding.
3. **Proportionality.** A binary justification rule (reason / no reason) approved a 46-line
   pure-motion hunk to fix a stale section banner. Connor reverted it. Weigh fix size against
   problem size, and never mix pure motion with substantive edits.

Note the pattern: gap 1 was partly *introduced* by the fix for an earlier gap — a read-budget
heuristic that buried the small-but-load-bearing file. Tuning edits need the same scrutiny as code.

**Why:** these cost a full dry-run cycle each to find.
**How to apply:** the current `quality-review` skill contains none of the three — apply them when
running it, or fold them in next time it is edited.
