---
name: over-producing-the-artifact
description: Connor's most repeated correction — I build more artifact than the job needs, and cutting it in one place tends to relocate it to another.
metadata:
  type: feedback
---

Four corrections in one week (2026-07-29..30), all the same defect: over-producing the artifact.
Comments and docstrings, then commit bodies, then a plot built to a full design spec, then restyling
a draft he handed over.

The pattern worth remembering is the *relocation*: when comments were cut, the material reappeared
in five-paragraph commit bodies. Removing the output in one place did not remove the impulse.

**Why:** he wants the minimum artifact that does the job — extra explanation is not free, it is
something he has to read and maintain.
**How to apply:** default to the smaller version and let him ask for more. Specific forms:
[[no-explanatory-comments]], [[plain-matplotlib-for-internal-plots]], and minimum-diff on his drafts
(covered in the `add-eval-model` skill). Commit messages: concise, per `CODING_STANDARDS.md`.
