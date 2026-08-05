# General context

## Where things live

| Path | Holds |
|---|---|
| `CODING_STANDARDS.md` | Coding standards for the project: general architecture, specific rules to enforce. |
| `.claude/NOTES.md` | Current project state: open threads, parked decisions, caveats on results. Dated, pruned when resolved. |
| `.claude/HISTORY.md` | Persistent memory dump. Any context you want to preserve - rationale, measurements, rejected alternatives, cut snippets, failed starts, subtle bugs - can go in here. Dated, append only, never pruned. Search, don't bulk read. |
| `.claude/scratch/` | Agent scratch space. Use for one off scripts, measurements, planning docs. Nb, the `.scratch/` directory in the project root is the user's scratch space and should not be used. |
| `.claude/skills/` | Reusable procedures. Frontmatter `scope: general` travels to a new project, `scope: project` stays. |
| `<area>/README.md` | How that area works, and the gotchas particular to it. Next to the code. Concise overview in main project `README.md`. |
| `experiments/<name>/` | Self-contained: config, launch script, analysis, figures. |

All persistent state should live inside the repo, where it can be read, edited, and deleted. Assume that memory outside the repo will not persist.

## Process

Work moves through stages. Each ends at a natural gate: stop, report, wait.
If the stage isn't explicitly given, make your best judgement and say what you picked.

| Stage | Means |
|---|---|
| `discuss` | We don't know what to do yet. No code. Options, trade-offs, a recommendation. |
| `sketch` | Show the idea concretely. Incomplete is fine — say what's missing. |
| `draft` | A full working version. It runs. Not verified, not polished. |
| `verify` | Correctness checks: tests, numerical comparison against a reference, invariants. |
| `polish` | Quality pass — simplify, delete, lint clean, tests green. See `CODING_STANDARDS.md`. |
| `commit` | Commit it. |

`complete` = draft + verify + polish. `ship` = complete + commit.
