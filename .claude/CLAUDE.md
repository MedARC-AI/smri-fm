# General context

## Where things live

| Path | Holds |
|---|---|
| `CODING_STANDARDS.md` | Coding standards for the project: general architecture, specific rules to enforce. **Read before writing any code.** |
| `.claude/NOTES.md` | Current project state: open threads, parked decisions, caveats on results. Dated, pruned when resolved. |
| `.claude/memory/` | Persistent project memory. Any context you want to preserve - rationale, measurements, rejected alternatives, cut snippets, failed starts, subtle bugs - can go in here. One fact per file, indexed by `MEMORY.md`; read what the hook matches, don't bulk read. Record observations and arguments, not a description of what another file currently contains. |
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
| `review` | Review for code quality. See `CODING_STANDARDS.md`. Run in a sub-agent to avoid bias to like your own code. |
| `polish` | Fix any significant reported code quality issues. |
| `commit` | Commit it. |

`complete` = draft + verify + polish. `ship` = complete + commit.

## Working style

Humans read slowly and have limited mental capacity. But they can still have good insight and vision sometimes. We can work together more effectively by sticking to the following preferences:

- For a given task, produce the minimum artifact to achieve the current goal. We can add complexity gradually as needed.
- Keep responses in the chat *concise*. Feel free to think through what you want to say in the background. You can also write reports in `.claude/scratch/` with full details. But don't overwhelm the human with walls of text that they will be unable to read and respond to.
