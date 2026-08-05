# Coding Standards

This document contains general coding standards for ML research codebases.

## Architecture

Codebases should consist of a reusable inner "core" and hackable outer "shell".

- The core contains isolated components, while the shell puts them together. Typical core components include datasets, transforms, and standard model building blocks. Meanwhile, the shell might include novel model implementations, custom data pipelines, and training loops.
- The core is stable and is typically reused from one project/branch to the next. It might expand over time, but once a component is "in", it should need few if any changes. Meanwhile, the shell is intended to evolve dynamically as we try out different ideas.

Core and shell modules are typically located together in a flat source tree.  Core and shell are often separated into different files, with shell importing from core. A minimal example codebase might look like:

```
package/
├── datasets.py -> core
├── transforms.py -> core
├── modules.py -> core
├── utils.py -> core
├── model.py -> shell
└── train.py -> shell
```

Fully developed codebases that follow this pattern reasonably well include:
- [capi](https://github.com/facebookresearch/capi)
- [mae](https://github.com/facebookresearch/mae)
- [nanochat](https://github.com/karpathy/nanochat)

> Nb, the core/shell distinction is inspired the well known functional core/imperative shell architecture. But we're not strict dogmatic software engineers and don't expect the core to be purely functional. E.g., it's fine for dataset implementations in the core to touch the outside world.

## Code Quality

Our goal is to write code that is easy to understand, easy to extend, and not bloated.

> we have to keep it crisp, disentangled, and simple if we refuse
to be crushed by the complexities of our own making - Dijkstra

Below are a set of rules (code smells and fix strategies) that we use to diagnose and treat less-than-ideal code.

### Rules

1. **Clever/dense code** -> Rewrite as plain, boring code with intermediate variables whose names document the steps. If the code uses some sophisticated approach, consider whether a simpler approach would be good enough. Optimize for the reader, not the writer. If a reader needs to be as smart as the author, it's a liability.

2. **Speculative generality** (unused abstraction layers, "just in case" config, premature interfaces) -> Delete it. Build for today's requirements (YAGNI) and refactor when a second use case actually arrives.

3. **The wrong abstraction** (a shared module accumulating flags, parameters, and conditionals to serve divergent callers) -> Fork it, delete the parts each caller doesn't use, and let the callers evolve independently. Duplication is cheaper than the wrong abstraction.

4. **Lots of tiny functions** (e.g. < 5 lines) -> Consolidate into meaningful logical units and/or inline. There is a mental overhead to each function call. Logic should be extracted to its own function only if it encapsulates a coherent concept that benefits from its own scope.

5. **Non-obvious interface** -> Replace clever bespoke data structures with standard data representations. Use type annotations and self-explanatory names. If an interface is still confusing, then perhaps it is not a coherent concept and should be refactored in some way.

6. **Long and/or lots of comments** -> Delete them and make the code self-explanatory (better names, type annotations, simpler patterns). Comments should be reserved for rare cases when some subtlety could not be avoided. And in these cases they should be ~1 line. Rationale and history go in `HISTORY.md`, not in comments.

7. **Excessive defensiveness** -> Remove defensive error handling and silent fallbacks. Assume the user is us and we know how to use the code. Fail fast and loudly when something unexpected happens.

8. **Dead code** -> Delete anything we don't need. Version control remembers.

9. **Inconsistency within a codebase** -> Match the existing conventions of the surrounding code. Consistency beats personal preference. This applies as well to matching structural patterns, not just style. When implementing a new variant of a concept (e.g. a new model), fork and edit an existing instance. But note that this rule is waived during explicit refactoring to improve adherence to the overall standards.

### Non-rules

These are some commonly suggested rules that we explicitly ignore/soften. Primarily because this is a research codebase.

1. **Duplicated code** -/-> Insisting on DRY in a dynamic research codebase leads to tight coupling and high friction. The stable core may contain canonical, verified implementations of standard components (e.g. Attention). But we should be free to fork and edit / "clone and own". If we end up needing to do a lot of shotgun surgeries, we can revisit this.

2. **Broad public surface area** -/-> There are no external API users so there is no need to worry about public/private. All functions should be public (no underscore prefix) to reduce mental overhead.

3. **Long functions doing many things** -/-> Functions in the stable core should be coherent blocks of work, not too short (< 5 lines) and not too long (> 1 screen). But for shell code (e.g. training main entrypoint), straight-line implementations are preferred over lots of indirection hopping.

## Correctness

ML code fails silently. So we need to be very careful, to the point of paranoia, to reduce the chances of silent issues.

1. **Unverified math** -> Numerical computation that runs but is wrong is the ultimate source of silent failures. Use standard libraries for standard things, don't unnecessarily roll your own. Prefer forking a known good implementation of a standard component rather than reimplementing from scratch. Use tests to verify implementations against a known good reference when possible. Use tests and assertions to check for the expected invariants.

2. **Silent coercion** (broadcasting that runs but computes the wrong thing) -> Assert shapes at tops of functions (or unpack shape to variables which functions as an assertion). Watch out for reshapes that can hide a mismatch. Prefer operations that crash on mismatch (e.g. `einops`).

3. **Unreproducibility** -> We need to systematically explore the space, while keeping track of our steps. New experiment code should be committed before running. Log git sha (and dirty status). Save the full config. Seed everything in one place. Custom prepared datasets must be reproducible from original sources.

4. **Too many bells and whistles** -> Prefer standard, battle-tested, easy to understand components. The more bells and whistles, the more we have to understand, the more surface area for problems.

## Git

Use concise commit messages. Detailed rationale and history notes go in `HISTORY.md`. For initial development, commit directly to main. Once the codebase is built out, use trunk-based branching off of main. Experimental branches should make minimal additions and *zero* deletions on the stable core. Fixes to the stable core should be rare, and should be made in short-lived branches that are merged quickly to main. The main branch should always run, and should contain the minimal baseline starting point for new experimental branches. Experimental branches are not expected to be merged to main. If an experimental branch is promising, it can be used as a starting point for further experiments.

## Tools

- `uv`
- `ruff` via pre-commit.
- `pytest` with tests in `test_*.py` next to the code it tests.
- Omegaconf for config with one flat yaml and struct mode on.
