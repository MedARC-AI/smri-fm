# ML Research Code Principles

Reference for how to write code in this repo. These principles apply generally to all of our ML research projects.

## 1. Simplicity

- A simple approach that performs as well, or nearly as well, as a more complex one is strongly preferred.
- Prefer standard, battle-tested components over the latest and fanciest method (e.g. default to torch SDPA over the latest flash variant, standard position encodings over the latest RoPE variant).
- Scrutinize any cleverness in the code. Is it actually needed?
- Enforcing simplicity helps us stay sane as we try to systematically explore the research landscape. Every piece of novelty is something we have to validate.

## 2. Data

- Look at your data before writing model code, and at every stage of preprocessing: raw samples, augmented samples, batches right before they hit the model.
- Most bugs live in the pipeline, not the model. And pipeline bugs don't crash, they quietly train something worse than it should be.
- Throwaway visualization tooling (a one-off Flask/Streamlit app to browse a stream of samples) is worth the time it takes to build.

## 3. Hackability

- The codebase's job is to make the next experiment cheap to run, not to be a production-grade framework.
- Prefer a flat, hackable codebase, e.g. nanochat.
- No abstraction/indirection unless a repeated pattern demands it. And even then, prefer a lightweight interface.
- Duplication to avoid coupling is good. Default to one shared implementation of common building blocks (e.g. an Attention module, preprocessing utils), but let a model fork its own version, rather than add flags/branches to the shared implementation for one model's special case.
- Design for hacking, but don't add speculative configurability: build the seams you'll actually pull on, not flags for variations you haven't tried yet.
- Avoid defensiveness. Minimize error handling, try/except, silent fallbacks. Let the code crash loudly on anything unexpected. This is narrow-use research code, not a production service. Cheap, well-understood failures (e.g. a corrupted file) can be handled, but never silently: log what got skipped.

## 4. Correctness

- Cover the main code paths with tests; don't chase exhaustive edge-case coverage. Tests should be fast enough to run before every commit.
- Keep a debug/smoke config that runs the full training loop end-to-end (tiny model, tiny data, few steps) to catch integration bugs before a real run.
- When reimplementing something with a known-good reference available, check the outputs match numerically (e.g. a from-scratch ViT against timm's). Assert the invariants a bug would otherwise violate silently (finite loss/grad-norm, a causal mask that actually masks). Numerical bugs don't crash, they quietly train a worse model or produce a wrong conclusion, which makes them more dangerous.
- When testing a new training implementation, it's encouraged to first check that you can overfit a tiny dataset before any real run.
- Use pytest for testing, and place tests in `test_*.py` files next to the code they are testing.

## 5. Scaling

- Start with the smallest model/data/step-count that can prove the pipeline works, and get it fully debugged before scaling up.
- When scaling, prefer changing one axis at a time (data size, model size, steps) so a regression is attributable.
- Bigger costs more compute. Scale because you have evidence it will help, not by default.

## 6. Code style

- Format and lint with ruff via pre-commit.
- Write code for a slow human reader (or even better, a golden retriever). Split complex expressions into multiple lines. Use functions to encapsulate pieces of logic as needed.
- Keep comments minimal. Code should ideally be obvious and not require comments as much as possible. Where needed, comments should not be longer than needed. Avoid block comments that explain the detailed rationale/history.
- Full docstrings are not required. Function name, signature, type hints should be enough documentation in most cases. Short docstrings can be used when there are non-obvious things to explain.
- Markdown documentation should also be concise. Again, think of the slow human reader.

## 7. Reproducibility

- Every run writes fully resolved config + git SHA to its run dir. One command + SHA reproduces it.
- Use `uv` + tracked lockfile for reproducible environments.
- Configuration should be lightweight: omegaconf + one flat yaml default config, no hydra nesting. Struct mode on: code reads keys directly and crashes on missing/typo'd keys; avoid `.get(k, default)` scattered around.
- Log loss / lr / grad-norm etc. to wandb and a local jsonl log.
- One seed flag, set everywhere. Aim for *reproducible enough to trust a result*, not exact numerical reproducibility. Prefer multiple seeds per experiment when feasible, to judge robustness.
- Have some system for keeping an organized history of experiments. E.g. a directory of self-contained experiment folders, each with configs, run scripts, analysis scripts/notebooks, result figures.

## 8. Evaluation

- Establish a set of evals up front, with dumb baselines and prior-work baselines in place before running experiments. This avoids hill-climbing the wrong hill with no reference point. Extend the suite as you learn what the model needs to be good at; up front doesn't mean frozen.
- Evals should be diverse and reliable, yet fast enough for quick iteration, and runnable standalone from any checkpoint — not tightly coupled into training.
- Output eval metrics in a form that's easy to collate and compare across runs.
- Compare against a fixed anchor/baseline checkpoint, not just the previous run, to avoid comparison drift.

## 9. Compute

- Know your throughput (tokens/sec, MFU) as a first-class number next to loss. A correct but much slower run is also a bug.
- Don't over-index on squeezing out the last few percent of utilization — the standard components from §1 are good enough in most cases. Put the effort into not stalling on the data pipeline; that's usually where the real waste hides.

## 10. Git

- Code changes should be committed before running experiments.
- Code should pass formatting/lint and tests before committing. One-off smoke tests to check a fix are fine and don't need to be kept around.
- Commit messages should be concise.
- Feel free to commit to `main` directly. Prefer a linear history. Branch off of `main` as needed to try out parallel experiment directions.
