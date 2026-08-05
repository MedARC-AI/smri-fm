---
name: gpu-session
description: Get an interactive GPU allocation on the Slurm cluster, run commands in it, and release it. Use when something genuinely needs a device — a forward pass, a CUDA-only code path, a timing measurement — and CPU verification is not enough.
scope: general
---

# GPU session

The login node has **no GPU driver at all**. Anything touching a device needs an allocation.

## Protocol

Verify that everything runs locally on CPU first. Save scripts in `.claude/scratch` so that they are accessible on the compute node. Note that `/tmp` is not shared.

Allocate without spawning a shell, run commands into the allocation, release it explicitly.

```bash
# 1. Allocate. Bounded time, one GPU, in the background.
salloc --partition=n --gpus=1 --time=1:00:00 --account=sophont --no-shell
#   -> "salloc: Granted job allocation <JOBID>"

# 2. Run commands in it, as many as you need.
srun --jobid=<JOBID> --overlap nvidia-smi
srun --jobid=<JOBID> --overlap uv run --no-sync python .claude/scratch/check_thing.py

# 3. Release as soon as you're done. Do not leave it idling.
scancel <JOBID>
```

Check for an allocation you already hold before making another: `squeue -u $USER`.

## Rules

- **One allocation per work chunk**, not per check. Allocating is the expensive part.
- **Always time-bounded.** Default `--time=1:00:00`; raise it deliberately, never to `infinite`.
- **One GPU** unless the work genuinely needs more. Nodes have 8x H100 80GB and are shared.
- **Always release.** `scancel` when the work is done, and check `squeue -u $USER` before ending
  the session. A forgotten allocation blocks someone else's sweep.
- **Never `sbatch` a long job** as part of a verification pass. Batch jobs are for real sweeps, and
  those are the user's to launch — write the script and hand back the command.

## Cluster facts

| | |
|---|---|
| Partitions | `n` (8 nodes, 8x H100 80GB each), `c` (6 nodes, CPU only), `main` (both) |
| Account | `sophont` |
| GPU partition | `n` — `main` will also give you a CPU-only node, so ask for `n` explicitly |
