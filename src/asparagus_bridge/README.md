# asparagus_bridge

Thin bridge between smri-fm pretraining and asparagus (official FOMO26
framework) finetuning + evaluation. We pretrain natively; asparagus handles
finetuning on downstream tasks and metric collection.

## Layout

| File | Purpose |
| --- | --- |
| `models_smri_mae.py` | `SmriMaeClsRegBackbone` (real), `SmriMaeSegBackbone` (placeholder) — asparagus-compatible wrappers around the MAE encoder |
| `checkpoint.py` | `convert_smri_mae_checkpoint(src, dst)` — rewrite a smri_mae `.pth` as `{"state_dict": ...}` with `model.encoder.*` keys |
| `configs/model/smri_mae.yaml` | Hydra overlay registering the wrappers under `+model=smri_mae` (picked up via `ASPARAGUS_*_CONFIGS` search paths) |

Per-implementation siblings (e.g. `models_smri_dino.py`,
`configs/model/smri_dino.yaml`) drop in alongside without cross-talk.

## End-to-end flow

### 1. Prereqs

- `uv sync` (installs asparagus from `third_party/asparagus` editable).
- `source scripts/setup_asparagus_env.sh` once per shell — exports
  `ASPARAGUS_*` env vars and registers our config overlay on asparagus'
  Hydra search path.
- Downstream task data laid out under `$ASPARAGUS_DATA/<TASK>/` per
  [docs/data-pipeline/data_structure.md](../../third_party/asparagus/docs/data-pipeline/data_structure.md).

### 2. Convert the pretrain checkpoint

```python
from asparagus_bridge.checkpoint import convert_smri_mae_checkpoint
convert_smri_mae_checkpoint("runs/mae/checkpoint-last.pth", "runs/mae/asparagus.ckpt")
```

Strips decoder/patchify/target_norm, keeps `encoder.*`, prefixes keys with
`model.` so asparagus' `BaseModule.load_state_dict(strict=False)` lands them
on the right submodule. Missing head weights are expected (head is randomly
initialised at finetune time).

### 3a. Per-task finetune (manual)

```sh
asp_finetune_cls task=<CLS_TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
asp_finetune_reg task=<REG_TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
# asp_finetune_seg blocked: SmriMaeSegBackbone is a fail-fast placeholder pending design.
```

### 3b. Linear probe

```sh
asp_linear_probe task=<TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
```

### 3c. Multi-task driver

```sh
FOMO_CLS_TASKS="<task1> <task2>" \
FOMO_REG_TASKS="<task1> <task2>" \
  scripts/eval_fomo26.sh runs/mae/checkpoint-last.pth
```

Converts the checkpoint and loops `asp_finetune_*` across the configured
tasks. Bypasses `asp_eval_box_run` because that command only forwards
`checkpoint_run_id` (no `checkpoint_path` passthrough); we may revisit if we
need its Slurm orchestration on HPC.

## Caveats

- **Architecture must match** the smri_mae model that produced the
  checkpoint: `img_size`, `patch_size`, `depth`, `embed_dim`, `num_heads`.
  Mismatches surface as shape errors during load (not silenced by
  `strict=False`). Override the relevant fields in
  `configs/model/smri_mae.yaml` or via Hydra CLI.
- **`SmriMaeSegBackbone` is a placeholder.** ViT→dense-prediction needs a
  design pass (UperNet vs simple upsampler vs transformer decoder, handling
  variable img_size, intermediate-block features).
- **Asparagus as a pretraining baseline** is reachable in principle —
  `asp_pretrain task=PT003_FOMO300K +model=unet_b` — but requires fomo300k
  materialized in asparagus' on-disk format. Not wired here.
