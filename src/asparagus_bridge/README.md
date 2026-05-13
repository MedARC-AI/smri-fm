# asparagus_bridge

Bridge between smri-fm pretraining and asparagus (official FOMO26 framework) finetuning + evaluation.  finetuning on downstream tasks and metric collection.

### 1. Prereqs

- `uv sync`
- `source scripts/setup_asparagus_env.sh` exports
  `ASPARAGUS_*` env vars and registers our configs on asparagus'
  Hydra search path.
- Download data to `$ASPARAGUS_DATA/<TASK>/` per
  [docs/data-pipeline/data_structure.md](../../third_party/asparagus/docs/data-pipeline/data_structure.md).

### 2. Convert the pretrain checkpoint

```python
from asparagus_bridge.checkpoint import convert_checkpoint
convert_checkpoint("smri_mae", "runs/mae/checkpoint-last.pth", "runs/mae/asparagus.ckpt")
```
Register additional model converters in `asparagus_bridge.checkpoint.CONVERTERS`.

### 3a. Per-task finetune

```sh
asp_finetune_cls task=<CLS_TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
```

### 3b. Linear probe

```sh
asp_linear_probe task=<TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
```

### 3c. Multi-task finetune eval

```sh
FOMO_CLS_TASKS="<task1> <task2>" \
FOMO_REG_TASKS="<task1> <task2>" \
  scripts/eval_fomo26.sh smri_mae runs/mae/checkpoint-last.pth
```