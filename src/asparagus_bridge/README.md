# asparagus_bridge

Bridge between smri-fm pretraining and asparagus (official FOMO26 framework) finetuning + evaluation.  finetuning on downstream tasks and metric collection.


### 0. Temp Asparagus fixes - 20260513
Asparagus repos have a couple of bugs that haven't been closed yet: [1](https://github.com/Sllambias/asparagus_preprocessing/pull/1), [2](https://github.com/Sllambias/asparagus_preprocessing/pull/2), [3](https://github.com/Sllambias/asparagus/pull/3). Until they are merged, please tell your favorite coding agent to `Patch the submodules with small changes from asparagus_quickfixes.md`.

### 1. Prereqs

- `uv sync`
- `source scripts/setup_asparagus_env.sh` exports
`ASPARAGUS_*` env vars and registers our configs on asparagus'
Hydra search path. Use `.env` if you want non-default environment variables.
- Download [raw FOMO26 finetuning data](https://sid.erda.dk/cgi-sid/ls.py?share_id=fmeuvo1EdF) to `$ASPARAGUS_SOURCE` folder.
- Convert the task data from raw to fine-tuning format, e.g.:

    ```
      cd "$ASPARAGUS_SOURCE"
      unzip -n Task_3.zip -d Task_3

      uv run asp_process --dataset REGR002 --save_as_tensor --num_workers 4
      uv run asp_split --dataset REGR002_FOMO26_BrainAge --vals 80 10 10
    ```

    That step is explained in the [official asparagus guide](https://sid.erda.dk/share_redirect/fmeuvo1EdF/FOMO26_Guide_v1.pdf).
- [TODO] To reduce the number of necessary steps, the processed data from the previous step will be moved to HF so no local script running is needed.

### 2. Convert the pretrain checkpoint
```python
from asparagus_bridge.checkpoint import convert_checkpoint
convert_checkpoint("smri_mae", "runs/mae/checkpoint-last.pth", "runs/mae/asparagus.ckpt")
```

Register additional model converters in `asparagus_bridge.checkpoint.CONVERTERS`.

### 3a. Per-task finetune
The preferred way of running finetuning runs is by using configs:

```sh
uv run asp_finetune_reg --config-name finetuning/smoke_test_bag.yaml
```
[smete_test_bag.yaml](./configs/finetuning/smoke_test_bag.yaml) can be used as a reference.

Specific fine-tuning tasks can also quickly be invoked by passing CLI arguments that override default configs, but that leads to worse reproducibility:

```sh
asp_finetune_cls task=<CLS_TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
```

### 3b. Linear probe - not tested yet

```sh
asp_linear_probe task=<TASK> +model=smri_mae checkpoint_path=runs/mae/asparagus.ckpt
```

### 3c. Multi-task finetune eval - not tested yet

```sh
FOMO_CLS_TASKS="<task1> <task2>" \
FOMO_REG_TASKS="<task1> <task2>" \
  scripts/eval_fomo26.sh smri_mae runs/mae/checkpoint-last.pth
```
