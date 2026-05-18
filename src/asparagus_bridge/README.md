# asparagus_bridge

Bridge between smri-fm pretraining and asparagus (official FOMO26 framework)
finetuning + evaluation.


### 0. Temp Asparagus fixes - 20260513
Asparagus repos have a couple of bugs that haven't been closed yet: [1](https://github.com/Sllambias/asparagus_preprocessing/pull/1), [2](https://github.com/Sllambias/asparagus_preprocessing/pull/2), [3](https://github.com/Sllambias/asparagus/pull/3). Until they are merged, please tell your favorite coding agent to `Patch the submodules with small changes from asparagus_quickfixes.md`.

### 1. Prereqs

- `uv sync`
- `source scripts/setup_asparagus_env.sh` exports
`ASPARAGUS_*` env vars and registers our configs on asparagus'
Hydra search path. Use `.env` if you want non-default environment variables.
- Download [raw FOMO26 finetuning data](https://sid.erda.dk/cgi-sid/ls.py?share_id=fmeuvo1EdF) to `$ASPARAGUS_SOURCE` folder.

Task data must be converted from raw FOMO26 format to asparagus' finetuning
format before training. The step is explained in the
[official asparagus guide](https://sid.erda.dk/share_redirect/fmeuvo1EdF/FOMO26_Guide_v1.pdf).

#### Task 1: infarct classification

Task 1 is `CLS002_FOMO26_Infarct`. For smri MAE, the recommended first variant
is FLAIR-only because the pretraining checkpoint is single-channel. Use the
custom Task 1 preprocessing module so the saved tensor channel count and
`dataset.json` metadata match the selected modalities:

```sh
cd "$ASPARAGUS_SOURCE"
unzip -n Task_1.zip -d Task_1

uv run asp_process \
  --dataset CLS002_FOMO26_Infarct_CUSTOM \
  --task_name CLS002_FOMO26_Infarct_FLAIR \
  --modalities flair \
  --save_as_tensor \
  --num_workers 4
uv run asp_split --dataset CLS002_FOMO26_Infarct_FLAIR --vals 80 10 10
```

`CLS002_FOMO26_Infarct_CUSTOM` can generate other modality-specific Task 1
datasets by changing `--task_name` and `--modalities`, for example `flair dwi`
or `flair adc dwi`. The selected modality order is the saved channel order, and
the generated `dataset.json` records matching `metadata.n_modalities`,
`metadata.modalities`, and `metadata.channel_names`.

#### Task 3: brain age regression

Task 3 is `REGR002_FOMO26_BrainAge`:

```sh
cd "$ASPARAGUS_SOURCE"
unzip -n Task_3.zip -d Task_3

uv run asp_process --dataset REGR002 --save_as_tensor --num_workers 4
uv run asp_split --dataset REGR002_FOMO26_BrainAge --vals 80 10 10
```

The `asp_split --vals 80 10 10` command writes both `split_80_10_10.json` and
`TEST_80_10_10.json` under the processed task directory.

#### Task 5: polymicrogyria classification

Task 5 is `CLS003_FOMO26_Polymicrogyria`. The organizers provide a standalone
extractor, `Task_5_extract.py`, which expects
`Zhang_Lingfeng_2022_PPMR_Dataset.zip` in its current working directory and
writes `Task_5/` there. Use the repo wrapper to run extraction, asparagus
preprocessing, and split creation:

```sh
scripts/eval_preprocess_task5.sh
```

This requires both organizer files to already be in `$ASPARAGUS_SOURCE`:

```text
$ASPARAGUS_SOURCE/Task_5_extract.py
$ASPARAGUS_SOURCE/Zhang_Lingfeng_2022_PPMR_Dataset.zip
```

Set `TASK5_NUM_WORKERS` to override the default `asp_process --num_workers 4`.

- [TODO] To reduce the number of necessary steps, the processed data from the previous step will be moved to HF so no local script running is needed.

### 2. Convert the pretrain checkpoint
```python
from asparagus_bridge.checkpoint import convert_checkpoint
convert_checkpoint("smri_mae", "runs/mae/checkpoint-last.pth", "runs/mae/asparagus.ckpt")
```

Register additional model converters in `asparagus_bridge.checkpoint.CONVERTERS`.

Use the converted `runs/mae/asparagus.ckpt` path in the finetuning and probing
commands below.

### 3. Per-task finetune and eval

`asp_finetune_cls` and `asp_finetune_reg` run test/eval after training. The
prediction JSON is written under the Hydra run directory, for example:

```text
predictions/<task>__TEST_80_10_10__best.json
```

#### Classification

Task 1 FLAIR-only smoke test:

```sh
uv run asp_finetune_cls --config-name finetuning/smoke_test_cls_task_1_flair_modality.yaml
```

[smoke_test_cls_task_1_flair_modality.yaml](./configs/finetuning/smoke_test_cls_task_1_flair_modality.yaml)
can be used as a reference for Task 1 classification runs.

The same task can also be run with CLI overrides:

```sh
uv run asp_finetune_cls \
  task=CLS002_FOMO26_Infarct_FLAIR \
  +model=smri_mae \
  checkpoint_path=runs/mae/asparagus.ckpt \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10
```

#### Regression

Task 3 brain age smoke test:

```sh
uv run asp_finetune_reg --config-name finetuning/smoke_test_bag_task3.yaml
```

[smoke_test_bag_task3.yaml](./configs/finetuning/smoke_test_bag_task3.yaml)
can be used as a reference for Task 3 regression runs.

#### Segmentation

TBD

#### Linear probing

TBD.

### 4. Multi-task finetune eval - not tested yet

```sh
FOMO_CLS_TASKS="CLS002_FOMO26_Infarct_FLAIR" \
FOMO_REG_TASKS="REGR002_FOMO26_BrainAge" \
  scripts/eval_fomo26.sh smri_mae runs/mae/checkpoint-last.pth
```
