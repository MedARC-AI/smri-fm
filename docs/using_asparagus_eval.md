# Using Asparagus Eval Locally with `SmriMaeClsRegBackbone`

This guide is for a small local end-to-end smoke run of FOMO26 Task 3
(`REGR002_FOMO26_BrainAge`) through `asp_finetune_reg` using
`asparagus_bridge.models_smri_mae.SmriMaeClsRegBackbone`.

The first goal is to prove that the Asparagus data/config path runs on a tiny
subset with an Asparagus-format dummy checkpoint. After that works, the next
step is to add a real `smri-fm` checkpoint converter.

## 0. Current Status and Known Gaps

There is one important implementation gap in the current repo state:

- There is no converter yet from native `smri-fm` checkpoints to Asparagus
  checkpoints. Native `smri-fm` checkpoints use a top-level `model` key;
  Asparagus expects either `state_dict` or `network_weights`.

`asp_finetune_reg` does resolve `checkpoint_path` and passes the loaded
`weights` into `RegressionModule`, matching the classification fine-tuning
path. The weight-loading path is:

```text
checkpoint_path
-> resolve_checkpoint(cfg)
-> weights dict
-> RegressionModule(..., model=SmriMaeClsRegBackbone, weights=weights)
-> BaseModule.__init__
-> self.load_state_dict(weights)
```

Suggested approach for the checkpoint converter:

- Load a native checkpoint with top-level `model`.
- Strip `module.` prefixes if present.
- Keep encoder-compatible MAE keys only.
- Prefix converted keys with `model.encoder.` because Asparagus loads into the
  Lightning module state dict.
- Save `{"network_weights": converted_weights, "smri_fm_args": ckpt.get("args")}`.
- Fail if zero weights are converted.

## 1. Environment

From the `smri-fm` repo root:

```bash
uv sync
source scripts/setup_asparagus_env.sh
```

If your Asparagus data lives outside the repo, export the paths before sourcing
the setup script:

```bash
export ASPARAGUS_DATA=/path/to/asparagus/processed_data
export ASPARAGUS_MODELS=/path/to/asparagus/models
export ASPARAGUS_RESULTS=/path/to/asparagus/results
export ASPARAGUS_RAW_LABELS=/path/to/asparagus/raw_labels
source scripts/setup_asparagus_env.sh
```

If Task 3 has not already been converted to Asparagus format, install
`asparagus_preprocessing` separately. This repo currently vendors Asparagus but
not the preprocessing repository that provides `asp_process` and `asp_split`.

```bash
cd /path/to/shared/project/root
git clone https://github.com/Sllambias/asparagus_preprocessing.git
cd /Users/lukasecerovic/Documents/repos/sMRI/smri-fm
uv pip install -e /path/to/shared/project/root/asparagus_preprocessing
```

## 2. Prepare Task 3 Data

If the raw FOMO26 Task 3 data has not been processed yet, set `ASPARAGUS_SOURCE`
to the raw download location and run:

```bash
export ASPARAGUS_SOURCE=/path/to/raw/hf_data/datasets_downloaded_from_hf
uv run asp_process --dataset REGR002 --save_as_tensor --num_workers 4
uv run asp_split --dataset REGR002_FOMO26_BrainAge --vals 80 10 10
```

Expected processed files:

```text
$ASPARAGUS_DATA/REGR002_FOMO26_BrainAge/dataset.json
$ASPARAGUS_DATA/REGR002_FOMO26_BrainAge/paths.json
$ASPARAGUS_DATA/REGR002_FOMO26_BrainAge/split_80_10_10.json
$ASPARAGUS_DATA/REGR002_FOMO26_BrainAge/TEST_80_10_10.json
```

## 3. Create a 10-Sample Subset

Do not copy the tensor files. Create tiny split JSON files that point at 10
existing `.pt` samples:

```bash
uv run python - <<'PY'
import json
import os
import random
from pathlib import Path

task = "REGR002_FOMO26_BrainAge"
task_dir = Path(os.environ["ASPARAGUS_DATA"]) / task
paths_path = task_dir / "paths.json"

paths = json.loads(paths_path.read_text())
if len(paths) < 10:
    raise RuntimeError(f"Need at least 10 samples, found {len(paths)} in {paths_path}")

rng = random.Random(42)
rng.shuffle(paths)
subset = paths[:10]

train = subset[:8]
val = subset[8:9]
test = subset[9:10]

(task_dir / "split_8_1_1_tiny10.json").write_text(
    json.dumps([{"train": train, "val": val}], indent=2) + "\n"
)
(task_dir / "TEST_8_1_1_tiny10.json").write_text(
    json.dumps(test, indent=2) + "\n"
)

print(f"Wrote {task_dir / 'split_8_1_1_tiny10.json'}")
print(f"Wrote {task_dir / 'TEST_8_1_1_tiny10.json'}")
PY
```

Optional sanity check:

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path
import torch

task_dir = Path(os.environ["ASPARAGUS_DATA"]) / "REGR002_FOMO26_BrainAge"
split = json.loads((task_dir / "split_8_1_1_tiny10.json").read_text())[0]
sample_path = split["train"][0]
sample = torch.load(sample_path, map_location="cpu", weights_only=False)
print("sample:", sample_path)
print("image shape:", tuple(sample[0].shape))
print("label:", sample[1])
PY
```

## 4. Create a Dummy Asparagus Checkpoint

This creates an Asparagus-format checkpoint with random weights from
`SmriMaeClsRegBackbone`. It is not pretrained. It exists only to validate the
Asparagus checkpoint format and the weight-loading path.

For a fast local smoke test, use a small crop size:

```bash
mkdir -p .scratch/asparagus_eval

uv run python - <<'PY'
import json
import os
from pathlib import Path
import torch

from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone

task_dir = Path(os.environ["ASPARAGUS_DATA"]) / "REGR002_FOMO26_BrainAge"
dataset_json = json.loads((task_dir / "dataset.json").read_text())
metadata = dataset_json.get("metadata") or dataset_json.get("dataset_config")

model = SmriMaeClsRegBackbone(
    input_channels=metadata["n_modalities"],
    output_channels=metadata["n_classes"],
    img_size=(64, 64, 64),
    patch_size=16,
)

network_weights = {
    f"model.{key}": value.detach().cpu()
    for key, value in model.state_dict().items()
}

out = Path(".scratch/asparagus_eval/dummy_smri_mae_reg_network_weights.pth")
torch.save(
    {
        "network_weights": network_weights,
        "note": "Random SmriMaeClsRegBackbone weights for Asparagus smoke testing.",
    },
    out,
)
print(out.resolve())
PY
```

## 5. Run the Tiny Task 3 Smoke Experiment

Use `asp_finetune_reg`, Task 3, the tiny split files, and the bridge model
config. This command uses CPU-friendly settings and disables W&B:

```bash
uv run asp_finetune_reg \
  task=REGR002_FOMO26_BrainAge \
  +model=smri_mae \
  checkpoint_path="$PWD/.scratch/asparagus_eval/dummy_smri_mae_reg_network_weights.pth" \
  data.train_split=split_8_1_1_tiny10 \
  data.test_split=TEST_8_1_1_tiny10 \
  data.fold=0 \
  hardware.accelerator=cpu \
  hardware.num_devices=1 \
  hardware.num_workers=2 \
  hardware.precision=32-true \
  hardware.compile_mode=null \
  training.batch_size=1 \
  training.epochs=1 \
  training.limit_train_batches=1 \
  training.limit_val_batches=1 \
  training.target_size=[64,64,64] \
  training.warmup_epochs=0 \
  logger.wandb_logging=false \
  logger.log_every_n_steps=1 \
  ++model._cls_net.img_size=[64,64,64] \
  ++model._cls_net.patch_size=16
```

Notes:

- `asp_finetune_reg` currently instantiates `cfg.model._cls_net`, not
  `_reg_net`. This works for `smri_mae` because both point to
  `SmriMaeClsRegBackbone`, but it is confusing and should eventually be cleaned
  up.
- The small `[64,64,64]` geometry is for local smoke testing only. For a real
  converted `smri-fm` checkpoint, set `training.target_size`,
  `model._cls_net.img_size`, and `model._cls_net.patch_size` to match the
  checkpoint geometry.
- The repo's default `smri-fm` MAE pretraining config uses
  `img_size=[208,240,208]` and `patch_size=8`.

## 6. Expected Success Criteria

The smoke run is successful when:

- Hydra finds `+model=smri_mae` through `scripts/setup_asparagus_env.sh`.
- `checkpoint_path` resolves and the checkpoint is accepted as an Asparagus
  checkpoint with a top-level `network_weights` key.
- Asparagus reads the tiny Task 3 split and creates train/val/test dataloaders.
- One train batch, one val batch, and the final test pass complete.
- A prediction JSON is written under `$ASPARAGUS_MODELS/.../predictions/`.

The dummy checkpoint is randomly initialized, so this validates wiring and
checkpoint compatibility only. It is not expected to produce meaningful
regression metrics.

## 7. Next Step: Real Checkpoint Conversion

After the dummy checkpoint smoke path works, add a converter such as:

```text
src/asparagus_bridge/convert_checkpoint.py
```

Expected CLI shape:

```bash
uv run python -m asparagus_bridge.convert_checkpoint \
  --input /path/to/smri-fm/checkpoint-last.pth \
  --output .scratch/asparagus_eval/smri_fm_asparagus_network_weights.pth
```

Then rerun the smoke command with:

```bash
checkpoint_path="$PWD/.scratch/asparagus_eval/smri_fm_asparagus_network_weights.pth"
training.target_size=[208,240,208]
++model._cls_net.img_size=[208,240,208]
++model._cls_net.patch_size=8
```

The converter should leave the regression head randomly initialized and only
load compatible encoder weights.
