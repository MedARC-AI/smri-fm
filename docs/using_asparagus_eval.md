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

If your Asparagus data lives outside the repo, put the path overrides in a
repo-root `.env` file before sourcing the setup script:

```bash
cat > .env <<'EOF'
ASPARAGUS_SOURCE=/path/to/fomo26/raw/hf_data/datasets_downloaded_from_hf
ASPARAGUS_DATA=/path/to/asparagus/processed_data
ASPARAGUS_MODELS=/path/to/asparagus/models
ASPARAGUS_RESULTS=/path/to/asparagus/results
ASPARAGUS_RAW_LABELS=/path/to/asparagus/raw_labels
EOF

source scripts/setup_asparagus_env.sh
```

The setup script prints each final `ASPARAGUS_*` value. Variables present in
`.env` override the script defaults; variables omitted from `.env` use the
repo-local defaults from `scripts/setup_asparagus_env.sh`.

### Vendoring `asparagus_preprocessing`

If Task 3 has not already been converted to Asparagus format, this repo needs
`asparagus_preprocessing` in addition to Asparagus. Add it the same way
Asparagus is already added: as a vendored editable submodule.

```bash
git submodule add https://github.com/Sllambias/asparagus_preprocessing.git third_party/asparagus_preprocessing
```

Then add it to `pyproject.toml`:

```toml
dependencies = [
    ...
    "asparagus_preprocessing",
]

[tool.uv.sources]
asparagus = { path = "third_party/asparagus", editable = true }
asparagus_preprocessing = { path = "third_party/asparagus_preprocessing", editable = true }
```

Finally sync the environment:

```bash
uv sync
```

`asparagus_preprocessing` provides the CLI commands used below:
`asp_process`, `asp_split`, `asp_update_paths`, and `asp_register_dataset`.

## 2. Download Task 3 Finetuning Data

Use `$ASPARAGUS_SOURCE` for raw FOMO data. This keeps the raw FOMO layout
separate from processed Asparagus tensors and matches the path convention used
by `asparagus_preprocessing`.

For a local smoke test, a convenient repo-local location is:

```bash
export ASPARAGUS_SOURCE="$PWD/.scratch/fomo26/raw/hf_data/datasets_downloaded_from_hf"
mkdir -p "$ASPARAGUS_SOURCE"
```

If you already have a shared FOMO data directory, prefer that instead:

```bash
export ASPARAGUS_SOURCE=/path/to/fomo26/raw/hf_data/datasets_downloaded_from_hf
mkdir -p "$ASPARAGUS_SOURCE"
```

Download and extract the Task 3 archive into `$ASPARAGUS_SOURCE`:

```bash
cd "$ASPARAGUS_SOURCE"

wget -nc https://sid.erda.dk/share_redirect/fmeuvo1EdF/Task_3.zip
unzip -n Task_3.zip -d Task_3
```

After extraction, the Task 3 preprocessing script expects this directory to
exist:

```text
$ASPARAGUS_SOURCE/Task_3/Task_3
```

That directory should contain the Task 3 `preprocessed` images and matching
`labels` tree. The Asparagus script uses each image path like
`preprocessed/.../t1w.nii.gz` and finds its label by replacing `preprocessed`
with `labels` and `t1w.nii.gz` with `labels.txt`.

## 3. Prepare Full Task 3 Data

Use this section if you want to preprocess the full Task 3 dataset. For the
smallest local end-to-end smoke test, skip this section and use
**Option A: Process Only 10 Raw Task 3 Cases** in Section 4 instead.

If the raw FOMO26 Task 3 data has not been processed yet, make sure
`ASPARAGUS_SOURCE` points to the raw download location from Section 2 and run:

```bash
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

## 4. Create a Small Task 3 Subset

There are two useful subset modes. For a true end-to-end smoke test, use
Option A.

- **Fastest after full preprocessing:** process all Task 3 data once, then make
  tiny split files that point to only a few processed `.pt` samples.
- **Smallest end-to-end smoke test:** copy only a few raw Task 3 cases into a
  tiny raw tree, process only those cases, then train/evaluate on the resulting
  tiny processed dataset.

### Option A: Process Only 10 Raw Task 3 Cases

Start from the full extracted Task 3 archive in `$ASPARAGUS_SOURCE`, then create
a tiny raw source tree under `.scratch`. The script copies each selected
`t1w.nii.gz` plus its matching `labels.txt` while preserving the directory
layout expected by `REGR002_FOMO26_BrainAge.py`.

```bash
export FOMO26_FULL_SOURCE="$ASPARAGUS_SOURCE"
export FOMO26_TINY_SOURCE="$PWD/.scratch/fomo26_tiny/raw/hf_data/datasets_downloaded_from_hf"

uv run python - <<'PY'
import os
import random
import shutil
from pathlib import Path

full_source = Path(os.environ["FOMO26_FULL_SOURCE"]).resolve()
tiny_source = Path(os.environ["FOMO26_TINY_SOURCE"]).resolve()
full_task = full_source / "Task_3" / "Task_3"
tiny_task = tiny_source / "Task_3" / "Task_3"

images = sorted((full_task / "preprocessed").rglob("t1w.nii.gz"))
if len(images) < 10:
    raise RuntimeError(f"Need at least 10 Task 3 images, found {len(images)} in {full_task}")

if tiny_task.exists():
    raise RuntimeError(f"Refusing to overwrite existing tiny source: {tiny_task}")

rng = random.Random(42)
selected = images[:]
rng.shuffle(selected)
selected = selected[:10]

for image in selected:
    rel_image = image.relative_to(full_task)
    label = Path(str(image).replace("/preprocessed/", "/labels/")).with_name("labels.txt")
    if not label.exists():
        raise FileNotFoundError(f"Missing label for {image}: {label}")

    rel_label = label.relative_to(full_task)
    out_image = tiny_task / rel_image
    out_label = tiny_task / rel_label
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, out_image)
    shutil.copy2(label, out_label)

print(f"Wrote tiny raw Task 3 source: {tiny_task}")
print("Selected cases:")
for image in selected:
    print(" ", image.relative_to(full_task))
PY
```

Point Asparagus at separate tiny output directories so the smoke run cannot mix
with a full Task 3 preprocessing run:

```bash
export ASPARAGUS_SOURCE="$FOMO26_TINY_SOURCE"
export ASPARAGUS_DATA="$PWD/.scratch/fomo26_tiny/processed_data"
export ASPARAGUS_RAW_LABELS="$PWD/.scratch/fomo26_tiny/raw_labels"
export ASPARAGUS_MODELS="$PWD/.scratch/fomo26_tiny/models"
export ASPARAGUS_RESULTS="$PWD/.scratch/fomo26_tiny/results"

mkdir -p "$ASPARAGUS_DATA" "$ASPARAGUS_RAW_LABELS" "$ASPARAGUS_MODELS" "$ASPARAGUS_RESULTS"

uv run asp_process --dataset REGR002 --save_as_tensor --num_workers 2
```

Then create explicit tiny train/val/test split files:

```bash
uv run python - <<'PY'
import json
import os
import random
from pathlib import Path

task_dir = Path(os.environ["ASPARAGUS_DATA"]) / "REGR002_FOMO26_BrainAge"
paths_path = task_dir / "paths.json"

paths = json.loads(paths_path.read_text())
if len(paths) != 10:
    raise RuntimeError(f"Expected 10 processed samples, found {len(paths)} in {paths_path}")

rng = random.Random(42)
rng.shuffle(paths)

train = paths[:8]
val = paths[8:9]
test = paths[9:10]

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

Use these split names in the finetuning command:

```bash
data.train_split=split_8_1_1_tiny10
data.test_split=TEST_8_1_1_tiny10
data.fold=0
```

### Option B: Reuse a Fully Processed Task 3 Dataset

Use this only if Section 3 has already completed successfully.

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

## 5. Create a Dummy Asparagus Checkpoint

This creates an Asparagus-format checkpoint with random weights from
`SmriMaeClsRegBackbone`. It is not pretrained. It exists only to validate the
Asparagus checkpoint format and the weight-loading path.

First try the same geometry as the MAE checkpoint was trained with:
`img_size=(208, 240, 208)` and `patch_size=8`.

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
    img_size=(208, 240, 208),
    patch_size=8,
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

## 6. Run the Tiny Task 3 Smoke Experiment

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
  training.target_size=[208,240,208] \
  training.warmup_epochs=0 \
  logger.wandb_logging=false \
  logger.log_every_n_steps=1 \
  ++model._cls_net.img_size=[208,240,208] \
  ++model._cls_net.patch_size=8
```

Notes:

- `asp_finetune_reg` currently instantiates `cfg.model._cls_net`, not
  `_reg_net`. This works for `smri_mae` because both point to
  `SmriMaeClsRegBackbone`, but it is confusing and should eventually be cleaned
  up.
- The repo's default `smri-fm` MAE pretraining config uses
  `img_size=[208,240,208]` and `patch_size=8`; use this geometry first so the
  dummy checkpoint path matches the real converted-checkpoint path.
- If your local machine runs out of memory, regenerate the dummy checkpoint with
  `img_size=(64,64,64)` and `patch_size=16`, then rerun with matching
  `training.target_size=[64,64,64]`, `++model._cls_net.img_size=[64,64,64]`,
  and `++model._cls_net.patch_size=16`.

## 7. Expected Success Criteria

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

## 8. Next Step: Real Checkpoint Conversion

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
