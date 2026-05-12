# FOMO26 evals - proposal
This document proposes a structure for evaluating the models on FOMO26 datasets.

## Brainstorm
Evaluate models on [FOMO26 fine-tuning datasets](https://sid.erda.dk/cgi-sid/ls.py?share_id=fmeuvo1EdF).
The FOMO26 allows a maximum of 3 validat attempts per task per track, so it's important to have a good estimate on local evals before submitting.
[Asparagus](https://github.com/Sllambias/asparagus/tree/main) is a repository that should contain organizer's baseline evals implementation, but it's not ready yet. Should keep an eye on it. When they publish how it should be used for the FOMO26 challenge, we can pick parts of it. The [FOMO25](https://www.synapse.org/Synapse:syn64895667/wiki/633093) docs state that the code should be wrapped into Apptainer containers and follow a strict structure. The container verification platform should open on 05/15 and the submissions on 06/15, so this is not an immediate focus, but should validate that flow well before 06/15 (e.g. target 06/01?).

Fine tuning tasks and datasets:

| Task | Type | Public modalities | Public counts | Primary official metrics |
| --- | --- | --- | --- | --- |
| 1. Infarct classification | Image classification | FLAIR, DWI ADC/b1000, T2* or SWI | 21 finetune, 80 validation, 320 test | AUROC |
| 2. Meningioma segmentation | Binary segmentation | FLAIR, DWI b1000, T2* or SWI | 23 finetune, 25 validation, 107 test | Dice, NSD |
| 3. Brain age estimation | Regression | T1w and T2w | 200 finetune, 200 validation, 800 test | Absolute error, Pearson correlation |
| 4. Trigeminal neuralgia segmentation | Multiclass segmentation | T1w | 40 finetune, 40 validation, 160 test | Dice, NSD |
| 5. Polymicrogyria classification | Image classification | T1w | 92 finetune, 37 validation, 146 test | AUROC |
| 6. Linear probing | Embedding evaluation | Undisclosed MRI sequences | 500 validation, 5000 test; no finetune set | Macro OvR AUROC, macro OvR F1 |
| 7. Bias and fairness | Grouped embedding evaluation | Same as Task 6 | 500 validation, 5000 test; no finetune set | Maximum group-disparity versions of Task 6 metrics |

Since we don’t have separate validation and test sets, I suggest using k-fold cross-validation for the first 5 tasks, to estimate how well the model finetuned on the whole finetuning dataset would perform.

There evals should support:
1. fitting small heads on top of the frozen backbone - easier
2. doing a full fine-tune of the model


## Interface
The evals code should go into `src/evaluations`.
An evaluation run is defined with a config, roughtly looking like this:

```python
from fomo26_eval import run_evals, run_evals_from_config

def run_evals(
    model,
    profile="probe",
    tasks=None,
    output_dir="fomo26_runs",
    data_dir="fomo26_data",
    # More
)

# Expose passing params as a config
def run_evals_from_config(yaml_config):
    params = load_config(yaml_config)
    return run_evals(**params)
```

Config fields:

`model`:
- a thin wrapper around the backbone that standardizes inputs and outputs

`profile`:
- probe -> uses the model as frozen backbone
- full -> finetunes the backbone as well

`tasks`:
- list of tasks to use `1`, `2`, ..., `7`
for each tasks there's probably a specific implementation of eval

More fields should be added once the minimal example is working. The default values should be used (overriding them just for dev/debug/experimental reasons).
- which metrics to report
- how to perform the split
- seed
- ...


Output format:

- timestamped run directory under `fomo26_runs`, for example
  `fomo26_runs/20260512-091530Z__probe__tasks-3__dummy/`
- run-level `config.json`, `metrics.json`, and `run_metadata.json`
- per-task subdirectories such as `task_3_brain_age/`
- per-task `metrics.json`, `run_metadata.json`, and `predictions.csv` where applicable


## Plan of work
I suggest ordering the work like this:
1. support one task, and evaluate it in "probe" profile. Can use a dummy model as backbone (or one of the already pretrained models)
2. Implement other tasks, this can be done in parallel also. I asked codex to order them by complexity:
    ```
    Easiest to hardest for writing local evals:

    Task 3: Brain age estimation
    Scalar regression on T1w/T2w. Easiest output contract: one number per case. Metrics are straightforward: MAE and Pearson correlation. No segmentation, no thresholding, no class imbalance edge cases.

    Task 5: Polymicrogyria classification
    Binary classification on T1w. Output is one score/probability per case, metric is AUROC. Single modality makes data plumbing simpler than Task 1.

    Task 1: Infarct classification
    Also binary AUROC, but harder than Task 5 because input is multi-sequence clinical MRI: FLAIR, ADC/b1000, and T2*/SWI. The eval metric is easy; the modality handling and missing/variant sequence handling are the hard part.

    Task 6: Linear probing
    Mechanically simple once the embedding API exists: extract embeddings, train linear probes, report macro OvR AUROC/F1. But official labels, sequence types, and exact categories are undisclosed, so a local mimic needs configurable substitute labels. I’d implement the generic probe runner early, but treat official Task 6 mimic as incomplete until labels/specs exist.

    Task 7: Bias and fairness
    Builds on Task 6, then adds group-wise metric computation and disparity aggregation. Not hard mathematically, but it depends on group labels, enough examples per group/class, and careful handling of undefined AUROC/F1 cases.

    Task 2: Meningioma segmentation
    Binary segmentation. Harder because outputs are dense NIfTI masks/probability maps, metrics include Dice and NSD, and evaluation must handle geometry, spacing, affine/header consistency, connected components/postprocessing, and surface-distance edge cases.

    Task 4: Trigeminal neuralgia segmentation
    Hardest. Multiclass segmentation of small anatomical structures around nerve/vessels. Same dense-output and geometry burden as Task 2, plus multiclass labels, tiny structures, per-class Dice/NSD, label-set validation, and likely more sensitivity to resampling errors.

    Pragmatic implementation order: 3 → 5 → 1 → 6 → 7 → 2 → 4.

    If the goal is reusable infrastructure rather than exact task completion, build the generic embedding/probe backend before or alongside Tasks 3/5, because it supports Tasks 3, 5, 6, and 7.
    ```

3. Establish FOMO26 baseline metrics for every task
4. Support containerizig the models and submitting to the evals pipeline
5. Support `profile = "full"`
    This requieres fine-tuning code. It should work as a standalone feature, without evals. Evals can consume it.




## TODOs:
- [ ] There's a ton of ways to implemnt this. I'd like to use code, reporting and config structure similar to an existing project, so it's easier to maintain. I'm aware of the following projects, I'll try to follow the pattern from: https://github.com/clane9/Brainmarks/tree/dev/clane9/src/fmri_fm_eval
