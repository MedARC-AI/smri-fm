"""Collate the global-probe sweep into one markdown table of models x tasks.

Cells are `mean [ci_low, ci_high]` for the task's headline metric: AUROC for classification,
Pearson r for regression. Both are higher-better on roughly [0, 1], so they share a table.

The last columns aggregate over tasks. Win rate is the usual tournament score: per task each
model plays every other, taking 1 for a reliable win (its point estimate clears the opponent's
CI), 0.5 for an inconclusive pair, 0 for a loss, averaged over opponents and then over tasks.
0.5 is middle of the pack. The CIs are marginal, so the comparison is unpaired.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "output"

MODELS = [
    "random_features",
    # "random_unet",
    "synthseg",
    "neurojepa",
    "neurovfm",
    "smri_mae_vitl_fomo300",
]

TASKS = [
    # clinical
    "abide_autism_control",
    "adhd200_adhd_control",
    "adni_ad_cn",
    "adni_amyloid_centiloid",
    "adni_tau_suvr",
    "cnp_adhd_control",
    "cnp_schz_bipolar_control",
    "fomo_task1_infarct",
    "fomo_task5_polymicrogyria",
    "ppmi_pd_cn",
    "ppmi_pd_prodromal",
    # age
    "adni_age",
    "ppmi_age",
    "dlbs_age",
    "fomo_task3_age",
    # sex
    # "adni_sex",
    # "cnp_sex",
    # "dlbs_sex",
]

# sMRI MAE checkpoints, keyed by the `smri_mae_<suffix>__<task>` output-dir suffix.
CONFIGS = [
    "vitb_12k",
    "vitb_52k",
    "vitl_12k",
    "vitl_52k",
    "vitl_fomo300",
    "vitl_fomo300_tr",
]

MODEL_NAMES = {
    "random_features": "Rand features",
    "random_unet": "Rand U-Net",
    "neurojepa": "Neuro-JEPA",
    "neurovfm": "NeuroVFM",
    "synthseg": "SynthSeg",
    "smri_mae_vitl_fomo300": "sMRI MAE (ViT-L, FOMO300)",
}

TASK_NAMES = {
    "abide_autism_control": "ABIDE (ASD)",
    "adhd200_adhd_control": "ADHD-200 (ADHD)",
    "adni_ad_cn": "ADNI (AD)",
    "adni_age": "ADNI (Age)",
    "adni_amyloid_centiloid": "ADNI (Amy)",
    "adni_sex": "ADNI (Sex)",
    "adni_tau_suvr": "ADNI (Tau)",
    "cnp_adhd_control": "CNP (ADHD)",
    "cnp_schz_bipolar_control": "CNP (SCHZ+BP)",
    "cnp_sex": "CNP (Sex)",
    "dlbs_age": "DLBS (Age)",
    "dlbs_sex": "DLBS (Sex)",
    "fomo_task1_infarct": "FOMO (Infarct)",
    "fomo_task3_age": "FOMO (Age)",
    "fomo_task5_polymicrogyria": "FOMO (PMG)",
    "ppmi_age": "PPMI (Age)",
    "ppmi_pd_cn": "PPMI (PD vs CN)",
    "ppmi_pd_prodromal": "PPMI (PD vs Pdml)",
}

CONFIG_NAMES = {
    "vitb_12k": "ViT-B, 12k",
    "vitb_52k": "ViT-B, 52k",
    "vitl_12k": "ViT-L, 12k",
    "vitl_52k": "ViT-L, 52k",
    "vitl_fomo300": "ViT-L, FOMO300",
    "vitl_fomo300_tr": "ViT-L, FOMO300, ZYX",
}

# Headline metric per task type.
METRICS = {"auroc": "AUC", "pearson_r": "r"}

# Append each cell's win rate on that task, to spot check the aggregate.
SHOW_CELL_WIN_RATE = False


def load_runs(out_dir: Path) -> pd.DataFrame:
    """One row per run: model, config, task, n, headline metric and its mean/CI.

    A run's `config` is whatever its output dir carries beyond the model name, which is how the
    sMRI MAE checkpoints are told apart -- they all log `model: smri_mae`.
    """
    rows = []
    for metrics_path in sorted(out_dir.glob("*/metrics.jsonl")):
        run = json.loads(metrics_path.read_text().splitlines()[-1])
        log = (metrics_path.parent / "log.txt").read_text()
        (metric,) = [key for key in METRICS if key in run]
        model = metrics_path.parent.name.split("__")[0]
        base_model = run["model"]
        rows.append(
            {
                "model": model,
                "config": model.removeprefix(base_model).lstrip("_"),
                "base_model": base_model,
                "task": run["task"],
                "n": int(re.search(r"dataset: (\d+) samples", log).group(1)),
                "metric": metric,
                "mean": run[metric],
                "ci_low": run[f"{metric}_ci_low"],
                "ci_high": run[f"{metric}_ci_high"],
            }
        )
    return pd.DataFrame(rows)


def win_rates(runs: pd.DataFrame, row_col: str) -> pd.Series:
    """Per (row, task), that row's win rate over the other rows on the task.

    Both metrics are higher-better, so one comparison covers regression and classification.
    """
    pairs = runs.merge(runs, on="task", suffixes=("", "_opp"))
    pairs = pairs[pairs[row_col] != pairs[f"{row_col}_opp"]]
    win = pairs["mean"] > pairs["ci_high_opp"]
    loss = pairs["mean_opp"] > pairs["ci_high"]
    pairs["points"] = np.where(win, 1.0, np.where(loss, 0.0, 0.5))
    return pairs.groupby([row_col, "task"])["points"].mean()


def format_cell(row: pd.Series) -> str:
    text = f"{row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
    if SHOW_CELL_WIN_RATE:
        text += f" ({row['win_rate']:.2f})"
    return f"**{text}**" if row["best"] else text


def markdown_table(runs: pd.DataFrame, rows: list[str], names: dict[str, str], row_col: str) -> str:
    """One row per entry of `rows`, drawn from the `row_col` column; scores over TASKS.

    Win rate, rank and the bolded best are all relative to the rows in this table only.
    """
    runs = runs[runs[row_col].isin(rows) & runs["task"].isin(TASKS)]
    runs = runs.merge(win_rates(runs, row_col).rename("win_rate"), on=[row_col, "task"])
    runs["rank"] = runs.groupby("task")["mean"].rank(ascending=False)
    runs["best"] = runs["mean"] == runs.groupby("task")["mean"].transform("max")
    runs["cell"] = runs.apply(format_cell, axis=1)

    labels = runs.drop_duplicates("task").set_index("task")
    tasks = [task for task in TASKS if task in labels.index]
    table = runs.pivot(index=row_col, columns="task", values="cell").reindex(
        index=rows, columns=tasks
    )
    aggregate = runs.groupby(row_col)[["win_rate", "rank"]].mean().reindex(rows)
    table["win_rate"] = aggregate["win_rate"].map("{:.2f}".format)
    table["rank"] = aggregate["rank"].map("{:.2f}".format)

    table.columns = [
        f"{TASK_NAMES[task]} ({METRICS[labels.loc[task, 'metric']]}, n={labels.loc[task, 'n']})"
        for task in tasks
    ] + ["Win rate", "Mean rank"]
    table.index = [names[row] for row in rows]
    return table.fillna("--").rename_axis(index=row_col, columns=None).to_markdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    runs = load_runs(args.out_dir)
    print("### Backbones\n")
    print(markdown_table(runs, MODELS, MODEL_NAMES, row_col="model"))
    print("\n### sMRI MAE checkpoints\n")
    print(markdown_table(runs, CONFIGS, CONFIG_NAMES, row_col="config"))
