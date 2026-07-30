"""Compare matched XYZ and legacy ZYX ADNI-mini diversity evaluations."""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


OLD_ROOT = Path("/admin/home/mihir.neal/smri-fm/output/eval_linear")
NEW_ROOT = Path("/admin/home/mihir.neal/smri-fm-adni-med/output/eval_linear")
OLD_TAG = "adni_light_wider_reg_fomolong_vitb_diversity_e099_20260729"
NEW_TAG = "adni_mini_xyz_pinned_wider_reg_fomolong_vitb_diversity_e099_20260730"
OUTPUT_PREFIX = NEW_ROOT / "adni_xyz_vs_zyx_fomolong_vitb_e099_20260730"
SUBJECT_COUNTS = (12000, 25000, 61051)

TASKS = {
    "adni_age": ("sanity", "r2"),
    "adni_sex": ("sanity", "bacc"),
    "adni_synthseg_volumes": ("sanity", "r2"),
    "adni_ad_cn": ("diagnosis_conversion", "bacc"),
    "adni_cn_mci_ad": ("diagnosis_conversion", "bacc"),
    "adni_ad_cn_bag": ("diagnosis_conversion", "pearson_r"),
    "adni_mci_conversion_3y": ("diagnosis_conversion", "bacc"),
    "adni_cn_aplus_to_mci_ad_3y": ("diagnosis_conversion", "bacc"),
    "adni_adas13_slope_48m": ("full_longitudinal", "r2"),
    "adni_ravlt_delayed_slope_48m": ("full_longitudinal", "r2"),
    "adni_ravlt_learning_slope_48m": ("full_longitudinal", "r2"),
    "adni_hippocampus_slope_48m": ("full_longitudinal", "r2"),
    "adni_ventricle_slope_48m": ("full_longitudinal", "r2"),
    "adni_entorhinal_thickness_slope_48m": ("full_longitudinal", "r2"),
    "adni_cn_aplus_adas13_slope_48m": ("cn_aplus_longitudinal", "r2"),
    "adni_cn_aplus_ravlt_delayed_slope_48m": ("cn_aplus_longitudinal", "r2"),
    "adni_cn_aplus_ravlt_learning_slope_48m": ("cn_aplus_longitudinal", "r2"),
    "adni_cn_aplus_hippocampus_slope_48m": ("cn_aplus_longitudinal", "r2"),
    "adni_cn_aplus_ventricle_slope_48m": ("cn_aplus_longitudinal", "r2"),
    "adni_cn_aplus_entorhinal_thickness_slope_48m": (
        "cn_aplus_longitudinal",
        "r2",
    ),
    "adni_amyloid_centiloid": ("biomarkers", "r2"),
    "adni_tau_suvr": ("biomarkers", "r2"),
    "adni_csf_abeta": ("biomarkers", "spearman_r"),
    "adni_csf_ptau": ("biomarkers", "spearman_r"),
    "adni_csf_ttau": ("biomarkers", "spearman_r"),
}


def result_path(root: Path, task: str, tag: str, subjects: int) -> Path:
    model_id = f"fomolong_v1_vitb_n{subjects}_e099"
    run = f"smri_mae__{task}__patch__{tag}__{model_id}"
    return root / run / "metrics.json"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    fold_records = []
    for subjects in SUBJECT_COUNTS:
        for task, (group, metric) in TASKS.items():
            old = json.loads(result_path(OLD_ROOT, task, OLD_TAG, subjects).read_text())
            new = json.loads(result_path(NEW_ROOT, task, NEW_TAG, subjects).read_text())
            old_value = float(old["summary"][metric])
            new_value = float(new["summary"][metric])
            records.append(
                {
                    "subjects": subjects,
                    "task": task,
                    "group": group,
                    "metric": metric,
                    "zyx": old_value,
                    "xyz": new_value,
                    "xyz_minus_zyx": new_value - old_value,
                }
            )
            for fold, (old_fold, new_fold) in enumerate(
                zip(old["folds"], new["folds"], strict=True)
            ):
                fold_records.append(
                    {
                        "subjects": subjects,
                        "task": task,
                        "group": group,
                        "metric": metric,
                        "fold": fold,
                        "zyx": float(old_fold[metric]),
                        "xyz": float(new_fold[metric]),
                        "xyz_minus_zyx": float(new_fold[metric])
                        - float(old_fold[metric]),
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(fold_records)


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    delta = "xyz_minus_zyx"
    return (
        frame.groupby(keys, sort=False)
        .agg(
            comparisons=(delta, "size"),
            xyz_wins=(delta, lambda values: int((values > 0).sum())),
            xyz_losses=(delta, lambda values: int((values < 0).sum())),
            xyz_wins_ge_0p01=(delta, lambda values: int((values >= 0.01).sum())),
            xyz_losses_le_neg_0p01=(
                delta,
                lambda values: int((values <= -0.01).sum()),
            ),
            median_delta=(delta, "median"),
            mean_delta=(delta, "mean"),
            median_absolute_delta=(delta, lambda values: values.abs().median()),
            mean_absolute_delta=(delta, lambda values: values.abs().mean()),
        )
        .reset_index()
    )


def task_summary(results: pd.DataFrame) -> pd.DataFrame:
    delta = "xyz_minus_zyx"
    return (
        results.groupby(["group", "task", "metric"], sort=False)
        .agg(
            models=(delta, "size"),
            xyz_wins=(delta, lambda values: int((values > 0).sum())),
            xyz_losses=(delta, lambda values: int((values < 0).sum())),
            median_zyx=("zyx", "median"),
            median_xyz=("xyz", "median"),
            median_delta=(delta, "median"),
            mean_delta=(delta, "mean"),
            min_delta=(delta, "min"),
            max_delta=(delta, "max"),
        )
        .reset_index()
        .sort_values("median_delta", ascending=False)
    )


def diversity_scaling(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summaries = []
    for orientation in ("zyx", "xyz"):
        pivot = results.pivot(
            index=["group", "task", "metric"],
            columns="subjects",
            values=orientation,
        )
        pivot["high_minus_low"] = pivot[61051] - pivot[12000]
        pivot["orientation"] = orientation
        records.extend(pivot.reset_index().to_dict("records"))
        delta = pivot["high_minus_low"]
        summaries.append(
            {
                "orientation": orientation,
                "tasks": len(delta),
                "higher_diversity_wins": int((delta > 0).sum()),
                "higher_diversity_losses": int((delta < 0).sum()),
                "median_high_minus_low": delta.median(),
                "mean_high_minus_low": delta.mean(),
            }
        )
    scores = pd.DataFrame(records)
    change = scores.pivot(
        index=["group", "task", "metric"],
        columns="orientation",
        values="high_minus_low",
    ).reset_index()
    change["xyz_minus_zyx_scaling"] = change["xyz"] - change["zyx"]
    return change, pd.DataFrame(summaries)


def main() -> None:
    results, folds = load_results()
    by_model = summarize(results, ["subjects"])
    by_group = summarize(results, ["group"])
    tasks = task_summary(results)
    scaling_change, scaling_summary = diversity_scaling(results)
    _, p_value = wilcoxon(results["xyz_minus_zyx"])

    results.to_csv(f"{OUTPUT_PREFIX}_matched_primary_metrics.csv", index=False)
    folds.to_csv(f"{OUTPUT_PREFIX}_matched_folds.csv", index=False)
    by_model.to_csv(f"{OUTPUT_PREFIX}_model_summary.csv", index=False)
    by_group.to_csv(f"{OUTPUT_PREFIX}_group_summary.csv", index=False)
    tasks.to_csv(f"{OUTPUT_PREFIX}_task_summary.csv", index=False)
    scaling_change.to_csv(f"{OUTPUT_PREFIX}_diversity_scaling_change.csv", index=False)
    scaling_summary.to_csv(
        f"{OUTPUT_PREFIX}_diversity_scaling_summary.csv",
        index=False,
    )

    formatter = lambda value: f"{value:+.4f}"
    print("\nXYZ minus legacy ZYX by model")
    print(by_model.to_string(index=False, float_format=formatter))
    print("\nXYZ minus legacy ZYX by task group")
    print(by_group.to_string(index=False, float_format=formatter))
    print("\nTasks across the three models")
    print(tasks.to_string(index=False, float_format=formatter))
    print("\nLargest individual gains")
    print(
        results.nlargest(10, "xyz_minus_zyx").to_string(
            index=False,
            float_format=formatter,
        )
    )
    print("\nLargest individual losses")
    print(
        results.nsmallest(10, "xyz_minus_zyx").to_string(
            index=False,
            float_format=formatter,
        )
    )
    print(f"\nPaired Wilcoxon over 75 primary metrics: p={p_value:.6g}")
    print("\n61,051 minus 12,000 diversity scaling")
    print(scaling_summary.to_string(index=False, float_format=formatter))
    print("\nChange in task-level diversity scaling")
    print(
        scaling_change.sort_values(
            "xyz_minus_zyx_scaling",
            ascending=False,
        ).to_string(index=False, float_format=formatter)
    )


if __name__ == "__main__":
    main()
