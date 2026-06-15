from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from eval.datasets.registry import create_dataset, list_datasets
from eval.models.registry import create_model, list_models
from eval.utils import load_config, select_representation, send_batch

DEFAULT_CONFIG = Path(__file__).parent / "config/default_linear.yaml"

def _rmse(targets, preds):
    return float(np.sqrt(sklearn.metrics.mean_squared_error(targets, preds)))


CLF_METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
    "bacc": sklearn.metrics.balanced_accuracy_score,
}
REG_METRICS = {
    "r2": sklearn.metrics.r2_score,
    "mae": sklearn.metrics.mean_absolute_error,
    "rmse": _rmse,
}


@torch.inference_mode()
def extract_features(backbone, datasets, representation, cfg, device):
    backbone.eval().to(device)
    result = {}
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False)
        features, targets = [], []
        for batch in loader:
            batch = send_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg.amp and device.type == "cuda"):
                embeds = select_representation(backbone(batch), representation)
            features.append(embeds.cpu().float().numpy())
            targets.append(batch["target"].cpu().numpy())
        result[split] = {
            "features": np.concatenate(features),
            "target": np.concatenate(targets),
        }
    return result


def _candidate_Cs(Cs):
    """Mirror LogisticRegressionCV: an int means that many log-spaced C values."""
    if isinstance(Cs, int):
        return np.logspace(-4, 4, Cs)
    return np.asarray(list(Cs), dtype=float)


def evaluate(cfg, task, features_dict, targets_dict):
    clf = task.type == "classification"
    metric_fns = CLF_METRICS if clf else REG_METRICS
    metric_names = list(cfg.metrics) if clf else ["r2", "mae", "rmse"]
    cv_metric = cfg.cv_metric if clf else "r2"
    if clf:
        sweep = [(c, LogisticRegression(C=c, max_iter=cfg.max_iter)) for c in _candidate_Cs(cfg.Cs)]
    else:
        sweep = [(a, Ridge(alpha=a)) for a in np.logspace(-1, 5, cfg.get("alphas", 10))]

    scaler = StandardScaler().fit(features_dict["train"])
    scaled = {split: scaler.transform(features_dict[split]) for split in features_dict}

    # Pick the regularization param on the curated validation split (no random CV).
    best_param, best_est, best_score = None, None, -float("inf")
    for param, est in sweep:
        est.fit(scaled["train"], targets_dict["train"])
        score = metric_fns[cv_metric](targets_dict["validation"], est.predict(scaled["validation"]))
        if score > best_score:
            best_param, best_est, best_score = param, est, score

    header = {"model": cfg.model, "representation": cfg.representation,
              "dataset": cfg.dataset, "reg_param": best_param}
    rows = []
    for split in features_dict:
        preds = best_est.predict(scaled[split])
        record = {**header, "split": split}
        ci = bootstrap_ci(cfg, metric_fns, metric_names, preds, targets_dict[split], stratify=clf)
        for metric in metric_names:
            record[metric] = float(metric_fns[metric](targets_dict[split], preds))
            record[f"{metric}_std"] = ci[metric]["std"]
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_ci(cfg, metric_fns, metric_names, preds, targets, stratify):
    rng = sklearn.utils.check_random_state(cfg.seed)
    strat = targets if stratify else None
    scores = defaultdict(list)
    for _ in range(500):
        p, t = sklearn.utils.resample(preds, targets, random_state=rng, stratify=strat)
        for metric in metric_names:
            scores[metric].append(metric_fns[metric](t, p))
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in scores.items()}


def main(cfg):
    device = torch.device(cfg.device)
    backbone = create_model(cfg.model, **OmegaConf.to_container(cfg.model_kwargs))
    datasets, task = create_dataset(cfg.dataset, img_size=cfg.img_size, map_workers=cfg.map_workers,
                                    **OmegaConf.to_container(cfg.dataset_kwargs))

    head = "logistic" if task.type == "classification" else "ridge"
    output = Path(cfg.output_root) / cfg.name_prefix / f"{cfg.dataset}__{cfg.model}__{cfg.representation}__{head}"
    output.mkdir(parents=True, exist_ok=True)

    data = extract_features(backbone, datasets, cfg.representation, cfg, device)

    features = {split: data[split]["features"] for split in data}
    targets = {split: data[split]["target"] for split in data}

    table = evaluate(cfg, task, features, targets)
    print(table.to_markdown(index=False, floatfmt=".5g"))

    table.to_csv(output / "eval_table.csv", index=False)
    OmegaConf.save(cfg, output / "config.yaml")
    return table


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list_models())
    parser.add_argument("representation", choices=["cls", "patch"])
    parser.add_argument("dataset", choices=list_datasets())
    parser.add_argument("--config")
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    cfg = load_config(DEFAULT_CONFIG, args, {
        "model": args.model, "representation": args.representation, "dataset": args.dataset,
    })
    main(cfg)


if __name__ == "__main__": cli()
