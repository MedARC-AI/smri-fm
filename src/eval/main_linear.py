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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from eval.datasets.registry import create_dataset, list_datasets
from eval.models.registry import create_model, list_models
from eval.utils import load_config, prepare_datasets, select_representation, send_batch

DEFAULT_CONFIG = Path(__file__).parent / "config/default_linear.yaml"

METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
    "bacc": sklearn.metrics.balanced_accuracy_score,
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


def evaluate(cfg, features_dict, targets_dict):
    select_metric = METRICS[cfg.cv_metric]

    scaler = StandardScaler().fit(features_dict["train"])
    scaled = {split: scaler.transform(features_dict[split]) for split in features_dict}

    # Pick C on the curated validation split (no random CV, no subject leakage).
    best_C, best_clf, best_score = None, None, -float("inf")
    for C in _candidate_Cs(cfg.Cs):
        clf = LogisticRegression(C=float(C), max_iter=cfg.max_iter)
        clf.fit(scaled["train"], targets_dict["train"])
        score = select_metric(targets_dict["validation"], clf.predict(scaled["validation"]))
        if score > best_score:
            best_C, best_clf, best_score = float(C), clf, score

    header = {"model": cfg.model, "representation": cfg.representation,
              "dataset": cfg.dataset, "C": best_C}
    rows = []
    for split in features_dict:
        preds = best_clf.predict(scaled[split])
        record = {**header, "split": split}
        ci = bootstrap_ci(cfg, preds, targets_dict[split])
        for metric in cfg.metrics:
            record[metric] = float(METRICS[metric](targets_dict[split], preds))
            record[f"{metric}_std"] = ci[metric]["std"]
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_ci(cfg, preds, targets):
    rng = sklearn.utils.check_random_state(cfg.seed)
    scores = defaultdict(list)
    for _ in range(500):
        p, t = sklearn.utils.resample(preds, targets, random_state=rng, stratify=targets)
        for metric in cfg.metrics:
            scores[metric].append(METRICS[metric](t, p))
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in scores.items()}


def main(cfg):
    output = Path(cfg.output_root) / cfg.name_prefix / f"{cfg.dataset}__{cfg.model}__{cfg.representation}__logistic"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    transform, backbone = create_model(cfg.model, **OmegaConf.to_container(cfg.model_kwargs))
    raw = create_dataset(cfg.dataset, **OmegaConf.to_container(cfg.dataset_kwargs))
    datasets, task = prepare_datasets(cfg, raw, transform, cfg.model_kwargs.ckpt_path)
    data = extract_features(backbone, datasets, cfg.representation, cfg, device)

    features = {split: data[split]["features"] for split in data}
    targets = {split: data[split]["target"] for split in data}

    table = evaluate(cfg, features, targets)
    print(table.to_markdown(index=False, floatfmt=".5g"))

    table.to_csv(output / "eval_table.csv", index=False)
    OmegaConf.save(cfg, output / "config.yaml")
    return table


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list_models())
    parser.add_argument("representation", choices=["cls", "reg", "patch"])
    parser.add_argument("dataset", choices=list_datasets())
    parser.add_argument("--config")
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    cfg = load_config(DEFAULT_CONFIG, args, {
        "model": args.model, "representation": args.representation, "dataset": args.dataset,
    })
    main(cfg)


if __name__ == "__main__": cli()
