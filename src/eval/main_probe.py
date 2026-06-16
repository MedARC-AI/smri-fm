from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from eval.classifiers import ClassifierGrid, create_classifier, list_classifiers
from eval.datasets.registry import create_dataset, list_datasets
from eval.models.registry import create_model, list_models
from eval.utils import (
    CLF_METRICS,
    REG_METRICS,
    get_sha,
    infinite_loader,
    load_config,
    make_lr_schedule,
    random_seed,
    select_representation,
    send_batch,
    update_lr,
    update_wd,
)

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


# ---------------------------------------------------------------------------
# Metrics (classification or regression)
# ---------------------------------------------------------------------------

def metric_values(kind, metric_names, targets, preds) -> dict[str, float]:
    table = CLF_METRICS if kind == "classification" else REG_METRICS
    return {m: float(table[m](targets, preds)) for m in metric_names}


def grid_metrics(kind, metric_names, preds, targets, num_clf) -> list[dict[str, float]]:
    out = []
    for ii in range(num_clf):
        pred_ii = preds[:, ii] if kind == "classification" else preds[..., ii]
        out.append(metric_values(kind, metric_names, targets, pred_ii))
    return out


def bootstrap_ci(cfg, kind, preds, targets) -> dict[str, dict[str, float]]:
    rng = sklearn.utils.check_random_state(cfg.seed)
    stratify = targets if kind == "classification" else None
    scores = defaultdict(list)
    for _ in range(500):
        p, t = sklearn.utils.resample(preds, targets, random_state=rng, stratify=stratify)
        for metric, value in metric_values(kind, cfg.metrics, t, p).items():
            scores[metric].append(value)
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in scores.items()}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def grid_loss(logits, target, kind, num_clf):
    """Per-classifier loss vector of shape [num_clf]. logits: [B, out_dim, num_clf]."""
    if kind == "classification":
        tgt = target.long().unsqueeze(-1).expand(-1, num_clf)
        loss = F.cross_entropy(logits, tgt, reduction="none")  # [B, num_clf]
        return loss.mean(dim=0)
    tgt = target.float()
    if tgt.ndim == 1:
        tgt = tgt.unsqueeze(-1)
    tgt = tgt.unsqueeze(-1).expand(-1, -1, num_clf)  # [B, out_dim, num_clf]
    loss = F.mse_loss(logits, tgt, reduction="none")
    return loss.reshape(-1, num_clf).mean(dim=0)


# ---------------------------------------------------------------------------
# Classifier sweep
# ---------------------------------------------------------------------------

def make_classifiers(cfg, embed_dim, out_dim):
    clf_fn = partial(create_classifier, name=cfg.classifier, in_dim=embed_dim,
                     out_dim=out_dim, **(OmegaConf.to_container(cfg.classifier_kwargs) or {}))
    classifiers, param_groups, init_state = {}, {}, None
    for lr_mult, wd_mult in product(cfg.lr_scale_grid, cfg.wd_scale_grid):
        clf = clf_fn()
        if init_state is None:
            init_state = clf.state_dict()
        else:
            clf.load_state_dict(init_state)  # every head shares the same init
        classifiers[(lr_mult, wd_mult)] = clf
        for name, param in clf.named_parameters():
            param_wd = 0.0 if (name.endswith(".bias") or "norm" in name) else wd_mult
            key = (lr_mult, param_wd)
            group = param_groups.setdefault(
                key, {"params": [], "lr_multiplier": lr_mult, "wd_multiplier": param_wd}
            )
            group["params"].append(param)
    return classifiers, list(param_groups.values())


def format_hparam(idx, hparam) -> str:
    lr_mult, wd_mult = hparam
    return f"{idx:03d}_lr{lr_mult:.1e}_wd{wd_mult:.1e}"


def best_hparam(cfg, model, val_metrics):
    metric, sign = cfg.cv_metric, 1.0
    if metric.startswith("neg_"):
        metric, sign = metric[4:], -1.0
    scores = [sign * val_metrics[ii][metric] for ii in range(len(model.hparams))]
    best = int(np.argmax(scores))
    return best, model.hparams[best], scores[best]


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(cfg, model, data_iter, optimizer, lr_schedule, epoch,
                    kind, num_clf, num_classes, device):
    model.train()
    model.backbone.eval()
    amp = cfg.amp and device.type == "cuda"
    diverge_threshold = 20 * math.log(num_classes) if kind == "classification" else float("inf")

    loss_sum = np.zeros(num_clf)
    update_count = 0
    optimizer.zero_grad(set_to_none=True)
    for step in range(cfg.steps_per_epoch):
        for _ in range(cfg.accum_iter):
            batch = send_batch(next(data_iter), device)
            target = batch.pop("target")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
                all_loss = grid_loss(logits, target, kind, num_clf)
                loss = all_loss.mean()
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()}, stopping")
            (loss / cfg.accum_iter).backward()
            loss_sum += all_loss.detach().float().cpu().numpy()

        for clf in model.classifiers:  # grad clip each head independently
            nn.utils.clip_grad_norm_(clf.parameters(), cfg.clip_grad)
        global_step = epoch * cfg.steps_per_epoch + step
        update_lr(optimizer.param_groups, float(lr_schedule[global_step]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_count += 1

        if kind == "classification":  # freeze + zero any diverged head
            mean_loss = loss_sum / (update_count * cfg.accum_iter)
            for ii, clf in enumerate(model.classifiers):
                if mean_loss[ii] > diverge_threshold and next(clf.parameters()).requires_grad:
                    print(f"  freezing diverged classifier {ii} {model.hparams[ii]} "
                          f"(loss={mean_loss[ii]:.2f})")
                    clf.requires_grad_(False)
                    for p in clf.parameters():
                        p.data.zero_()

    return loss_sum / (update_count * cfg.accum_iter)


@torch.inference_mode()
def evaluate_grid(cfg, model, loader, kind, num_clf, device):
    model.eval()
    amp = cfg.amp and device.type == "cuda"
    logits_all, targets_all = [], []
    for batch in loader:
        batch = send_batch(batch, device)
        target = batch.pop("target")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
        logits_all.append(logits.cpu().float())
        targets_all.append(target.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    losses = grid_loss(logits, targets, kind, num_clf).tolist()
    if kind == "classification":
        preds = logits.argmax(dim=1).numpy()  # [N, num_clf]
    else:
        preds = logits.numpy()  # [N, out_dim, num_clf]
    return preds, targets.numpy(), losses


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_model(output_dir, epoch, classifiers, optimizer, meta, is_best):
    payload = {"model": classifiers.state_dict(), "optimizer": optimizer.state_dict(),
               "epoch": epoch, "meta": meta}
    for label, save in (("last", True), ("best", is_best)):
        if not save:
            continue
        tmp = output_dir / f".tmp-checkpoint-{label}.pth"
        torch.save(payload, tmp)
        tmp.rename(output_dir / f"checkpoint-{label}.pth")


def load_model(output_dir, classifiers, optimizer):
    ckpt_path = output_dir / "checkpoint-last.pth"
    if not ckpt_path.exists():
        return 0, float("-inf")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifiers.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"resuming from epoch {ckpt['epoch'] + 1}")
    return ckpt["epoch"] + 1, ckpt["meta"].get("best_score", float("-inf"))


def evaluate_checkpoint(cfg, model, eval_loaders, kind, num_clf, device, label, output_dir):
    ckpt = torch.load(output_dir / f"checkpoint-{label}.pth", map_location="cpu", weights_only=False)
    model.classifiers.load_state_dict(ckpt["model"])
    meta = ckpt["meta"]
    hid, hparam = meta["hparam_id"], meta["hparam"]
    print(f"evaluating {label} checkpoint (epoch={meta['epoch']}, hparam={hparam})")

    rows = []
    for split, loader in eval_loaders.items():
        preds, targets, losses = evaluate_grid(cfg, model, loader, kind, num_clf, device)
        pred_best = preds[:, hid] if kind == "classification" else preds[..., hid]
        metrics = metric_values(kind, cfg.metrics, targets, pred_best)
        ci = bootstrap_ci(cfg, kind, pred_best, targets)
        record = {
            "model": cfg.model, "representation": cfg.representation,
            "classifier": cfg.classifier, "dataset": cfg.dataset,
            "ckpt": label, "epoch": meta["epoch"], "hparam_id": hid,
            "lr": hparam[0] * cfg.lr, "wd": hparam[1] * cfg.weight_decay,
            "split": split, "loss": losses[hid],
        }
        for metric in cfg.metrics:
            record[metric] = metrics[metric]
            record[f"{metric}_std"] = ci[metric]["std"]
        rows.append(record)
        np.savez(output_dir / f"preds_{label}_{split}.npz", preds=pred_best, targets=targets)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / f"eval_table_{label}.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    random_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_name = f"{cfg.dataset}__{cfg.model}__{cfg.representation}__{cfg.classifier}"
    output = Path(cfg.output_root) / cfg.name_prefix / run_name
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml")

    run = None
    if cfg.get("wandb"):
        import wandb
        run = wandb.init(entity=cfg.get("wandb_entity"),
                         project=cfg.get("wandb_project") or "smri-eval",
                         name=run_name, config=OmegaConf.to_container(cfg, resolve=True))

    print(f"probe eval — sha={get_sha()} — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    backbone = create_model(cfg.model, **OmegaConf.to_container(cfg.model_kwargs))
    backbone.requires_grad_(False).eval().to(device)
    datasets, task = create_dataset(cfg.dataset, img_size=cfg.img_size, map_workers=cfg.map_workers,
                                    **OmegaConf.to_container(cfg.dataset_kwargs))
    kind, out_dim = task.type, task.output_dim
    num_classes = out_dim if kind == "classification" else 1

    # balanced class sampling (classification only)
    sampler = None
    if kind == "classification" and cfg.get("balanced_sampling"):
        labels = np.asarray(datasets["train"][task.target])
        counts = np.bincount(labels, minlength=out_dim)
        class_weight = counts.max() / np.clip(counts, 1, None)
        sampler = WeightedRandomSampler(class_weight[labels], num_samples=len(labels))
        print(f"balanced sampling weights: {np.round(class_weight, 2)}")

    prefetch = cfg.prefetch_factor if cfg.num_workers else None
    train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=cfg.num_workers, prefetch_factor=prefetch, drop_last=True)
    eval_loaders = {split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, prefetch_factor=prefetch,
                                      drop_last=False)
                    for split, ds in datasets.items()}

    # embedding dim from one example
    example = send_batch(next(iter(DataLoader(datasets["train"], batch_size=1))), device)
    example.pop("target", None)
    with torch.inference_mode():
        embed_dim = select_representation(backbone(example), cfg.representation).shape[-1]

    classifiers, param_groups = make_classifiers(cfg, embed_dim, out_dim)
    model = ClassifierGrid(backbone, cfg.representation, classifiers).to(device)
    num_clf = len(model.classifiers)
    print(f"sweep of {num_clf} classifier head(s) over "
          f"lr_scale={list(cfg.lr_scale_grid)} x wd_scale={list(cfg.wd_scale_grid)}")

    update_lr(param_groups, cfg.lr)
    update_wd(param_groups, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    if not cfg.get("steps_per_epoch"):
        cfg.steps_per_epoch = max(len(train_loader) // cfg.accum_iter, 1)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    warmup_steps = cfg.warmup_epochs * cfg.steps_per_epoch
    lr_schedule = make_lr_schedule(cfg.lr, total_steps, warmup_steps, no_decay=cfg.get("no_decay", False))

    start_epoch, best_score = load_model(output, model.classifiers, optimizer)
    data_iter = infinite_loader(train_loader)
    start_time = time.monotonic()

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(cfg, model, data_iter, optimizer, lr_schedule,
                                     epoch, kind, num_clf, num_classes, device)
        val_preds, val_targets, val_losses = evaluate_grid(cfg, model, eval_loaders["validation"],
                                                           kind, num_clf, device)
        val_metrics = grid_metrics(kind, cfg.metrics, val_preds, val_targets, num_clf)
        hid, hparam, cv_score = best_hparam(cfg, model, val_metrics)

        print(f"epoch={epoch} train_loss={train_loss[hid]:.4g} "
              f"val_{cfg.cv_metric}={cv_score:.4g} best_hparam={hparam} ({hid:03d})")

        if run is not None:
            log = {f"val/{m}_best": val_metrics[hid][m] for m in cfg.metrics}
            log.update({"val/loss_best": val_losses[hid], "train/loss_best": float(train_loss[hid]),
                        "lr_best": hparam[0] * cfg.lr, "wd_best": hparam[1] * cfg.weight_decay})
            run.log(log, step=(epoch + 1) * cfg.steps_per_epoch)

        is_best = cv_score > best_score
        best_score = max(best_score, cv_score)
        meta = {"score": cv_score, "hparam": hparam, "hparam_id": hid,
                "epoch": epoch, "is_best": is_best, "best_score": best_score}
        save_model(output, epoch, model.classifiers, optimizer, meta, is_best)

    tables = {label: evaluate_checkpoint(cfg, model, eval_loaders, kind, num_clf, device, label, output)
              for label in ("best", "last")}
    preferred = "best" if cfg.get("early_stopping", True) else "last"
    tables[preferred].to_csv(output / "eval_table.csv", index=False)

    print(tables[preferred].to_markdown(index=False, floatfmt=".5g"))
    print(f"done in {datetime.timedelta(seconds=int(time.monotonic() - start_time))}")
    if run is not None:
        run.finish()
    return tables[preferred]


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list_models())
    parser.add_argument("representation", choices=["cls", "patch"])
    parser.add_argument("classifier", choices=list_classifiers())
    parser.add_argument("dataset", choices=list_datasets())
    parser.add_argument("--config")
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    cfg = load_config(DEFAULT_CONFIG, args, {"model": args.model, "representation": args.representation,
                                             "classifier": args.classifier, "dataset": args.dataset})
    main(cfg)


if __name__ == "__main__":
    cli()
