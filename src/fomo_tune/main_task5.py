"""FOMO task 5: polymicrogyria classification

Method (tune):

1. Extract patch features
2. Global average pool, or cortex pool
3. sklearn logistic regression

Protocol (fixed):

- 20 fold cross-validation
- AUROC metric on out-of-fold predictions
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fomo_tune.synthseg as synthseg
from fomo_tune.backbone import SmriMaeTransform, load_backbone
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]

# The cohort's controls are clipped and its cases are not, which scores AUROC 0.997 by
# itself. 133mm is the smallest extent any subject covers.
AP_EXTENT_MM = 133.0
CORTEX = (3, 42)  # SynthSeg's left/right cerebral cortex labels


@dataclass
class Config:
    task: str = "task5"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task5"
    device: str = "cuda"
    seed: int = 4466
    masking: str = "zero"
    crop_ap: bool = True
    crop_test_ap: bool = True
    pooling: str = "cortex"
    cortex_frac: float = 0.1


def ap_window(img: nib.Nifti1Image, extent_mm: float) -> tuple[int, int]:
    img = nib.Nifti1Image(img.dataobj, img.affine, img.header)
    img = nib.as_closest_canonical(nib.funcs.squeeze_image(img))
    data = img.get_fdata(dtype=np.float32)
    zoom = img.header.get_zooms()[1]

    live = np.flatnonzero((data > 0).sum(axis=(0, 2)))
    extent = round(extent_mm / zoom)
    return round((live[0] + live[-1] - extent) / 2), extent


def crop_ap(img: nib.Nifti1Image, window: tuple[int, int]) -> nib.Nifti1Image:
    img = nib.Nifti1Image(img.dataobj, img.affine, img.header)
    img = nib.as_closest_canonical(nib.funcs.squeeze_image(img))
    data = img.get_fdata(dtype=np.float32)

    start, extent = window
    lo, hi = max(start, 0), min(start + extent, data.shape[1])
    cropped = np.zeros((data.shape[0], extent, data.shape[2]), dtype=np.float32)
    cropped[:, lo - start : hi - start] = data[:, lo:hi]
    step = np.eye(4)
    step[1, 3] = start
    return nib.Nifti1Image(cropped, img.affine @ step)


# ---- method: the part we tune -----------------------------------------------------------


class Task5Method:
    """Frozen sMRI MAE, mean-pooled tokens over the t1w, logistic head."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, transform = load_backbone(cfg.ckpt_path)
        # using zero masking now that data are synthseg masked
        self.transform = SmriMaeTransform(
            img_size=transform.img_size, spacing=transform.spacing, masking=cfg.masking
        )
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.patch_size = tuple(patchify.patch_size)
        self.cache: dict[str, np.ndarray] = {}
        self.head = None

    @torch.inference_mode()
    def features(self, images: Images) -> np.ndarray:
        """(D,) per subject. A pure function of the images, so training and inference agree."""
        img, seg = images["t1w"], images["synthseg"]
        if self.cfg.crop_ap:
            window = ap_window(img, AP_EXTENT_MM)
            img = crop_ap(img, window)
            seg = crop_ap(seg, window)

        sample = self.transform(img)
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        if self.cfg.pooling == "cortex":
            cortex, _ = self.transform.resize(seg, mode="nearest-exact")
            cortex = cortex.to(self.device)
            gx, gy, gz = self.grid_size
            px, py, pz = self.patch_size
            is_cortex_grid = ((cortex == CORTEX[0]) | (cortex == CORTEX[1])).float()
            is_cortex_grid = (
                is_cortex_grid.reshape(gx, px, gy, py, gz, pz).mean(dim=(1, 3, 5)).reshape(-1)
            )
            is_cortex_grid = is_cortex_grid > self.cfg.cortex_frac
            is_cortex = is_cortex_grid[out["patch_ids"]]
            token_mask = (out["token_mask"].bool() & is_cortex).unsqueeze(-1)
        else:
            token_mask = out["token_mask"].bool().unsqueeze(-1)

        patch_embeds = out["patch_embeds"]
        embed = (patch_embeds * token_mask).sum(dim=1) / token_mask.sum(dim=1)
        return embed[0].float().cpu().numpy()

    def cached_features(self, row: dict) -> np.ndarray:
        if row["subject"] not in self.cache:
            self.cache[row["subject"]] = self.features(row)
        return self.cache[row["subject"]]

    def fit(self, rows: list[dict]) -> None:
        X = np.stack([self.cached_features(row) for row in rows])
        y = np.array([row["label"] for row in rows])

        clf = LogisticRegressionCV(
            Cs=10,
            class_weight="balanced",
            scoring="roc_auc",
            max_iter=1000,
            l1_ratios=(0,),
            use_legacy_attributes=False,
        )
        self.head = make_pipeline(StandardScaler(), clf)
        self.head.fit(X, y)
        self.positive = list(self.head.classes_).index(1)

    def predict(self, images: Images) -> float:
        """Positive-class probability. Indexes `classes_` rather than assuming column 1, which
        would silently score the wrong class if the label order differed."""
        X = self.features(images)[None]
        probs = self.head.predict_proba(X)[0]
        return float(probs[self.positive])

    def save(self, model_dir: Path) -> None:
        """Everything `load` needs but the backbone weights, which stay wherever `ckpt_path`
        points -- a few hundred KB, so a run saves one without copying a 3.7G checkpoint."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump({"head": self.head, "positive": self.positive}, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task5Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields, for what differs
        between here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = joblib.load(model_dir / "head.joblib")
        method.head, method.positive = state["head"], state["positive"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over the modalities whether or not a model uses them.
IMAGE_COLS = ("t1w", "synthseg")


def cross_validate(
    rows: list[dict],
    method: Task5Method,
    seed: int = 0,
    n_folds: int = 20,
    crop_test_ap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold score for every subject, each predicted by a head fit on the other folds."""
    y = np.array([row["label"] for row in rows])
    oof = np.zeros(len(rows), dtype=float)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    start = time.perf_counter()
    for fold, (train, test) in enumerate(folds.split(rows)):
        method.fit([rows[i] for i in train])
        for i in test:
            images = {key: rows[i][key] for key in IMAGE_COLS}
            if crop_test_ap:
                window = ap_window(images["t1w"], AP_EXTENT_MM)
                images = {key: crop_ap(img, window) for key, img in images.items()}
            oof[i] = method.predict(images)
        logger.info(
            f"fold {fold + 1}/{n_folds} n={len(test)} y={y[test]} "
            f"p={np.round(oof[test], 3)} ({time.perf_counter() - start:.0f}s)"
        )
    return y, oof


def score(
    y: np.ndarray, oof: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05
) -> dict:
    """AUROC, the challenge metric, plus a percentile CI resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[rows])) < 2:
            continue
        samples.append(roc_auc_score(y[rows], oof[rows]))

    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auroc": float(roc_auc_score(y, oof)),
        "auroc_ci_low": float(low),
        "auroc_ci_high": float(high),
    }


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task5

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    ds = load_fomo_task5()
    # skullstrip with synthseg, cached with hf (keeps the label map too, as a "synthseg" column)
    ds = synthseg.synthseg_strip_dataset(ds, source="t1w")
    rows = list(ds)
    logger.info(f"dataset: {len(rows)} subjects, {sum(r['label'] for r in rows)} positive")

    method = Task5Method(cfg)
    start = time.perf_counter()
    y, oof = cross_validate(rows, method, crop_test_ap=cfg.crop_test_ap)
    run_time = time.perf_counter() - start
    summary = score(y, oof)

    # the shipped head sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.save(run_dir / "model")

    preds = [
        {"subject": row["subject"], "label": int(label), "pred": float(pred)}
        for row, label, pred in zip(rows, y, oof)
    ]
    (run_dir / "preds.json").write_text("".join(json.dumps(pred) + "\n" for pred in preds))

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: a t1 path in, one probability written to `--output`.

    `/app/predict.py` in the container is a shim over this, so what scores the submission is the
    code cross-validation already ran, not something generated at build time.
    """
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task5Method.load(args.model_dir, **overrides)

    img = nib.load(args.t1)
    seg = synthseg.synthseg(img)
    img = synthseg.applymask(img, seg)
    probability = method.predict({"t1w": img, "synthseg": seg})

    args.output.write_text(f"{probability:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="cross-validate over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one probability")
    predict_parser.add_argument("--t1", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
