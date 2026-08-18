"""FOMO task 3: brain age regression, scored by Pearson r and MAE as the challenge scores it.

`Task3Method` is the part we tune -- features, head, hyperparameters. The protocol below it is
fixed so scores stay comparable across iterations: fit on all 494 supplied SALD subjects and
evaluate once on the fixed 128-subject augmented DLBS development set.

`train` runs that protocol and saves the SALD-fitted head; `predict` is the challenge contract, one
t1 path in and one age out. Both go through `Task3Method`, so evaluation exercises the path the
submission will run.
"""

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import ndimage
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune.backbone import load_backbone
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]


@dataclass
class Config:
    task: str = "task3"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task3"
    device: str = "cuda"
    seed: int = 4466
    augmentation: bool = False
    age_balance: bool = False


# ---- method: the part we tune -----------------------------------------------------------

FIT_WEIGHTS = {
    "clean": 0.25,
    "acquisition": 0.15,
    "lowres_extreme": 0.10,
    "geometry": 0.15,
    "intensity_artifact": 0.15,
    "motion_coverage": 0.10,
    "domain": 0.10,
}
AGE_EDGES = np.array([18, 30, 40, 50, 60, 70, 81])


def resample_acquisition(
    data: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    target_spacing: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a slice profile and resample while preserving voxel-centre world coordinates."""
    old_shape = np.asarray(data.shape)
    spacing = nib.affines.voxel_sizes(affine)
    if profile == "gaussian":
        added_fwhm = np.sqrt(np.maximum(target_spacing**2 - spacing**2, 0))
        data = ndimage.gaussian_filter(data, added_fwhm / (2.355 * spacing))
    else:
        assert profile == "boxcar"
        for axis, width in enumerate(np.maximum(1, np.rint(target_spacing / spacing)).astype(int)):
            data = ndimage.uniform_filter1d(data, width, axis=axis, mode="nearest")
    new_shape = np.maximum(1, np.rint(old_shape * spacing / target_spacing)).astype(int)
    data = (
        F.interpolate(
            torch.from_numpy(data)[None, None],
            size=tuple(new_shape),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )
    mask = (
        F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=tuple(new_shape),
            mode="nearest-exact",
        )
        .squeeze()
        .bool()
        .numpy()
    )
    scale = old_shape / new_shape
    step = np.diag([*scale, 1.0])
    step[:3, 3] = 0.5 * scale - 0.5
    return data, mask, affine @ step


def change_contrast(
    data: np.ndarray, mask: np.ndarray, rng: np.random.Generator, strength: float
) -> np.ndarray:
    brain = data[mask]
    low, high = np.percentile(brain, (1, 99))
    gamma = rng.uniform(1 - 0.35 * strength, 1 + 0.45 * strength)
    data = low + (high - low) * np.clip((data - low) / (high - low), 0, 1) ** gamma
    coefficients = rng.uniform(-0.25 * strength, 0.25 * strength, 3)
    coordinates = np.meshgrid(
        *[np.linspace(-1, 1, n, dtype=np.float32) for n in data.shape], indexing="ij"
    )
    data *= np.clip(
        1
        + sum(
            coefficient * coordinate for coefficient, coordinate in zip(coefficients, coordinates)
        ),
        0.5,
        1.5,
    )
    sigma = 0.025 * strength * (high - low)
    n1 = rng.normal(0, sigma, data.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, data.shape).astype(np.float32)
    return np.sqrt((np.maximum(data, 0) + n1) ** 2 + n2**2)


def change_pose(
    data: np.ndarray, mask: np.ndarray, rng: np.random.Generator, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.deg2rad(rng.uniform(-8 * strength, 8 * strength, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    scale = rng.uniform(1 - 0.10 * strength, 1 + 0.10 * strength)
    inverse = np.linalg.inv(scale * (rz @ ry @ rx))
    shift = rng.uniform(-8 * strength, 8 * strength, 3)
    centre = (np.asarray(data.shape) - 1) / 2
    offset = centre - inverse @ (centre + shift)
    data = ndimage.affine_transform(data, inverse, offset=offset, order=1)
    mask = ndimage.affine_transform(mask, inverse, offset=offset, order=0) > 0
    return data, mask


def augment_row(row: dict, seed: int) -> Generator[dict, None, None]:
    """One clean and six fixed-seed acquisition/domain views of a SALD subject."""
    image = row["t1w"]
    source_data = image.get_fdata(dtype=np.float32)
    source_mask = source_data > 0
    for variant_index, (variant, weight) in enumerate(FIT_WEIGHTS.items()):
        if variant == "clean":
            yield {
                **row,
                "subject": f"{row['subject']}__clean",
                "base_subject": row["subject"],
                "variant": variant,
                "fit_weight": weight,
                "age_bin": int(np.searchsorted(AGE_EDGES, row["age"], side="right") - 1),
            }
            continue

        rng = np.random.default_rng(
            np.random.SeedSequence([seed, sum(map(ord, row["subject"])), variant_index])
        )
        data, mask, affine = source_data.copy(), source_mask.copy(), image.affine.copy()

        if variant == "acquisition":
            data = change_contrast(data, mask, rng, 0.5)
            family = ("anisotropic", "isotropic", "reconstruction")[int(rng.integers(0, 3))]
            if family == "anisotropic":
                target = nib.affines.voxel_sizes(affine).copy()
                target[int(rng.integers(0, 3))] = rng.uniform(2, 5)
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            elif family == "isotropic":
                target = np.full(3, rng.uniform(1.5, 2.5))
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            else:
                fwhm = rng.uniform(1.5, 3)
                data = ndimage.gaussian_filter(
                    data, fwhm / (2.355 * nib.affines.voxel_sizes(affine))
                )
        elif variant == "lowres_extreme":
            data = change_contrast(data, mask, rng, 0.4)
            family = ("thick_slice", "dual_axis", "isotropic")[int(rng.integers(0, 3))]
            target = nib.affines.voxel_sizes(affine).copy()
            if family == "thick_slice":
                target[int(rng.integers(0, 3))] = rng.uniform(5, 9)
            elif family == "dual_axis":
                axes = rng.choice(3, 2, replace=False)
                target[axes[0]] = rng.uniform(4, 8)
                target[axes[1]] = rng.uniform(1.5, 2.5)
            else:
                target[:] = rng.uniform(2.5, 3.5)
            data, mask, affine = resample_acquisition(data, mask, affine, target, "boxcar")
        elif variant == "geometry":
            data, mask = change_pose(data, mask, rng, 2.0)
            data = change_contrast(data, mask, rng, 0.35)
        elif variant == "intensity_artifact":
            data = change_contrast(data, mask, rng, 1.5)
            fwhm = rng.uniform(1.5, 3.5)
            data = ndimage.gaussian_filter(data, fwhm / (2.355 * nib.affines.voxel_sizes(affine)))
        elif variant == "motion_coverage":
            data, mask = change_pose(data, mask, rng, 0.8)
            data = change_contrast(data, mask, rng, 0.7)
            axis = int(rng.integers(0, 3))
            shift = np.zeros(3)
            shift[axis] = rng.uniform(6, 16)
            mix = rng.uniform(0.15, 0.30)
            data = (1 - mix) * data + mix * ndimage.shift(data, shift, order=1)
            dropout_axis = int(rng.integers(0, 3))
            for _ in range(int(rng.integers(2, 6))):
                start = int(rng.integers(0, data.shape[dropout_axis] - 3))
                width = int(rng.integers(1, 4))
                index = [slice(None)] * 3
                index[dropout_axis] = slice(start, start + width)
                data[tuple(index)] *= rng.uniform(0.25, 0.75)
            crop_axis = int(rng.integers(0, 3))
            crop_mm = int(rng.integers(4, 13))
            crop_voxels = max(1, round(crop_mm / nib.affines.voxel_sizes(affine)[crop_axis]))
            index = [slice(None)] * 3
            index[crop_axis] = (
                slice(0, crop_voxels) if rng.random() < 0.5 else slice(-crop_voxels, None)
            )
            mask[tuple(index)] = False
            erosion = int(rng.integers(0, 3))
            if erosion:
                mask = ndimage.binary_erosion(mask, iterations=erosion)
        else:
            assert variant == "domain"
            data, mask = change_pose(data, mask, rng, 1.15)
            data = change_contrast(data, mask, rng, 1.0)
            axis = int(rng.integers(0, 3))
            shift = np.zeros(3)
            shift[axis] = rng.uniform(6, 12)
            mix = rng.uniform(0.12, 0.22)
            data = (1 - mix) * data + mix * ndimage.shift(data, shift, order=1)
            erosion = int(rng.integers(0, 3))
            if erosion:
                mask = ndimage.binary_erosion(mask, iterations=erosion)
            family = ("anisotropic", "isotropic", "reconstruction")[int(rng.integers(0, 3))]
            if family == "anisotropic":
                target = nib.affines.voxel_sizes(affine).copy()
                target[int(rng.integers(0, 3))] = rng.uniform(3, 6)
                data, mask, affine = resample_acquisition(data, mask, affine, target, "boxcar")
            elif family == "isotropic":
                target = np.full(3, rng.uniform(2, 3))
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            else:
                fwhm = rng.uniform(3, 5)
                data = ndimage.gaussian_filter(
                    data, fwhm / (2.355 * nib.affines.voxel_sizes(affine))
                )

        data = np.where(mask, np.maximum(data, 0), 0).astype(np.float32)
        yield {
            "subject": f"{row['subject']}__{variant}",
            "base_subject": row["subject"],
            "age": row["age"],
            "age_bin": int(np.searchsorted(AGE_EDGES, row["age"], side="right") - 1),
            "variant": variant,
            "fit_weight": weight,
            "t1w": nib.Nifti1Image(data, affine),
        }


AUGMENT_CACHE = Path("/data/smri-datasets/task3_sald_augmented")


def cached_augment_row(row: dict, seed: int) -> Generator[dict, None, None]:
    """`augment_row` backed by a disk cache. The views are fixed-seed, so a cached view is
    identical to a fresh one and the ~10s/subject CPU cost is paid once across all runs."""
    cache_dir = AUGMENT_CACHE / f"seed{seed}"
    paths = {
        variant: cache_dir / f"{row['subject']}__{variant}.npz"
        for variant in FIT_WEIGHTS
        if variant != "clean"
    }
    if not all(path.exists() for path in paths.values()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        for view in augment_row(row, seed):
            if view["variant"] == "clean":
                continue
            path = paths[view["variant"]]
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, data=np.asarray(view["t1w"].dataobj), affine=view["t1w"].affine)
            os.replace(tmp, path)
    age_bin = int(np.searchsorted(AGE_EDGES, row["age"], side="right") - 1)
    for variant, weight in FIT_WEIGHTS.items():
        if variant == "clean":
            t1w = row["t1w"]
        else:
            cached = np.load(paths[variant])
            t1w = nib.Nifti1Image(cached["data"], cached["affine"])
        yield {
            "subject": f"{row['subject']}__{variant}",
            "base_subject": row["subject"],
            "age": row["age"],
            "age_bin": age_bin,
            "variant": variant,
            "fit_weight": weight,
            "t1w": t1w,
        }


class Task3Method:
    """Frozen sMRI MAE, mean-pooled tokens over the t1w, ridge head."""

    def __init__(self, cfg: Config):
        assert not cfg.age_balance or cfg.augmentation
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}
        self.head = None

    @torch.inference_mode()
    def features(self, images: Images) -> np.ndarray:
        """(D,) per subject. A pure function of the images, so training and inference agree."""
        sample = self.transform(images["t1w"])
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        patch_embeds = out["patch_embeds"]
        token_mask = out["token_mask"].bool().unsqueeze(-1)
        embed = (patch_embeds * token_mask).sum(dim=1) / token_mask.sum(dim=1)
        return embed[0].float().cpu().numpy()

    def cached_features(self, row: dict) -> np.ndarray:
        """Per-view features, cached on disk per backbone checkpoint. Delete the cache dir if the
        transform or pooling changes -- the key only identifies the checkpoint and view."""
        if row["subject"] not in self.cache:
            tag = re.sub(r"[^A-Za-z0-9]+", "_", self.cfg.ckpt_path)
            cache_dir = Path(self.cfg.output_root) / "feature_cache"
            path = cache_dir / f"{tag}_{row['subject']}.npy"
            if path.exists():
                self.cache[row["subject"]] = np.load(path)
            else:
                self.cache[row["subject"]] = self.features(row)
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp.npy")
                np.save(tmp, self.cache[row["subject"]])
                os.replace(tmp, path)
        return self.cache[row["subject"]]

    def fit(self, rows: list[dict]) -> None:
        if self.cfg.augmentation:
            features, ages, variants, weights, bins = [], [], [], [], []
            for row in rows:
                for view in cached_augment_row(row, self.cfg.seed):
                    features.append(self.cached_features(view))
                    ages.append(view["age"])
                    variants.append(view["variant"])
                    weights.append(view["fit_weight"])
                    bins.append(view["age_bin"])
            X = np.stack(features)
            y = np.array(ages, dtype=float)
        else:
            X = np.stack([self.cached_features(row) for row in rows])
            y = np.array([row["age"] for row in rows], dtype=float)

        # RidgeCV picks alpha by efficient leave-one-out on clean SALD only; DLBS is never touched.
        clean = np.array(variants) == "clean" if self.cfg.augmentation else np.ones(len(rows), bool)
        clean_head = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 19)))
        clean_head.fit(X[clean], y[clean])
        if not self.cfg.augmentation:
            self.head = clean_head
            return

        weights = np.array(weights)
        if self.cfg.age_balance:
            bins = np.array(bins)
            subject_bins = np.array(
                [np.searchsorted(AGE_EDGES, row["age"], side="right") - 1 for row in rows]
            )
            counts = np.bincount(subject_bins)
            weights *= len(rows) / (len(np.unique(subject_bins)) * counts[bins])
        self.head = make_pipeline(StandardScaler(), Ridge(alpha=clean_head[-1].alpha_))
        self.head.fit(
            X,
            y,
            standardscaler__sample_weight=weights,
            ridge__sample_weight=weights,
        )

    def predict(self, images: Images) -> float:
        """Age in years."""
        X = self.features(images)[None]
        return float(self.head.predict(X)[0])

    def save(self, model_dir: Path) -> None:
        """Everything `load` needs but the backbone weights, which stay wherever `ckpt_path`
        points -- a few hundred KB, so a run saves one without copying a 3.7G checkpoint."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump(self.head, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task3Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields, for what differs
        between here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        method.head = joblib.load(model_dir / "head.joblib")
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over the modalities whether or not a model uses them.
IMAGE_COLS = ("t1w",)


def evaluate(
    sald: list[dict], dlbs: list[dict], method: Task3Method
) -> tuple[np.ndarray, np.ndarray]:
    """Fit only on all supplied SALD subjects, then predict every fixed DLBS subject once."""
    start = time.perf_counter()
    method.fit(sald)
    y = np.array([row["age"] for row in dlbs], dtype=float)
    predicted = np.zeros(len(dlbs), dtype=float)
    for index, row in enumerate(dlbs):
        predicted[index] = method.predict({key: row[key] for key in IMAGE_COLS})
        if (index + 1) % 16 == 0:
            logger.info(
                f"DLBS {index + 1}/{len(dlbs)} "
                f"mae={np.abs(y[: index + 1] - predicted[: index + 1]).mean():.2f} "
                f"({time.perf_counter() - start:.0f}s)"
            )
    return y, predicted


def metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "pearson_r": float(np.corrcoef(y, predicted)[0, 1]),
        "mae": float(np.abs(y - predicted).mean()),
    }


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task3, load_fomo_task3_dlbs

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    sald = list(load_fomo_task3())
    dlbs = list(load_fomo_task3_dlbs())
    logger.info(f"fit: {len(sald)} SALD subjects; evaluate: {len(dlbs)} DLBS subjects")

    method = Task3Method(cfg)
    start = time.perf_counter()
    y, predicted = evaluate(sald, dlbs, method)
    run_time = time.perf_counter() - start
    summary = metrics(y, predicted)

    # This is the same SALD-fitted head scored above; DLBS never influences its weights.
    method.save(run_dir / "model")

    preds = [
        {"subject": row["subject"], "age": float(age), "pred": float(pred)}
        for row, age, pred in zip(dlbs, y, predicted)
    ]
    (run_dir / "preds.json").write_text("".join(json.dumps(pred) + "\n" for pred in preds))

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: a t1 path in, one age written to `--output`.

    `/app/predict.py` in the container is a shim over this, so what scores the submission is the
    code the DLBS protocol already ran, not something generated at build time.
    """
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task3Method.load(args.model_dir, **overrides)

    age = method.predict({"t1w": nib.load(args.t1)})

    args.output.write_text(f"{age:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser(
        "train", help="fit on all SALD, evaluate on fixed DLBS, then save"
    )
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one age in years")
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
