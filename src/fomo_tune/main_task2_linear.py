"""FOMO task 2: meningioma segmentation, with a linear probe over sub-patch targets.

Method (tune):

1. The backbone's own transform: 1mm isotropic, whole head on a 208x240x208 canvas.
2. Extract patch features, `depth` blocks in. `depth=0` is the patch embedding, `depth=None`
   the full model post-norm.
3. A multi-output logistic head predicting each sub-cell's tumour fraction, so one token carries
   `prod(subcell)` predictions rather than one.

`subcell` is per axis because flair is acquired at ~0.86x0.86x6.5mm: splitting x and y buys
resolution the scan has, and splitting z would manufacture it.

Protocol (fixed):

- Leave one subject out over the 23.
- Dice at every threshold in a fixed grid, after the method's own postprocessing.
- Choose the single cut maximizing mean Dice over the out-of-fold subjects.
- Reported with a bootstrap CI over subjects, alongside the per-subject oracle cut.
- Every fold is saved with its probability volume, so it can be inspected without refitting.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import joblib
import nibabel as nib
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from omegaconf import OmegaConf
from scipy import ndimage

from fomo_tune.backbone import load_backbone
from fomo_tune.logistic import Logistic
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]


@dataclass
class Config:
    task: str = "task2_linear"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
    modality: str = "flair"
    output_root: str = "output/fomo_tune"
    name: str = "task2_linear"
    subcell: list[int] = field(default_factory=lambda: [4, 4, 1])
    target_sigma_mm: float = 0.0
    depth: int | None = 4
    alpha: float = 1e1
    largest_component: bool = True
    device: str = "cuda"
    seed: int = 4466


# ---- geometry -----------------------------------------------------------------------------


def repack(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Round-trip through nibabel: the HF Nifti wrapper's own reorientation is not trustworthy."""
    return nib.Nifti1Image(img.dataobj, img.affine, img.header)


def resample_nearest(
    volume: np.ndarray, source_affine: np.ndarray, target_affine: np.ndarray, target_shape: tuple
) -> np.ndarray:
    """`volume` read at every voxel of the target grid, nearest neighbour, zero outside it."""
    target_to_source = np.linalg.inv(source_affine) @ target_affine
    matrix = target_to_source[:3, :3]
    offset = target_to_source[:3, 3]
    return ndimage.affine_transform(
        volume, matrix, offset, output_shape=target_shape, order=0, mode="constant", cval=0.0
    )


# ---- method: the part we tune ---------------------------------------------------------------


class Patches(NamedTuple):
    """One subject's tokens, the grid they sit on, and each sub-cell's tumour fraction."""

    features: np.ndarray  # (n_patches, dim)
    patch_ids: np.ndarray  # (n_patches,) indices into the flattened patch grid
    grid_affine: np.ndarray  # (4, 4) voxel-to-world of the grid the tokens were read from
    targets: np.ndarray | None = None  # (n_patches, prod(subcell)), None at inference


class Task2LinearMethod:
    """Frozen FLAIR tokens, one linear head decoding each token to its sub-cells."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.modality = cfg.modality

        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.patch_size = tuple(patchify.patch_size)
        self.img_size = tuple(patchify.img_size)

        self.subcell = tuple(cfg.subcell)
        assert len(self.subcell) == 3, f"subcell {self.subcell} is not one count per axis"
        self.cell_size = tuple(size // count for size, count in zip(self.patch_size, self.subcell))
        assert all(size % count == 0 for size, count in zip(self.patch_size, self.subcell)), (
            f"subcell {self.subcell} does not divide patch {self.patch_size}"
        )
        self.n_cells = int(np.prod(self.subcell))

        self.cache: dict[str, Patches] = {}
        self.head = None
        self.threshold = None

    @torch.inference_mode()
    def embed(self, images: Images) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kept patches' features and grid indices, and the affine of the grid they live on."""
        start = time.perf_counter()
        sample = self.transform(images[self.modality])
        prepared = time.perf_counter()
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        encoder = self.backbone.encoder
        captured = []
        handle = None
        if self.cfg.depth is not None:
            handle = encoder.blocks[self.cfg.depth].register_forward_pre_hook(
                lambda module, args: captured.append(args[0])
            )
        try:
            with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.backbone(batch)
        finally:
            if handle is not None:
                handle.remove()

        # batch size 1 leaves no padded token slots, which the depth hook's flat slice relies on
        assert out["token_mask"].all(), "the token sequence is padded"
        if self.cfg.depth is None:
            features = out["patch_embeds"][0]
        else:
            # batch size 1, so the jagged pack is one flat sequence behind the prefix tokens
            features = captured[0][encoder.num_prefix_tokens :]

        logger.info(
            f"embed prepare={prepared - start:.1f}s forward={time.perf_counter() - prepared:.1f}s"
        )
        return (
            features.float().cpu().numpy(),
            out["patch_ids"][0].cpu().numpy(),
            sample["affine"].numpy(),
        )

    def cell_targets(self, seg: nib.Nifti1Image, grid_affine: np.ndarray) -> np.ndarray:
        """Every sub-cell's tumour fraction, over the whole grid, in flattened order."""
        seg = repack(seg)
        labels = np.asarray(seg.dataobj, dtype=np.float32).round()
        on_grid = resample_nearest(labels, seg.affine, grid_affine, self.img_size) > 0
        field_ = on_grid.astype(np.float32)

        if self.cfg.target_sigma_mm > 0:
            voxel_mm = np.linalg.norm(grid_affine[:3, :3], axis=0).mean()
            field_ = ndimage.gaussian_filter(field_, self.cfg.target_sigma_mm / voxel_mm)

        gx, gy, gz = self.grid_size
        sx, sy, sz = self.subcell
        cx, cy, cz = self.cell_size
        return reduce(
            field_,
            "(gx sx cx) (gy sy cy) (gz sz cz) -> (gx gy gz) (sx sy sz)",
            "mean",
            gx=gx,
            gy=gy,
            gz=gz,
            sx=sx,
            sy=sy,
            sz=sz,
            cx=cx,
            cy=cy,
            cz=cz,
        )

    def cached_patches(self, row: dict) -> Patches:
        """Cached: leave-one-out revisits every subject n times."""
        if row["subject"] not in self.cache:
            start = time.perf_counter()
            features, patch_ids, grid_affine = self.embed(row)
            embedded = time.perf_counter()
            targets = self.cell_targets(row["seg"], grid_affine)
            self.cache[row["subject"]] = Patches(
                features, patch_ids, grid_affine, targets[patch_ids]
            )
            logger.info(
                f"cache {row['subject']} embed={embedded - start:.1f}s "
                f"targets={time.perf_counter() - embedded:.1f}s "
                f"tokens={len(features)} positive_cells={int((targets[patch_ids] > 0).sum())}"
            )
        return self.cache[row["subject"]]

    def fit(self, rows: list[dict]) -> None:
        """One logistic head from a token to every sub-cell's tumour fraction, alpha shared."""
        subjects = [self.cached_patches(row) for row in rows]
        features = torch.from_numpy(np.concatenate([s.features for s in subjects]))
        targets = torch.from_numpy(np.concatenate([s.targets for s in subjects]))

        head = Logistic(alpha=self.cfg.alpha)
        head.fit(features.to(self.device), targets.to(self.device))
        # kept on the host: predict is one small matmul, and the saved model has to load on cpu
        self.head = head.to("cpu")

    def predict_proba(self, images: Images, key: str | None = None) -> nib.Nifti1Image:
        """Tumour probability per voxel on the input's own grid, constant within each sub-cell.

        Pass `key` to score a subject already in the cache; the tokens are the frozen backbone's
        and do not depend on the fold, so this is the same embedding the uncached path computes.

        The prediction is carried out to the input rather than the truth carried in: a cell is
        larger than a label voxel in plane, and scoring on the model's grid would credit a
        cell-sized prediction with matching a voxel-sized label.
        """
        patches = self.cache[key] if key is not None else Patches(*self.embed(images))

        predicted = self.head.predict(torch.from_numpy(patches.features)).numpy()
        scores = np.zeros((int(np.prod(self.grid_size)), self.n_cells), dtype=np.float32)
        scores[patches.patch_ids] = predicted

        gx, gy, gz = self.grid_size
        sx, sy, sz = self.subcell
        cx, cy, cz = self.cell_size
        on_grid = rearrange(
            scores,
            "(gx gy gz) (sx sy sz) -> gx sx gy sy gz sz",
            gx=gx,
            gy=gy,
            gz=gz,
            sx=sx,
            sy=sy,
            sz=sz,
        )
        on_grid = repeat(
            on_grid,
            "gx sx gy sy gz sz -> (gx sx cx) (gy sy cy) (gz sz cz)",
            cx=cx,
            cy=cy,
            cz=cz,
        )

        image = repack(images[self.modality])
        on_input = resample_nearest(
            np.ascontiguousarray(on_grid), patches.grid_affine, image.affine, image.shape
        )
        return nib.Nifti1Image(on_input, image.affine)

    def binarize(self, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """Probabilities to a mask. All postprocessing lives here, so the protocol can search it
        by calling this at every candidate threshold rather than knowing what it does."""
        mask = probabilities >= threshold
        if not self.cfg.largest_component or not mask.any():
            return mask
        blobs, _ = ndimage.label(mask)
        sizes = np.bincount(blobs.reshape(-1))
        sizes[0] = 0
        return blobs == sizes.argmax()

    def predict(self, images: Images) -> nib.Nifti1Image:
        """A binary mask on the input's own grid, which is what the challenge scores."""
        assert self.threshold is not None, "threshold is set by `train` or by `load`, not by `fit`"
        probabilities = self.predict_proba(images)
        mask = self.binarize(np.asarray(probabilities.dataobj), self.threshold)
        return nib.Nifti1Image(mask.astype(np.uint8), probabilities.affine)

    def save(self, model_dir: Path) -> None:
        """Config, head and threshold; the backbone weights stay wherever `ckpt_path` points."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump({"head": self.head, "threshold": self.threshold}, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task2LinearMethod":
        """Rebuild a fitted method from `save`. Overrides are Config fields: ckpt path, device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = joblib.load(model_dir / "head.joblib")
        method.head, method.threshold = state["head"], state["threshold"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over all the modalities whether or not a model uses them.
IMAGE_COLS = ("dwi_b1000", "flair")

# Scores estimate a sub-cell tumour fraction whose prevalence is ~2e-4, so the grid is geometric
# rather than linear. Task 4's range, dropped a decade for a label an order of magnitude rarer.
THRESHOLDS = np.logspace(-4, -0.3, 60)


class Curves(NamedTuple):
    """Everything the protocol reports is a read off these, so no fold is ever recomputed."""

    dice: np.ndarray  # (n_subjects, n_thresholds)
    predicted_voxels: np.ndarray  # (n_subjects, n_thresholds)
    true_voxels: np.ndarray  # (n_subjects,)


def subject_curves(
    method: Task2LinearMethod, probabilities: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One subject's Dice and predicted voxel count at every threshold in THRESHOLDS."""
    true_voxels = int(truth.sum())

    dice = np.zeros(len(THRESHOLDS))
    predicted = np.zeros(len(THRESHOLDS))
    for i, threshold in enumerate(THRESHOLDS):
        prediction = method.binarize(probabilities, threshold)
        predicted_voxels = int(prediction.sum())
        overlap = int(np.logical_and(prediction, truth).sum())
        denominator = predicted_voxels + true_voxels

        predicted[i] = predicted_voxels
        dice[i] = 2 * overlap / denominator if denominator else 1.0
    return dice, predicted


def leave_one_out(rows: list[dict], method: Task2LinearMethod, folds_dir: Path) -> Curves:
    """Every subject's threshold curves, predicted by a head fit on the other n-1.

    Per-fold heads and predicted probability volumes are saved to `folds_dir/<held-out subject>`.
    Nb the head stores the subject's oracle threshold.
    """
    # every subject embedded once up front, so a fold reads the cache on both sides of the split
    start = time.perf_counter()
    for row in rows:
        method.cached_patches(row)
    logger.info(f"cache: {len(rows)} subjects in {time.perf_counter() - start:.0f}s")

    dice, predicted, true = [], [], []
    start = time.perf_counter()
    for row in rows:
        fold_start = time.perf_counter()
        method.fit([r for r in rows if r["subject"] != row["subject"]])
        fitted = time.perf_counter()

        image = method.predict_proba({key: row[key] for key in IMAGE_COLS}, key=row["subject"])
        probabilities = np.asarray(image.dataobj)
        truth = np.asarray(repack(row["seg"]).dataobj).round() > 0
        assert probabilities.shape == truth.shape, "probabilities are not on the label grid"

        predicted_at = time.perf_counter()
        subject_dice, subject_predicted = subject_curves(method, probabilities, truth)
        scored = time.perf_counter()
        dice.append(subject_dice)
        predicted.append(subject_predicted)
        true.append(int(truth.sum()))

        best = subject_dice.argmax()
        fold_dir = folds_dir / row["subject"]
        method.threshold = float(THRESHOLDS[best])
        method.save(fold_dir)
        np.savez_compressed(
            fold_dir / "prediction.npz",
            probability=probabilities,
            truth_voxels=np.flatnonzero(truth),
            affine=image.affine,
            threshold=method.threshold,
        )

        logger.info(
            f"fold {len(dice)}/{len(rows)} {row['subject']} phases: fit={fitted - fold_start:.1f}s "
            f"predict={predicted_at - fitted:.1f}s curves={scored - predicted_at:.1f}s "
            f"save={time.perf_counter() - scored:.1f}s"
        )
        logger.info(
            f"fold {len(dice)}/{len(rows)} {row['subject']} best={subject_dice[best]:.3f} "
            f"at thr={THRESHOLDS[best]:.2e} alpha={method.head.alpha_:.0e} "
            f"proba_max={probabilities.max():.3g} vox={true[-1]} "
            f"({time.perf_counter() - start:.0f}s)"
        )
    return Curves(np.stack(dice), np.stack(predicted), np.array(true))


def score(curves: Curves, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Mean per-subject Dice at the best single threshold, plus a percentile CI over subjects.

    `dice_oracle` lets every subject cut where it likes, which bounds any thresholding rule.
    """
    best = int(curves.dice.mean(axis=0).argmax())
    dice = curves.dice[:, best]

    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(dice), size=(n_boot, len(dice)))
    samples = dice[resamples].mean(axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "dice": float(dice.mean()),
        "dice_ci_low": float(low),
        "dice_ci_high": float(high),
        "dice_oracle": float(curves.dice.max(axis=1).mean()),
        "threshold": float(THRESHOLDS[best]),
    }


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task2

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # decoded once: leave-one-out revisits every subject n times, and the niftis are small
    rows = list(load_fomo_task2())
    logger.info(f"dataset: {len(rows)} subjects")

    method = Task2LinearMethod(cfg)
    start = time.perf_counter()
    curves = leave_one_out(rows, method, run_dir / "folds")
    run_time = time.perf_counter() - start
    summary = score(curves)

    # the shipped model sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.threshold = summary["threshold"]
    method.save(run_dir / "model")

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    np.savez(
        run_dir / "curves.npz",
        subjects=[row["subject"] for row in rows],
        thresholds=THRESHOLDS,
        **curves._asdict(),
    )
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: modality paths in, a mask nifti written to `--output`."""
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task2LinearMethod.load(args.model_dir, **overrides)

    # every image the challenge hands over, as in `leave_one_out`; the method takes what it uses
    paths = {"dwi_b1000": args.dwi, "flair": args.flair}
    mask = method.predict({key: nib.load(path) for key, path in paths.items()})

    nib.save(mask, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="leave-one-out over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one mask nifti")
    for flag in ("--flair", "--dwi"):
        predict_parser.add_argument(flag, type=Path, required=True)
    # accepted and ignored: the 4th modality is t2s on some subjects and swi on others
    for flag in ("--t2s", "--swi"):
        predict_parser.add_argument(flag, type=Path)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
