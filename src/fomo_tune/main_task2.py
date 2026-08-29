"""FOMO task 2: meningioma segmentation, scored by per-subject Dice as the challenge scores it.

**`Task2Method` is what we tune.** Today: a frozen sMRI MAE over flair, one token per 8mm patch,
and nine progressive convolutional heads restoring the 1mm grid. The heads form a CAPI-style
hyperparameter grid trained on the same crops; `predict_proba` returns their uniform average as a
volume on the input's own grid, and the method does not decide where to cut it.

**The protocol is held fixed**, or scores stop being comparable across iterations: leave one
subject out, Dice at every threshold in a fixed grid, then the single cut maximizing mean Dice
over the out-of-fold subjects. That cut is tuned on the subjects it is then scored on, so the
number is somewhat inflated -- as is anything else tuned by re-running and reading it.

`train` runs the protocol then fits and saves the heads; `predict` is the challenge contract. Both
go through `Task2Method.predict_proba`, so every fold exercises the path the submission runs.
"""

import argparse
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import ndimage

from fomo_tune.backbone import load_backbone
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]


@dataclass
class Config:
    task: str = "task2"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
    modality: str = "flair"
    output_root: str = "output/fomo_tune"
    name: str = "task2"
    largest_component: bool = True
    device: str = "cuda"
    seed: int = 4466


# CAPI trains a Cartesian grid of heads in one module and optimizer. Positive weight is the
# task-specific second axis here; the remaining training recipe is shared by every candidate.
RECIPES = tuple(itertools.product((1e-3, 2e-3, 4e-3), (10.0, 20.0, 40.0)))
CORE_TOKENS = 8
HALO_TOKENS = 2
BATCH_SIZE = 4
STEPS = 300
EMA_START = 100
EMA_DECAY = 0.99


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
    """One subject's dense token and label grids, sampling locations, and token statistics."""

    tokens: np.ndarray
    kept: np.ndarray
    seg: np.ndarray
    positive: np.ndarray
    brain_tokens: np.ndarray
    sum: np.ndarray
    square: np.ndarray
    count: int


class ProgressiveDecoder(nn.Module):
    """Three 2x upsampling stages turn the final 8mm token grid into 1mm voxel logits."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(1024, 32, 1)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv3d(32, 16, 3, padding=1),
                nn.Conv3d(16, 8, 3, padding=1),
                nn.Conv3d(8, 4, 3, padding=1),
            ]
        )
        self.output = nn.Conv3d(4, 1, 3, padding=1)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.projection(tokens))
        for convolution in self.convolutions:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = F.gelu(convolution(x))
        return self.output(x)


class Task2Method:
    """Frozen FLAIR tokens and a uniformly averaged grid of progressive CNN decoders."""

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
        assert self.patch_size == (8, 8, 8)

        self.cache: dict[str, Patches] = {}
        self.head = None
        self.mean = None
        self.inverse_std = None
        self.threshold = None

    @torch.inference_mode()
    def embed(self, images: Images) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kept patches' features and grid indices, and the affine of the grid they live on."""
        sample = self.transform(images[self.modality])
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        keep = out["token_mask"][0].bool()
        features = out["patch_embeds"][0][keep].float().cpu().numpy()
        patch_ids = out["patch_ids"][0][keep].cpu().numpy()
        return features, patch_ids, sample["affine"].numpy()

    def cached_patches(self, row: dict) -> Patches:
        """Dense token/label grids cached because leave-one-out revisits every subject."""
        if row["subject"] not in self.cache:
            features, patch_ids, grid_affine = self.embed(row)
            dense = np.zeros((int(np.prod(self.grid_size)), 1024), dtype=np.float16)
            dense[patch_ids] = features.astype(np.float16)
            dense = np.moveaxis(dense.reshape(*self.grid_size, 1024), -1, 0)
            kept = np.zeros(int(np.prod(self.grid_size)), dtype=bool)
            kept[patch_ids] = True
            kept = kept.reshape(self.grid_size)[None]

            seg = repack(row["seg"])
            labels = np.asarray(seg.dataobj, dtype=np.float32).round()
            labels = resample_nearest(labels, seg.affine, grid_affine, self.img_size) > 0
            self.cache[row["subject"]] = Patches(
                np.pad(dense, ((0, 0),) + ((HALO_TOKENS, HALO_TOKENS),) * 3),
                np.pad(kept, ((0, 0),) + ((HALO_TOKENS, HALO_TOKENS),) * 3),
                labels,
                np.argwhere(labels),
                np.argwhere(kept[0]),
                features.sum(0, dtype=np.float64),
                np.square(features, dtype=np.float64).sum(0),
                len(features),
            )
        return self.cache[row["subject"]]

    def fit(self, rows: list[dict], seed: int) -> None:
        """Fit the entire decoder grid on one shared stream of balanced spatial crops."""
        subjects = [self.cached_patches(row) for row in rows]
        count = sum(subject.count for subject in subjects)
        total = sum((subject.sum for subject in subjects), start=np.zeros(1024))
        square = sum((subject.square for subject in subjects), start=np.zeros(1024))
        mean = total / count
        variance = square / count - np.square(mean)
        self.mean = torch.from_numpy(mean.astype(np.float32)).to(self.device)[
            None, :, None, None, None
        ]
        self.inverse_std = torch.from_numpy(
            np.maximum(variance, 1e-6).astype(np.float32) ** -0.5
        ).to(self.device)[None, :, None, None, None]

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.head = nn.ModuleList([ProgressiveDecoder() for _ in RECIPES]).to(self.device)
        optimizer = torch.optim.AdamW(
            [
                {"params": list(head.parameters()), "lr": learning_rate}
                for head, (learning_rate, _) in zip(self.head, RECIPES)
            ],
            lr=0.0,
            weight_decay=1e-4,
        )
        ema = None
        for step in range(STEPS):
            token_crops, kept_crops, targets = [], [], []
            for batch_index in range(BATCH_SIZE):
                subject = subjects[(step * BATCH_SIZE + batch_index) % len(subjects)]
                if batch_index < BATCH_SIZE // 2:
                    center = subject.positive[rng.integers(len(subject.positive))] // 8
                else:
                    center = subject.brain_tokens[rng.integers(len(subject.brain_tokens))]
                origin = np.clip(
                    center - rng.integers(1, CORE_TOKENS, size=3),
                    0,
                    np.asarray(self.grid_size) - CORE_TOKENS,
                )
                ox, oy, oz = (int(value) for value in origin)
                width = CORE_TOKENS + 2 * HALO_TOKENS
                token_crops.append(
                    subject.tokens[:, ox : ox + width, oy : oy + width, oz : oz + width]
                )
                kept_crops.append(
                    subject.kept[:, ox : ox + width, oy : oy + width, oz : oz + width]
                )
                x, y, z = ox * 8, oy * 8, oz * 8
                voxels = CORE_TOKENS * 8
                targets.append(subject.seg[None, x : x + voxels, y : y + voxels, z : z + voxels])

            tokens = torch.from_numpy(np.stack(token_crops)).to(self.device, dtype=torch.float32)
            kept = torch.from_numpy(np.stack(kept_crops)).to(self.device)
            target = torch.from_numpy(np.stack(targets)).to(self.device, dtype=torch.float32)
            inputs = (tokens - self.mean) * self.inverse_std * kept

            optimizer.zero_grad(set_to_none=True)
            for head, (_, positive_weight) in zip(self.head, RECIPES):
                with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
                    halo = HALO_TOKENS * 8
                    logits = head(inputs)[:, :, halo:-halo, halo:-halo, halo:-halo]
                    bce = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=torch.tensor(positive_weight, device=self.device)
                    )
                    probability = torch.sigmoid(logits)
                    dims = (1, 2, 3, 4)
                    dice = (2 * (probability * target).sum(dims) + 1) / (
                        probability.sum(dims) + target.sum(dims) + 1
                    )
                    loss = bce + 1 - dice.mean()
                loss.backward()
            optimizer.step()

            if step == EMA_START:
                ema = [
                    [parameter.detach().clone() for parameter in head.parameters()]
                    for head in self.head
                ]
            elif step > EMA_START:
                for averages, head in zip(ema, self.head):
                    for average, parameter in zip(averages, head.parameters()):
                        average.lerp_(parameter.detach(), 1 - EMA_DECAY)

        with torch.no_grad():
            for averages, head in zip(ema, self.head):
                for parameter, average in zip(head.parameters(), averages):
                    parameter.copy_(average)
        self.head.eval()

    def predict_proba(self, images: Images) -> nib.Nifti1Image:
        """Uniformly averaged voxel probabilities on the input image's native grid."""
        sparse, patch_ids, grid_affine = self.embed(images)
        dense = np.zeros((int(np.prod(self.grid_size)), 1024), dtype=np.float32)
        dense[patch_ids] = sparse
        tokens = torch.from_numpy(
            np.moveaxis(dense.reshape(*self.grid_size, 1024), -1, 0)[None]
        ).to(self.device)
        kept = np.zeros(int(np.prod(self.grid_size)), dtype=bool)
        kept[patch_ids] = True
        kept = torch.from_numpy(kept.reshape(1, 1, *self.grid_size)).to(self.device)
        inputs = (tokens - self.mean) * self.inverse_std * kept

        probability = torch.zeros(self.img_size, device=self.device)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"),
        ):
            for head in self.head:
                probability += torch.sigmoid(head(inputs)[0, 0].float()) / len(self.head)

        image = repack(images[self.modality])
        on_input = resample_nearest(
            probability.cpu().numpy(), grid_affine, image.affine, image.shape
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
        torch.save(
            {
                "head": self.head.state_dict(),
                "mean": self.mean.cpu(),
                "inverse_std": self.inverse_std.cpu(),
                "threshold": self.threshold,
            },
            model_dir / "head.pt",
        )

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task2Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields: ckpt path, device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = torch.load(model_dir / "head.pt", map_location=method.device, weights_only=True)
        method.head = nn.ModuleList([ProgressiveDecoder() for _ in RECIPES]).to(method.device)
        method.head.load_state_dict(state["head"])
        method.head.eval()
        method.mean = state["mean"].to(method.device)
        method.inverse_std = state["inverse_std"].to(method.device)
        method.threshold = state["threshold"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over all the modalities whether or not a model uses them.
IMAGE_COLS = ("dwi_b1000", "flair")

# Keep the baseline protocol's geometric threshold grid.
THRESHOLDS = np.logspace(-6, -0.3, 60)


class Curves(NamedTuple):
    """Everything the protocol reports is a read off these, so no fold is ever recomputed."""

    dice: np.ndarray  # (n_subjects, n_thresholds)
    predicted_voxels: np.ndarray  # (n_subjects, n_thresholds)
    true_voxels: np.ndarray  # (n_subjects,)


def subject_curves(
    method: Task2Method, probabilities: np.ndarray, truth: np.ndarray
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


def leave_one_out(rows: list[dict], method: Task2Method, folds_dir: Path) -> Curves:
    """Every subject's threshold curves, predicted by heads fit on the other n-1.

    Per-fold heads and predicted probability volumes are saved to `folds_dir/<held-out subject>`.
    Nb the head stores the subject's oracle threshold.
    """
    dice, predicted, true = [], [], []
    start = time.perf_counter()
    for held_out, row in enumerate(rows):
        method.fit(
            [r for r in rows if r["subject"] != row["subject"]],
            method.cfg.seed + held_out,
        )

        image = method.predict_proba({key: row[key] for key in IMAGE_COLS})
        probabilities = np.asarray(image.dataobj)
        truth = np.asarray(repack(row["seg"]).dataobj).round() > 0
        assert probabilities.shape == truth.shape, "probabilities are not on the label grid"

        subject_dice, subject_predicted = subject_curves(method, probabilities, truth)
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
            f"fold {len(dice)}/{len(rows)} {row['subject']} best={subject_dice[best]:.3f} "
            f"at thr={THRESHOLDS[best]:.2e} vox={true[-1]} "
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

    method = Task2Method(cfg)
    start = time.perf_counter()
    curves = leave_one_out(rows, method, run_dir / "folds")
    run_time = time.perf_counter() - start
    summary = score(curves)

    # the shipped model sees all n subjects, so it is not any of the models scored above
    method.fit(rows, cfg.seed + len(rows))
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
    method = Task2Method.load(args.model_dir, **overrides)

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
