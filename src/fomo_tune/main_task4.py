"""FOMO task 4: trigeminal nerve and vessel segmentation.

Method (tune):

1. Scale the 0.5mm input to `1 / scale` mm, so scale 2 is native and above that magnifies, and
   crop around an anchor relative to the subject's mask centroid.
2. Extract patch features, `depth` blocks in. `depth=0` is the patch embedding, `depth=None`
   the full model post-norm.
3. Ridge regression head predicting "subcell" targets. At `subcell=8`, you get a
   separate prediction per voxel.

The rationale is that the structures are ~2mm, so one prediction per 8mm patch will be
too coarse. Too address this, we have two strategies: upscaling the model input, and
making sub-patch predictions.

Protocol (fixed):

- Leave one subject out over the 40.
- Per-label Dice at every pair of per-label cuts drawn from a fixed grid.
- Choose the global pair of cuts to maximize mean dice over out-of-fold subjects.
- Reported with a bootstrap CI over subjects, alongside the per-subject oracle cut.
- Every fold is saved, at its subject's oracle cut, so it can be inspected without refitting.
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

from fomo_tune.backbone import load_backbone, rescale
from fomo_tune.logistic import Logistic
from fomo_tune.ridge import Ridge
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]

MODALITY = "t2w"
LABELS = (1, 2)
LABEL_NAMES = ("nerve", "vessel")

# mm from the brain-mask centroid to the trigeminal region: an anatomical offset, so it scales with
# the voxel. Zeroing it centres the canvas on the brain.
ANCHOR_OFFSET_MM = (0.0, 2.0, -8.0)

# voxels from the canvas centre to where the pretrained frame puts that region: a position in the
# patch grid, so it does not scale. Zeroing it centres the region in the canvas.
CANVAS_CENTRE_VOXELS = (0.0, -9.4, -5.2)


@dataclass
class Config:
    task: str = "task4"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task4"
    scale: int = 4
    subcell: int = 4
    target_sigma_mm: float = 0.0
    head: str = "ridge"
    depth: int | None = 4
    alphas: list[float] = field(default_factory=lambda: [1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    alpha: float = 1e4
    n_splits: int = 5
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
    """One subject's tokens, the grid they sit on, and each sub-cell's label fractions."""

    features: np.ndarray  # (n_patches, dim)
    patch_ids: np.ndarray  # (n_patches,) indices into the flattened patch grid
    grid_affine: np.ndarray  # (4, 4) voxel-to-world of the grid the tokens were read from
    targets: np.ndarray | None = None  # (n_patches, n_labels * subcell ** 3), None at inference


class Task4Method:
    """Frozen sMRI MAE over a rescaled crop, one ridge decoding each token to its sub-cells."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, _ = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)

        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.patch_size = tuple(patchify.patch_size)
        self.img_size = tuple(patchify.img_size)

        self.subcell = cfg.subcell
        self.cell_size = tuple(size // cfg.subcell for size in self.patch_size)
        assert all(size % cfg.subcell == 0 for size in self.patch_size), (
            f"subcell {cfg.subcell} does not divide patch {self.patch_size}"
        )

        self.cache: dict[str, Patches] = {}
        self.head = None
        self.thresholds = None

    def prepare(self, image: nib.Nifti1Image) -> dict[str, torch.Tensor]:
        """One volume at `1 / scale` mm, cropped to the canvas around the subject's own anchor."""
        image = nib.as_closest_canonical(nib.funcs.squeeze_image(repack(image)))
        volume = torch.from_numpy(np.ascontiguousarray(image.get_fdata(dtype=np.float32)))
        assert volume.ndim == 3, f"expected a 3D volume, got {tuple(volume.shape)}"

        affine = np.asarray(image.affine)
        spacing = tuple(float(zoom) for zoom in image.header.get_zooms()[:3])
        target = 1.0 / self.cfg.scale
        if max(abs(size - target) for size in spacing) > 0.05:
            volume, affine = rescale(volume, affine, spacing, (target,) * 3)
        voxel_mm = np.linalg.norm(affine[:3, :3], axis=0)

        data = volume.numpy()
        head_mask = data > data.mean()
        brain = data[head_mask]
        mean, std = brain.mean(), max(brain.std(), 1e-6)

        centroid = np.array(ndimage.center_of_mass(head_mask))
        assert (np.abs(centroid / np.array(data.shape) - 0.5) < 1 / 6).all(), (
            f"brain-mask centroid {centroid.round(1)} is not near the middle of {data.shape}"
        )
        anchor = centroid + np.array(ANCHOR_OFFSET_MM) / voxel_mm
        start = np.round(
            anchor - np.array(self.img_size) / 2 - np.array(CANVAS_CENTRE_VOXELS)
        ).astype(int)

        source_lo = np.maximum(start, 0)
        source_hi = np.minimum(start + np.array(self.img_size), data.shape)
        assert (source_hi > source_lo).all(), "the canvas does not overlap the volume"
        source = tuple(slice(a, b) for a, b in zip(source_lo, source_hi))
        placed = tuple(slice(a, b) for a, b in zip(source_lo - start, source_hi - start))

        canvas = np.zeros(self.img_size, dtype=np.float32)
        mask = np.zeros(self.img_size, dtype=bool)
        mask[placed] = head_mask[source]
        canvas[placed] = np.where(head_mask[source], (data[source] - mean) / std, 0.0)

        # canvas voxel k holds resampled voxel k + start, a pure integer translation
        step = np.eye(4)
        step[:3, 3] = start
        return {
            "image": torch.from_numpy(canvas).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "affine": torch.as_tensor(affine @ step, dtype=torch.float32),
        }

    @torch.inference_mode()
    def embed(self, images: Images) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kept tokens' features and grid indices, and the affine of the grid they live on."""
        start = time.perf_counter()
        sample = self.prepare(images[MODALITY])
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
        """Each label's fraction of every sub-cell, over the whole grid, in flattened order."""
        seg = repack(seg)
        labels = np.asarray(seg.dataobj, dtype=np.float32).round()
        on_grid = resample_nearest(labels, seg.affine, grid_affine, self.img_size)

        voxel_mm = np.linalg.norm(grid_affine[:3, :3], axis=0).mean()
        gx, gy, gz = self.grid_size
        cx, cy, cz = self.cell_size

        fractions = []
        for value in LABELS:
            field_ = (on_grid == value).astype(np.float32)
            if self.cfg.target_sigma_mm > 0:
                field_ = ndimage.gaussian_filter(field_, self.cfg.target_sigma_mm / voxel_mm)
            fractions.append(
                reduce(
                    field_,
                    "(gx sx cx) (gy sy cy) (gz sz cz) -> (gx gy gz) (sx sy sz)",
                    "mean",
                    gx=gx,
                    gy=gy,
                    gz=gz,
                    sx=self.subcell,
                    sy=self.subcell,
                    sz=self.subcell,
                    cx=cx,
                    cy=cy,
                    cz=cz,
                )
            )
        return rearrange(np.stack(fractions), "label patch cell -> patch (label cell)")

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
                f"targets={time.perf_counter() - embedded:.1f}s"
            )
        return self.cache[row["subject"]]

    def fit(self, rows: list[dict]) -> None:
        """One ridge from a token to every sub-cell's label fraction, alpha shared over outputs.

        Alpha is chosen by held-out *subjects*: tokens within a subject are far from independent, so
        splitting on them would pick an alpha for a sample size we do not have.
        """
        subjects = [self.cached_patches(row) for row in rows]
        features = torch.from_numpy(np.concatenate([s.features for s in subjects]))
        targets = torch.from_numpy(np.concatenate([s.targets for s in subjects]))
        groups = np.concatenate([np.full(len(s.features), i) for i, s in enumerate(subjects)])

        if self.cfg.head == "ridge":
            head = Ridge(alphas=self.cfg.alphas, n_splits=self.cfg.n_splits, seed=self.cfg.seed)
            head.fit(features.to(self.device), targets.to(self.device), groups)
        else:
            head = Logistic(alpha=self.cfg.alpha)
            head.fit(features.to(self.device), targets.to(self.device))
        # kept on the host: predict is one small matmul, and the saved model has to load on cpu
        self.head = head.to("cpu")

    def predict_scores(self, images: Images, key: str | None = None) -> nib.Nifti1Image:
        """Per-label score per voxel on the input's own grid, constant within each sub-cell.

        Pass `key` to score a subject already in the cache; the tokens are the frozen backbone's
        and do not depend on the fold, so this is the same embedding the uncached path computes.

        The prediction is carried out to the input rather than the truth carried in: at `scale=1` a
        cell is larger than a label voxel, and scoring on the model's grid would credit a cell-sized
        prediction with matching a voxel-sized label.
        """
        patches = self.cache[key] if key is not None else Patches(*self.embed(images))

        predicted = self.head.predict(torch.from_numpy(patches.features)).numpy()
        scores = np.zeros(
            (int(np.prod(self.grid_size)), len(LABELS), self.subcell**3), dtype=np.float32
        )
        scores[patches.patch_ids] = rearrange(
            predicted, "patch (label cell) -> patch label cell", label=len(LABELS)
        )

        gx, gy, gz = self.grid_size
        cx, cy, cz = self.cell_size
        on_grid = rearrange(
            scores,
            "(gx gy gz) label (sx sy sz) -> label gx sx gy sy gz sz",
            gx=gx,
            gy=gy,
            gz=gz,
            sx=self.subcell,
            sy=self.subcell,
            sz=self.subcell,
        )
        on_grid = repeat(
            on_grid,
            "label gx sx gy sy gz sz -> label (gx sx cx) (gy sy cy) (gz sz cz)",
            cx=cx,
            cy=cy,
            cz=cz,
        )

        image = repack(images[MODALITY])
        on_input = np.empty((*image.shape, len(LABELS)), dtype=np.float32)
        for label, volume in enumerate(on_grid):
            on_input[..., label] = resample_nearest(
                np.ascontiguousarray(volume), patches.grid_affine, image.affine, image.shape
            )
        return nib.Nifti1Image(on_input, image.affine)

    def binarize(self, scores: torch.Tensor, thresholds: np.ndarray) -> torch.Tensor:
        """Scores to a label map, one cut per label. All postprocessing lives here, so the protocol
        can search it by calling this at every candidate rather than knowing what it does.

        The two labels' scores are not on a common scale, so each fires on its own cut and a voxel
        both claim goes to whichever is furthest above its own.
        """
        cuts = torch.as_tensor(thresholds, dtype=scores.dtype, device=scores.device)
        best, labels = (scores / cuts).max(dim=-1)
        return torch.where(best >= 1.0, labels + 1, 0).to(torch.uint8)

    def predict(self, images: Images) -> nib.Nifti1Image:
        """A label map on the input's own grid, which is what the challenge scores."""
        assert self.thresholds is not None, "thresholds are set by `train` or `load`, not `fit`"
        scores = self.predict_scores(images)
        labels = self.binarize(torch.from_numpy(np.asarray(scores.dataobj)), self.thresholds)
        return nib.Nifti1Image(labels.numpy(), scores.affine)

    def save(self, model_dir: Path) -> None:
        """Config, head and thresholds; the weights stay wherever `ckpt_path` points."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump({"head": self.head, "thresholds": self.thresholds}, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task4Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields: ckpt path, device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = joblib.load(model_dir / "head.joblib")
        method.head, method.thresholds = state["head"], state["thresholds"]
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Scores estimate a sub-cell label fraction whose prevalence is ~2e-3, so the grid is geometric
# rather than linear. The floor doubles as the candidate filter in `leave_one_out`.
THRESHOLDS = np.logspace(-3, -0.3, 60)


class Curves(NamedTuple):
    """Everything the protocol reports is a read off these, so no fold is ever recomputed.

    The two threshold axes are one cut per label, in `LABELS` order. A label's Dice depends on the
    other's cut as well as its own, since the two compete for the voxels they both claim.
    """

    dice: np.ndarray  # (n_subjects, n_labels, n_thresholds, n_thresholds)
    predicted_voxels: np.ndarray  # (n_subjects, n_labels, n_thresholds, n_thresholds)
    true_voxels: np.ndarray  # (n_subjects, n_labels)


def subject_curves(
    method: Task4Method, scores: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One subject's per-label Dice and predicted voxel count at every pair of cuts in THRESHOLDS."""
    scores = torch.as_tensor(scores, device=method.device)
    truth = torch.as_tensor(truth, device=method.device)
    hits = [truth == value for value in LABELS]
    true_voxels = torch.stack([hit.sum() for hit in hits])

    n_cuts = len(THRESHOLDS)
    shape = (len(LABELS), n_cuts, n_cuts)
    dice = torch.zeros(shape, device=method.device, dtype=torch.float64)
    predicted = torch.zeros(shape, device=method.device, dtype=torch.float64)
    for nerve, vessel in np.ndindex(n_cuts, n_cuts):
        prediction = method.binarize(scores, THRESHOLDS[[nerve, vessel]])
        for j, (value, hit) in enumerate(zip(LABELS, hits)):
            claimed = prediction == value
            predicted_voxels = claimed.sum()
            overlap = torch.logical_and(claimed, hit).sum()
            denominator = predicted_voxels + true_voxels[j]

            predicted[j, nerve, vessel] = predicted_voxels
            # one sync at the end instead of per pair, so the empty case stays on device
            dice[j, nerve, vessel] = torch.where(denominator > 0, 2 * overlap / denominator, 1.0)
    return dice.cpu().numpy(), predicted.cpu().numpy()


def leave_one_out(rows: list[dict], method: Task4Method, folds_dir: Path) -> Curves:
    """Every subject's threshold curves, predicted by a head fit on the other n-1.

    Each fold is saved under `folds_dir/<held-out subject>`, at that subject's oracle pair of cuts
    and with the sparse label map they give. The cuts are chosen on the subject's own labels, so
    they are for inspection and are not a score.
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

        scores_img = method.predict_scores({MODALITY: row[MODALITY]}, key=row["subject"])
        scores = np.asarray(scores_img.dataobj)
        truth = np.asarray(repack(row["seg"]).dataobj, dtype=np.float32).round().astype(np.uint8)
        assert scores.shape[:3] == truth.shape, "scores are not on the label grid"
        assert scores.shape[3] == len(LABELS) == 2, "expected two classes"

        # Voxels that can affect the dice score.
        # Nb, this is ~24x faster than scores.max(axis=-1), whose reduction axis is too short
        # for numpy to do well on.
        best_score = np.maximum(scores[..., 0], scores[..., 1])
        candidates = (best_score >= THRESHOLDS[0]) | (truth > 0)
        candidate_scores = scores[candidates]
        predicted_at = time.perf_counter()
        subject_dice, subject_predicted = subject_curves(
            method, candidate_scores, truth[candidates]
        )
        scored = time.perf_counter()
        dice.append(subject_dice)
        predicted.append(subject_predicted)
        true.append([int((truth == value).sum()) for value in LABELS])

        mean_dice = subject_dice.mean(axis=0)
        best = np.unravel_index(mean_dice.argmax(), mean_dice.shape)
        fold_dir = folds_dir / row["subject"]
        method.thresholds = THRESHOLDS[list(best)]
        method.save(fold_dir)

        # every claim is a candidate, since no cut is below the grid's lowest threshold
        claimed = np.zeros(truth.shape, dtype=np.uint8)
        claimed[candidates] = method.binarize(
            torch.from_numpy(candidate_scores), method.thresholds
        ).numpy()
        voxels = np.flatnonzero(claimed)
        np.savez_compressed(
            fold_dir / "prediction.npz",
            voxels=voxels,
            labels=claimed.reshape(-1)[voxels],
            shape=truth.shape,
            affine=scores_img.affine,
            thresholds=method.thresholds,
        )

        logger.info(
            f"fold {len(dice)}/{len(rows)} {row['subject']} phases: fit={fitted - fold_start:.1f}s "
            f"predict={predicted_at - fitted:.1f}s curves={scored - predicted_at:.1f}s "
            f"save={time.perf_counter() - scored:.1f}s"
        )

        peak = " ".join(f"{name}={subject_dice[j].max():.3f}" for j, name in enumerate(LABEL_NAMES))
        cuts = "/".join(f"{cut:.1e}" for cut in method.thresholds)
        logger.info(
            f"fold {len(dice)}/{len(rows)} {row['subject']} mean={mean_dice[best]:.3f} "
            f"at thr={cuts} oracle {peak} alpha={method.head.alpha_:.0e} "
            f"score_max={scores.max():.3g} "
            f"candidates={int(candidates.sum())} ({time.perf_counter() - start:.0f}s)"
        )
    return Curves(np.stack(dice), np.stack(predicted), np.array(true))


def score(curves: Curves, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Mean per-subject Dice at the best single pair of cuts, plus a percentile CI over subjects.

    `dice_oracle` lets every subject cut where it likes, which bounds any thresholding rule.
    """
    mean_dice = curves.dice.mean(axis=(0, 1))
    best = np.unravel_index(mean_dice.argmax(), mean_dice.shape)
    dice = curves.dice[:, :, best[0], best[1]]

    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(dice), size=(n_boot, len(dice)))
    samples = dice.mean(axis=1)[resamples].mean(axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "dice": float(dice.mean()),
        "dice_ci_low": float(low),
        "dice_ci_high": float(high),
        **{f"dice_{name}": float(dice[:, j].mean()) for j, name in enumerate(LABEL_NAMES)},
        "dice_oracle": float(curves.dice.mean(axis=1).max(axis=(1, 2)).mean()),
        "thresholds": [float(cut) for cut in THRESHOLDS[list(best)]],
    }


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_fomo_task4

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    rows = list(load_fomo_task4())
    logger.info(f"dataset: {len(rows)} subjects")

    method = Task4Method(cfg)
    start = time.perf_counter()
    curves = leave_one_out(rows, method, run_dir / "folds")
    run_time = time.perf_counter() - start
    summary = score(curves)

    # the shipped model sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.thresholds = np.array(summary["thresholds"])
    method.save(run_dir / "model")

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    np.savez(
        run_dir / "curves.npz",
        subjects=[row["subject"] for row in rows],
        thresholds=THRESHOLDS,
        **curves._asdict(),
    )
    cuts = "/".join(f"{cut:.2e}" for cut in summary["thresholds"])
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items() if k != "thresholds")
    logger.info(f"result: {scores}  thr={cuts}  ({run_time:.0f}s)")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: modality paths in, a label nifti written to `--output`."""
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task4Method.load(args.model_dir, **overrides)

    labels = method.predict({"t2w": nib.load(args.t2w)})
    nib.save(labels, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="leave-one-out over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one label nifti")
    predict_parser.add_argument("--t2w", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
