"""FOMO task 4: trigeminal nerve and vessel segmentation, with a convolutional head.

Method (tune):

1. Scale the 0.5mm input to `1 / scale` mm, so scale 2 is native and above that magnifies, and
   crop around an anchor relative to the subject's mask centroid.
2. Extract patch features, `depth` blocks in. `depth=0` is the patch embedding, `depth=None`
   the full model post-norm.
3. Task 2's grid of progressive CNN decoders, climbing from the token grid back to the voxel grid
   in three 2x stages. Both labels come off one decoder as independent sigmoids.

The rationale is that the structures are ~2mm, so one prediction per 8mm patch will be
too coarse. Where `main_task4.py` buys the resolution with sub-patch ridge targets, here the
decoder carries the upsampling, and can shape a boundary the ridge could only step across.

Protocol (fixed):

- Leave one subject out over the 40.
- Per-label Dice at every pair of per-label cuts drawn from a fixed grid.
- Choose the global pair of cuts to maximize mean dice over out-of-fold subjects.
- Reported with a bootstrap CI over subjects, alongside the per-subject oracle cut.
- Every fold is saved, at its subject's oracle cut, so it can be inspected without refitting.
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

from fomo_tune.backbone import load_backbone, rescale
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
    name: str = "task4_conv"
    scale: int = 4
    depth: int | None = 4
    device: str = "cuda"
    seed: int = 4466


# CAPI trains a Cartesian grid of heads in one module and optimizer. Positive weight is the
# task-specific second axis here, lifted from task 2's 10-40 for a much rarer structure.
RECIPES = tuple(itertools.product((1e-3, 2e-3, 4e-3), (20.0, 60.0, 180.0)))
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

    tokens: np.ndarray  # (dim, gx + 2h, gy + 2h, gz + 2h) float16
    kept: np.ndarray  # (1, gx + 2h, gy + 2h, gz + 2h) bool
    seg: np.ndarray  # (X, Y, Z) uint8 label map on the canvas
    grid_affine: np.ndarray  # (4, 4) voxel-to-world of the grid the tokens were read from
    positive: np.ndarray  # (n_labelled, 3) canvas voxel indices carrying either label
    brain_tokens: np.ndarray  # (n_kept, 3) grid indices the encoder kept
    sum: np.ndarray
    square: np.ndarray
    count: int


class ProgressiveDecoder(nn.Module):
    """Three 2x upsampling stages turn the token grid into per-voxel logits, one channel a label."""

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
        self.output = nn.Conv3d(4, len(LABELS), 3, padding=1)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.projection(tokens))
        for convolution in self.convolutions:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = F.gelu(convolution(x))
        return self.output(x)


class Task4ConvMethod:
    """Frozen MAE features over a magnified crop, and a uniformly averaged grid of CNN decoders."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, _ = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)

        patchify = self.backbone.encoder.patchify
        self.grid_size = tuple(patchify.grid_size)
        self.patch_size = tuple(patchify.patch_size)
        self.img_size = tuple(patchify.img_size)
        assert self.patch_size == (8, 8, 8), "the decoder's three 2x stages assume a patch of 8"

        self.cache: dict[str, Patches] = {}
        self.head = None
        self.mean = None
        self.inverse_std = None
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

    def scatter(self, features: np.ndarray, patch_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The token sequence back onto the dense patch grid, and which cells it filled."""
        dim = features.shape[1]
        dense = np.zeros((int(np.prod(self.grid_size)), dim), dtype=np.float32)
        dense[patch_ids] = features
        kept = np.zeros(int(np.prod(self.grid_size)), dtype=bool)
        kept[patch_ids] = True
        return (
            np.moveaxis(dense.reshape(*self.grid_size, dim), -1, 0),
            kept.reshape(self.grid_size)[None],
        )

    def cached_patches(self, row: dict) -> Patches:
        """Cached: leave-one-out revisits every subject n times."""
        if row["subject"] not in self.cache:
            start = time.perf_counter()
            features, patch_ids, grid_affine = self.embed(row)
            embedded = time.perf_counter()
            tokens, kept = self.scatter(features, patch_ids)
            tokens = tokens.astype(np.float16)
            assert np.isfinite(tokens).all(), "features do not fit in float16"

            seg = repack(row["seg"])
            labels = np.asarray(seg.dataobj, dtype=np.float32).round()
            on_grid = resample_nearest(labels, seg.affine, grid_affine, self.img_size)
            on_grid = on_grid.astype(np.uint8)

            halo = ((0, 0),) + ((HALO_TOKENS, HALO_TOKENS),) * 3
            self.cache[row["subject"]] = Patches(
                np.pad(tokens, halo),
                np.pad(kept, halo),
                on_grid,
                grid_affine,
                np.argwhere(on_grid > 0),
                np.argwhere(kept[0]),
                features.sum(0, dtype=np.float64),
                np.square(features, dtype=np.float64).sum(0),
                len(features),
            )
            counts = " ".join(
                f"{name}={int((on_grid == value).sum())}"
                for name, value in zip(LABEL_NAMES, LABELS)
            )
            logger.info(
                f"cache {row['subject']} embed={embedded - start:.1f}s "
                f"targets={time.perf_counter() - embedded:.1f}s {counts}"
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
                seg = subject.seg[x : x + voxels, y : y + voxels, z : z + voxels]
                targets.append(np.stack([seg == value for value in LABELS]))

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
                    # per label, so the nerve is not pooled into the vessel's overlap
                    dims = (2, 3, 4)
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

    def predict_scores(self, images: Images, key: str | None = None) -> nib.Nifti1Image:
        """Per-label score per voxel on the input's own grid, averaged over the heads.

        Pass `key` to score a subject already in the cache; the tokens are the frozen backbone's
        and do not depend on the fold, so this is the same embedding the uncached path computes.
        """
        if key is not None:
            patches = self.cache[key]
            interior = slice(HALO_TOKENS, -HALO_TOKENS)
            dense = patches.tokens[:, interior, interior, interior].astype(np.float32)
            kept_grid = patches.kept[:, interior, interior, interior]
            grid_affine = patches.grid_affine
        else:
            features, patch_ids, grid_affine = self.embed(images)
            dense, kept_grid = self.scatter(features, patch_ids)

        tokens = torch.from_numpy(dense[None]).to(self.device)
        kept = torch.from_numpy(kept_grid[None]).to(self.device)
        inputs = (tokens - self.mean) * self.inverse_std * kept

        probability = torch.zeros((len(LABELS), *self.img_size), device=self.device)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"),
        ):
            for head in self.head:
                probability += torch.sigmoid(head(inputs)[0].float()) / len(self.head)

        image = repack(images[MODALITY])
        on_input = np.empty((*image.shape, len(LABELS)), dtype=np.float32)
        for label, volume in enumerate(probability.cpu().numpy()):
            on_input[..., label] = resample_nearest(
                np.ascontiguousarray(volume), grid_affine, image.affine, image.shape
            )
        return nib.Nifti1Image(on_input, image.affine)

    def binarize(self, scores: torch.Tensor, thresholds: np.ndarray) -> torch.Tensor:
        """Scores to a label map, one cut per label. All postprocessing lives here, so the protocol
        can search it by calling this at every candidate rather than knowing what it does.

        The two labels are independent sigmoids rather than a softmax, so each fires on its own cut
        and a voxel both claim goes to whichever is furthest above its own.
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
        """Config, heads and thresholds; the weights stay wherever `ckpt_path` points."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        torch.save(
            {
                "head": self.head.state_dict(),
                "mean": self.mean.cpu(),
                "inverse_std": self.inverse_std.cpu(),
                "thresholds": torch.as_tensor(self.thresholds, dtype=torch.float64),
            },
            model_dir / "head.pt",
        )

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task4ConvMethod":
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
        method.thresholds = state["thresholds"].cpu().numpy()
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Scores are sigmoid probabilities rather than label fractions, so the grid runs to ~0.95. It stays
# geometric, and the floor doubles as the candidate filter in `leave_one_out`.
THRESHOLDS = np.logspace(-3, -0.02, 60)


class Curves(NamedTuple):
    """Everything the protocol reports is a read off these, so no fold is ever recomputed.

    The two threshold axes are one cut per label, in `LABELS` order. A label's Dice depends on the
    other's cut as well as its own, since the two compete for the voxels they both claim.
    """

    dice: np.ndarray  # (n_subjects, n_labels, n_thresholds, n_thresholds)
    predicted_voxels: np.ndarray  # (n_subjects, n_labels, n_thresholds, n_thresholds)
    true_voxels: np.ndarray  # (n_subjects, n_labels)


def subject_curves(
    method: Task4ConvMethod, scores: np.ndarray, truth: np.ndarray
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


def leave_one_out(rows: list[dict], method: Task4ConvMethod, folds_dir: Path) -> Curves:
    """Every subject's threshold curves, predicted by heads fit on the other n-1.

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
    for held_out, row in enumerate(rows):
        fold_start = time.perf_counter()
        method.fit([r for r in rows if r["subject"] != row["subject"]], method.cfg.seed + held_out)
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
            f"at thr={cuts} oracle {peak} score_max={scores.max():.3g} "
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

    method = Task4ConvMethod(cfg)
    start = time.perf_counter()
    curves = leave_one_out(rows, method, run_dir / "folds")
    run_time = time.perf_counter() - start
    summary = score(curves)

    # the shipped model sees all n subjects, so it is not any of the models scored above
    method.fit(rows, cfg.seed + len(rows))
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
    method = Task4ConvMethod.load(args.model_dir, **overrides)

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
