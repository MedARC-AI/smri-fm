"""One forward pass per subject, every pooling applied inside it.

The only GPU step in the tasks 6/7 bench. The encoder is the whole cost, and
every candidate embedding is a different reduction of the same token output, so
running the backbone once and applying all of `poolings.VARIANTS` to that output
funds the entire grid for the price of a single pass.

Caching the raw tokens instead would be simpler and is not an option: ViT-L at
patch 8 over 208x240x208 gives ~20k tokens x 1024 dims, about 83MB per subject,
41GB across task 3's 494. The pooled cache is ~30MB and the bench that reads it
runs on a laptop.

    python -m fomo_tune.cache_pooled --out pooled.npz
    python -m fomo_tune.bench_task7 --cache pooled.npz

Writes an npz with one (n_subjects, D_pooling) array per pooling plus `age` and
`subject`, in a fixed subject order shared by every array.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from fomo_tune import poolings
from fomo_tune.backbone import load_backbone
from fomo_tune.datasets import load_fomo_task3
from fomo_tune.utils import git_sha, setup_logging

logger = logging.getLogger("fomo_tune")

DEFAULT_CKPT = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"


@torch.inference_mode()
def tokens_for(backbone, transform, image, device: torch.device):
    """(N, D) tokens and the (N,) keep-mask, matching what `main_task*.features`
    reduces. Kept on CPU as float32: the poolings are numpy and the whole point
    is that they are cheap next to the encoder."""
    sample = transform(image)
    batch = {key: value[None].to(device) for key, value in sample.items()}
    with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
        out = backbone(batch)
    tokens = out["patch_embeds"][0].float().cpu().numpy()
    mask = out["token_mask"][0].bool().cpu().numpy()
    return tokens, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N subjects only, for a smoke run")
    args = ap.parse_args()

    setup_logging(args.out.parent if args.out.parent.name else Path("."))
    logger.info(f"cache_pooled (git {git_sha()})")
    logger.info(f"ckpt {args.ckpt_path}")

    backbone, transform = load_backbone(args.ckpt_path)
    device = torch.device(args.device)
    backbone.to(device).eval().requires_grad_(False)

    dataset = load_fomo_task3()
    rows = list(dataset)
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"{len(rows)} subjects")

    names = list(poolings.VARIANTS)
    acc: dict[str, list] = {n: [] for n in names}
    subjects, ages = [], []

    for i, row in enumerate(rows):
        tok, mask = tokens_for(backbone, transform, row["t1w"], device)
        for name in names:
            # A pooling that fails on one subject would otherwise write a short
            # column and misalign every array in the npz. Fail loudly instead.
            acc[name].append(poolings.apply(name, tok, mask))
        subjects.append(row["subject"])
        ages.append(float(row["age"]))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            logger.info(f"  {i + 1}/{len(rows)}  tokens {tok.shape}")

    blob = {name: np.stack(acc[name]).astype(np.float32) for name in names}
    widths = {n: v.shape[1] for n, v in blob.items()}
    logger.info("pooled widths: " + ", ".join(f"{n}={d}" for n, d in widths.items()))

    blob["age"] = np.array(ages, dtype=np.float32)
    blob["subject"] = np.array(subjects)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **blob)
    logger.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
