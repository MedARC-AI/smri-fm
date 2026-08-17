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

Resumable. Progress is flushed every `--flush-every` subjects through a
temporary file and an atomic replace, and a re-run skips whatever is already
cached, so a preemption or a walltime kill costs minutes rather than the whole
pass. On a busy cluster the queue wait for another GPU can be a day, which is
longer than the pass itself.
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
    ap.add_argument("--flush-every", type=int, default=50,
                    help="save partial progress every N subjects, so a preempted "
                         "or timed-out job resumes instead of starting over")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        blob = {n: np.stack(acc[n]).astype(np.float32) for n in names}
        blob["age"] = np.array(ages, dtype=np.float32)
        blob["subject"] = np.array(subjects)
        tmp = args.out.with_suffix(".part.npz")
        np.savez_compressed(tmp, **blob)
        tmp.replace(args.out)          # atomic: a kill mid-write cannot corrupt it

    # Resume. The GPU pass is the only expensive step and the queue wait to get
    # another one can be a day, so a preemption or a walltime kill must not cost
    # the whole run.
    done: set[str] = set()
    if args.out.exists():
        prev = np.load(args.out, allow_pickle=False)
        if "subject" in prev.files:
            for n in names:
                acc[n] = list(prev[n]) if n in prev.files else []
            subjects = [str(s) for s in prev["subject"]]
            ages = [float(a) for a in prev["age"]]
            done = set(subjects)
            if any(len(acc[n]) != len(subjects) for n in names):
                logger.warning("existing cache is inconsistent; starting over")
                acc = {n: [] for n in names}
                subjects, ages, done = [], [], set()
            else:
                logger.info(f"resuming: {len(done)} subjects already cached")

    todo = [r for r in rows if r["subject"] not in done]
    logger.info(f"{len(todo)} subjects to encode")

    for i, row in enumerate(todo):
        tok, mask = tokens_for(backbone, transform, row["t1w"], device)
        for name in names:
            # A pooling that fails on one subject would otherwise write a short
            # column and misalign every array in the npz. Fail loudly instead.
            acc[name].append(poolings.apply(name, tok, mask))
        subjects.append(row["subject"])
        ages.append(float(row["age"]))
        if (i + 1) % args.flush_every == 0:
            flush()
            logger.info(f"  {len(subjects)}/{len(rows)}  tokens {tok.shape}  (saved)")
        elif (i + 1) % 25 == 0:
            logger.info(f"  {len(subjects)}/{len(rows)}  tokens {tok.shape}")

    flush()
    widths = {n: np.asarray(acc[n][0]).shape[0] for n in names}
    logger.info("pooled widths: " + ", ".join(f"{n}={d}" for n, d in widths.items()))
    logger.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), "
                f"{len(subjects)} subjects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
