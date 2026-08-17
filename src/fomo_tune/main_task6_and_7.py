"""FOMO tasks 6 and 7: linear probing and bias & fairness, both scored off one embedding.

`Task6And7Method` is the part we tune -- which pooling of the frozen encoder we ship. There is no
protocol section: the challenge withholds the labels and fits its own probes, so there is nothing
to cross-validate here. The embedding is the vector tasks 1, 3 and 5 call `features`, and their
out-of-fold scores are the evidence it carries signal.

`export` writes the run dir `build.py` packages; `predict` is the challenge contract, one nifti of
any modality in and one fixed-length float32 `.npy` out.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

from fomo_tune import poolings, posttransform
from fomo_tune.backbone import load_backbone
from fomo_tune.utils import git_sha, setup_logging

logger = logging.getLogger("fomo_tune")


@dataclass
class Config:
    task: str = "task6_and_7"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task6_and_7"
    device: str = "cuda"
    # The whole tunable surface, ranked by experiments/task7_pooling_bench.
    # Defaults reproduce the mean-pooled vector this shipped before, so an
    # export with no overrides is byte-identical to the previous submission.
    pooling: str = "mean"
    transform: str = "identity"
    # Pooled-embedding npz from cache_pooled.py. Required only when `transform`
    # is fitted (anything but identity/l2): the projection is estimated here,
    # once, and frozen into the container. It cannot be fitted at inference --
    # the challenge hands predict.py one image at a time.
    fit_cache: str = ""


# ---- method: the part we tune -----------------------------------------------------------


class Task6And7Method:
    """Frozen sMRI MAE, mean-pooled tokens over whichever modality arrives."""

    def __init__(self, cfg: Config):
        if cfg.pooling not in poolings.VARIANTS:
            raise ValueError(f"unknown pooling {cfg.pooling!r}; "
                             f"one of {sorted(poolings.VARIANTS)}")
        if cfg.transform not in posttransform.VARIANTS:
            raise ValueError(f"unknown transform {cfg.transform!r}; "
                             f"one of {sorted(posttransform.VARIANTS)}")
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.post = None
        self.post_in_dim = None

    @torch.inference_mode()
    def predict(self, image: nib.Nifti1Image) -> np.ndarray:
        """(D,) float32. Pooling over the token axis, so D does not depend on the input grid."""
        sample = self.transform(image)
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        tokens = out["patch_embeds"][0].float().cpu().numpy()
        mask = out["token_mask"][0].bool().cpu().numpy()
        embed = poolings.apply(self.cfg.pooling, tokens, mask)
        if self.post is not None:
            if self.post_in_dim is not None and embed.shape[0] != self.post_in_dim:
                # Only reachable if the transform was fitted on a different
                # pooling than the one configured. Left unchecked it surfaces as
                # a broadcast error inside the container, at submission time.
                raise ValueError(
                    f"pooling {self.cfg.pooling!r} gives {embed.shape[0]} dims but "
                    f"transform {self.cfg.transform!r} was fitted on "
                    f"{self.post_in_dim}; re-export with a fit_cache built from "
                    "the same pooling"
                )
            embed = self.post(embed[None])[0]
        return np.asarray(embed, dtype=np.float32)

    def fit_post(self, pooled: np.ndarray) -> None:
        """Freeze the post-transform from pooled embeddings of allowed data.

        Fitted once, offline, and shipped as a matrix. `identity` and `l2` need
        no fit; everything else does, and a container built without one would
        silently ship the raw pooling instead of the variant that was chosen.
        """
        self.post = posttransform.fit(self.cfg.transform, pooled)
        self.post_in_dim = int(pooled.shape[1])
        probe = self.post(pooled[:1])
        logger.info(f"post-transform {self.cfg.transform}: "
                    f"{pooled.shape[1]} -> {probe.shape[1]} dims")

    def save(self, model_dir: Path) -> None:
        """Config, plus the fitted post-transform when there is one. The backbone
        weights stay wherever `ckpt_path` points; `build.py` copies them in."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        if self.post is not None:
            posttransform.save_state(self.post, model_dir / "post.npz")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task6And7Method":
        """Rebuild the method from `save`. Overrides are Config fields, for what differs between
        here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        post_path = model_dir / "post.npz"
        if post_path.exists():
            method.post = posttransform.load_state(post_path)
            state = method.post.state
            if state.get("kind") == "pca":
                method.post_in_dim = int(state["V"].shape[1])
        elif cfg.transform not in ("identity", "l2"):
            # Loading a fitted variant without its matrix would silently ship the
            # raw pooling under the winning variant's name.
            raise FileNotFoundError(
                f"config asks for transform={cfg.transform!r} but {post_path} is "
                "missing; re-run export with fit_cache=<pooled.npz>"
            )
        elif cfg.transform == "l2":
            method.post = posttransform.fit("l2", np.zeros((1, 1), dtype=np.float32))
        return method


# ---- entrypoints ------------------------------------------------------------------------


def export(args: argparse.Namespace) -> None:
    """The run dir the other tasks get from `train`, without the fitting there is nothing to do."""
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    method = Task6And7Method(cfg)

    if cfg.transform not in ("identity", "l2"):
        if not cfg.fit_cache:
            raise SystemExit(
                f"transform={cfg.transform!r} has to be fitted; pass "
                "fit_cache=<pooled.npz> from fomo_tune.cache_pooled"
            )
        blob = np.load(cfg.fit_cache, allow_pickle=False)
        if cfg.pooling not in blob.files:
            raise SystemExit(f"{cfg.fit_cache} has no '{cfg.pooling}' pooling; "
                             f"it holds {[f for f in blob.files if f not in ('age', 'subject')]}")
        method.fit_post(blob[cfg.pooling])
    elif cfg.transform == "l2":
        method.post = posttransform.fit("l2", np.zeros((1, 1), dtype=np.float32))

    method.save(run_dir / "model")
    logger.info(f"pooling {cfg.pooling}  transform {cfg.transform}")
    logger.info(f"backbone width {method.backbone.encoder.patch_embed.out_features}")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: one nifti path in, one embedding written to `--output`.

    `/app/predict.py` in the container is a shim over this, so what the challenge probes is the
    same vector tasks 1, 3 and 5 score out of fold.
    """
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task6And7Method.load(args.model_dir, **overrides)

    embedding = method.predict(nib.load(args.input))

    np.save(args.output, embedding)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    export_parser = modes.add_parser("export", help="write the run dir a container is built from")
    export_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    export_parser.set_defaults(run=export)

    predict_parser = modes.add_parser("predict", help="one image, one embedding")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
