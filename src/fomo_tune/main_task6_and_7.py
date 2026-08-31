"""FOMO tasks 6 and 7: linear probing and bias & fairness."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

import fomo_tune.synthseg as synthseg
from fomo_tune.backbone import SmriMaeTransform, load_backbone
from fomo_tune.utils import git_sha, setup_logging

logger = logging.getLogger("fomo_tune")

NORMALIZATION_PATH = Path(__file__).parent / "assets/task6_and_7_walnut_normalization.npy"


@dataclass
class Config:
    task: str = "task6_and_7"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task6_and_7"
    device: str = "cuda"


# ---- method: the part we tune -----------------------------------------------------------


class Task6And7Method:
    """SynthSeg-masked Walnut, mean-pooled over valid final-layer patch tokens."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, transform = load_backbone(cfg.ckpt_path)
        self.transform = SmriMaeTransform(
            img_size=transform.img_size, spacing=transform.spacing, masking="zero"
        )
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.normalization = None

    def set_normalization(self, normalization: np.ndarray) -> None:
        assert normalization.shape == (1024, 2)
        self.normalization = normalization

    @torch.inference_mode()
    def predict(self, image: nib.Nifti1Image) -> np.ndarray:
        """(D,) float32. Pooling over the token axis, so D does not depend on the input grid."""
        segmentation = synthseg.synthseg(image, device=str(self.device))
        image = synthseg.applymask(image, segmentation)
        sample = self.transform(image)
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(batch)

        patch_embeds = out["patch_embeds"]
        token_mask = out["token_mask"].bool().unsqueeze(-1)
        embed = (patch_embeds * token_mask).sum(dim=1) / token_mask.sum(dim=1)
        embedding = embed[0].float().cpu().numpy()
        standardized = (embedding - self.normalization[:, 0]) / self.normalization[:, 1]
        return standardized.astype(np.float32, copy=False)

    def save(self, model_dir: Path) -> None:
        """The config and the normalization -- the backbone weights stay wherever `ckpt_path`
        points, and `build.py` is what copies them into a container."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        state = {"normalization": self.normalization}
        joblib.dump(state, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task6And7Method":
        """Rebuild the method from `save`. Overrides are Config fields, for what differs between
        here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        state = joblib.load(model_dir / "head.joblib")
        method.set_normalization(state["normalization"])
        return method


# ---- entrypoints ------------------------------------------------------------------------


def export(args: argparse.Namespace) -> None:
    """The run dir the other tasks get from `train`; the only state is the fixed normalization."""
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    method = Task6And7Method(cfg)
    method.set_normalization(np.load(NORMALIZATION_PATH))
    method.save(run_dir / "model")
    logger.info(f"embedding dim {method.backbone.encoder.patch_embed.out_features}")


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
