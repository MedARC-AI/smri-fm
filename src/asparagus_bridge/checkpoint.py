"""Convert pretraining checkpoints into asparagus-compatible ones."""

from collections.abc import Callable
from pathlib import Path

import torch


def convert_smri_mae_checkpoint(src_path: str | Path, dst_path: str | Path) -> None:
    """Read a smri_mae checkpoint, write an asparagus-compatible one."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    state_dict = {f"model.{k}": v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    out = {"state_dict": state_dict, "epoch": ckpt.get("epoch")}
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_path)


# Model name -> converter. Register new entries when adding sibling implementations
# (e.g. "smri_dino": convert_smri_dino_checkpoint).
CONVERTERS: dict[str, Callable[[str | Path, str | Path], None]] = {
    "smri_mae": convert_smri_mae_checkpoint,
}


def convert_checkpoint(model_name: str, src_path: str | Path, dst_path: str | Path) -> None:
    """Dispatch to the registered converter for `model_name`."""
    if model_name not in CONVERTERS:
        raise ValueError(
            f"no converter registered for model '{model_name}'. known: {sorted(CONVERTERS)}"
        )
    CONVERTERS[model_name](src_path, dst_path)