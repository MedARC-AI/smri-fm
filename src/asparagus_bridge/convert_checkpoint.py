"""Convert native smri-fm MAE checkpoints to Asparagus network checkpoints.

Native smri-fm pretraining checkpoints save the training model state under a
top-level ``model`` key. Those keys are named relative to the MAE module, for
example ``encoder.patch_embed.weight`` and ``decoder.head.weight``. Asparagus
fine-tuning does not load that object directly. It passes a ``network_weights``
dictionary into a Lightning module whose backbone lives below ``model`` in the
Lightning state dict.

For cls/reg fine-tuning with ``SmriMaeClsRegBackbone``, only the MAE encoder is
shared with the downstream model. The reconstruction decoder and other
pretraining-only tensors do not exist in the downstream backbone and should not
be loaded. This converter therefore keeps encoder keys, remaps them to
``model.encoder.*``, and writes the Asparagus-compatible ``network_weights``
format. The task-specific regression/classification head is intentionally left
randomly initialized.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def convert_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return Asparagus Lightning-module keys for compatible MAE encoder weights.

    The source may be a plain smri-fm model state dict (``encoder.*``), a
    distributed state dict (``module.encoder.*``), a compiled state dict
    (``_orig_mod.encoder.*``), or an already Lightning-prefixed state dict
    (``model.encoder.*``). The output always uses ``model.encoder.*`` keys
    because Asparagus loads weights into a Lightning module containing the
    backbone at ``self.model``.
    """

    converted: dict[str, Tensor] = {}
    for source_key, value in state_dict.items():
        key = _strip_runtime_prefixes(source_key)
        if key.startswith("model.encoder."):
            target_key = key
        elif key.startswith("encoder."):
            target_key = f"model.{key}"
        else:
            continue

        if target_key in converted:
            raise ValueError(
                f"Multiple source keys map to {target_key!r}; "
                "checkpoint contains ambiguous encoder weights."
            )
        converted[target_key] = value

    if not converted:
        raise ValueError(
            "No MAE encoder weights were converted. Expected keys like "
            "'encoder.patch_embed.weight' or 'model.encoder.patch_embed.weight'."
        )

    return converted


def convert_checkpoint_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Convert ``input_path`` and write an Asparagus-compatible checkpoint."""

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    network_weights = convert_state_dict(state_dict)

    output = {
        "network_weights": network_weights,
        "smri_fm_args": checkpoint.get("args") if isinstance(checkpoint, dict) else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dictionary checkpoint.")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format. Expected top-level 'model' or 'state_dict'.")

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint state is not a dictionary.")
    return state_dict


def _strip_runtime_prefixes(key: str) -> str:
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Native smri-fm checkpoint path.")
    parser.add_argument("--output", required=True, type=Path, help="Output Asparagus checkpoint path.")
    args = parser.parse_args(argv)

    output = convert_checkpoint_file(args.input, args.output)
    print(f"Wrote {args.output} with {len(output['network_weights'])} network weights.")


if __name__ == "__main__":
    main()
