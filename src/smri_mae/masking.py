from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor


MaskingPolicy = Literal["batch_min", "per_sample_pad"]


def trim_patch_mask(
    patch_mask: Float[Tensor, "B N"],
    mask_ratio: float,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "B N"], Int[Tensor, "B L"]]:
    """
    Trim a batch of patch masks to the same number of patches.
    Kept patches are selected randomly (shuffle=True) or sequentially (shuffle=False).
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    B, N = patch_mask.shape
    device = patch_mask.device

    if shuffle:
        noise = torch.rand(B, N, generator=generator, device=device)
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)
        patch_mask = patch_mask.gather(1, shuffle_ids)

    min_count = patch_mask.sum(dim=1).min()
    num_keep = int((1 - mask_ratio) * min_count.item())
    patch_mask = patch_mask * (patch_mask.cumsum(dim=1) <= num_keep)

    if shuffle:
        patch_mask = patch_mask.gather(1, restore_ids)

    mask_ids = patch_mask.nonzero(as_tuple=False)[:, 1].reshape(B, -1)
    return patch_mask, mask_ids


def pad_patch_mask(
    patch_mask: Float[Tensor, "B N"],
    mask_ratio: float,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "B N"], Int[Tensor, "B L"], Tensor]:
    """
    Select each row's own mask-ratio count, then pad ids to the batch max length.

    Returns:
    - selected patch mask [B, N]
    - padded selected patch ids [B, Lmax]
    - token mask [B, Lmax], true for real ids and false for padding
    """

    B, N = patch_mask.shape
    device = patch_mask.device
    patch_mask = patch_mask.to(dtype=torch.bool)

    if shuffle:
        noise = torch.rand(B, N, generator=generator, device=device)
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)
        patch_mask = patch_mask.gather(1, shuffle_ids)

    valid_counts = patch_mask.sum(dim=1)
    num_keep = torch.floor(valid_counts.to(torch.float64) * (1.0 - mask_ratio)).to(torch.long)
    selected = patch_mask & (patch_mask.cumsum(dim=1) <= num_keep.unsqueeze(1))

    if shuffle:
        selected = selected.gather(1, restore_ids)

    padded_ids, token_mask = patch_ids_from_mask(selected)
    return selected, padded_ids, token_mask


def patch_ids_from_mask(patch_mask: Tensor) -> tuple[Int[Tensor, "B L"], Tensor]:
    """Return padded patch ids and token-validity mask for a variable-count mask."""
    patch_mask = patch_mask.to(dtype=torch.bool)
    B, N = patch_mask.shape
    device = patch_mask.device
    counts = patch_mask.sum(dim=1)
    max_count = int(counts.max().item())

    patch_ids = torch.zeros((B, max_count), dtype=torch.long, device=device)
    token_mask = torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)
    if max_count == 0:
        return patch_ids, token_mask

    batch_ids, selected_ids = patch_mask.nonzero(as_tuple=True)
    slot_ids = patch_mask.cumsum(dim=1)[batch_ids, selected_ids].to(torch.long) - 1
    patch_ids[batch_ids, slot_ids] = selected_ids
    return patch_ids, token_mask
