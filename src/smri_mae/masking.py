import math
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


def _random_mask_subset(
    candidate_mask: Tensor,
    counts: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select a different exact number of true entries from each batch row."""
    B, N = candidate_mask.shape
    noise = torch.rand(B, N, generator=generator, device=candidate_mask.device)
    noise = noise.masked_fill(~candidate_mask, 2.0)
    shuffle_ids = torch.argsort(noise, dim=1)
    restore_ids = torch.argsort(shuffle_ids, dim=1)
    shuffled_mask = candidate_mask.gather(1, shuffle_ids)
    selected = shuffled_mask & (shuffled_mask.cumsum(dim=1) <= counts.unsqueeze(1))
    return selected.gather(1, restore_ids)


def block_patch_mask(
    valid_patch_mask: Tensor,
    grid_size: tuple[int, int, int],
    mask_ratio: float,
    block_size_min: int,
    block_size_max: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Select visible patches after masking unions of random 3D cubes.

    Cubes are centered on valid patches and shifted inward at grid boundaries so
    their sampled side length is preserved. The final cube is trimmed when needed
    so every batch item has the same visible count.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if block_size_min <= 0 or block_size_max <= 0:
        raise ValueError("block sizes must be positive")
    if block_size_min > block_size_max:
        raise ValueError(
            f"block_size_min ({block_size_min}) must not exceed "
            f"block_size_max ({block_size_max})"
        )
    if block_size_max > min(grid_size):
        raise ValueError(
            f"block_size_max ({block_size_max}) must fit patch grid {grid_size}"
        )

    B, N = valid_patch_mask.shape
    if N != math.prod(grid_size):
        raise ValueError(
            f"patch mask has {N} patches, expected {math.prod(grid_size)} for grid {grid_size}"
        )

    valid_patch_mask = valid_patch_mask.to(dtype=torch.bool)
    valid_counts = valid_patch_mask.sum(dim=1)
    num_keep = int((1 - mask_ratio) * valid_counts.min().item())
    if num_keep == 0:
        return torch.zeros_like(valid_patch_mask)

    target_hidden = valid_counts - num_keep
    if not target_hidden.any():
        return valid_patch_mask

    device = valid_patch_mask.device
    grid_axes = torch.meshgrid(
        *(torch.arange(size, device=device) for size in grid_size),
        indexing="ij",
    )
    patch_coords = torch.stack(grid_axes, dim=-1).reshape(N, 3)
    coord_t, coord_h, coord_w = patch_coords.unbind(dim=-1)

    hidden = torch.zeros_like(valid_patch_mask)
    complete = target_hidden == 0
    chunk_size = 4
    max_blocks = max(64, 4 * math.ceil(N / (block_size_max**3)))
    num_chunks = math.ceil(max_blocks / chunk_size)

    center_weights = valid_patch_mask.to(dtype=torch.float32)
    empty_rows = center_weights.sum(dim=1) == 0
    if empty_rows.any():
        center_weights[empty_rows, 0] = 1.0

    for _ in range(num_chunks):
        center_ids = torch.multinomial(
            center_weights,
            num_samples=chunk_size,
            replacement=True,
            generator=generator,
        )
        centers = patch_coords[center_ids]
        side_lengths = torch.randint(
            block_size_min,
            block_size_max + 1,
            (B, chunk_size),
            device=device,
            generator=generator,
        )
        starts = centers - side_lengths.unsqueeze(-1) // 2
        max_starts = torch.tensor(grid_size, device=device) - side_lengths.unsqueeze(-1)
        starts = starts.clamp(min=0)
        starts = torch.minimum(starts, max_starts)
        ends = starts + side_lengths.unsqueeze(-1)
        cube_mask = (
            (coord_t >= starts[:, :, 0].unsqueeze(-1))
            & (coord_t < ends[:, :, 0].unsqueeze(-1))
            & (coord_h >= starts[:, :, 1].unsqueeze(-1))
            & (coord_h < ends[:, :, 1].unsqueeze(-1))
            & (coord_w >= starts[:, :, 2].unsqueeze(-1))
            & (coord_w < ends[:, :, 2].unsqueeze(-1))
        )
        cube_mask &= valid_patch_mask[:, None, :]
        cube_mask &= ~complete[:, None, None]

        union_states = []
        running_hidden = hidden
        for block_idx in range(chunk_size):
            running_hidden = running_hidden | cube_mask[:, block_idx]
            union_states.append(running_hidden)
        union_states = torch.stack(union_states, dim=1)
        hidden_counts = union_states.sum(dim=-1)
        reaches_target = hidden_counts >= target_hidden.unsqueeze(1)
        reaches_in_chunk = reaches_target.any(dim=1) & ~complete

        first_reach = reaches_target.to(torch.int64).argmax(dim=1)
        batch_ids = torch.arange(B, device=device)
        previous_idx = (first_reach - 1).clamp(min=0)
        previous_hidden = union_states[batch_ids, previous_idx]
        previous_hidden = torch.where(
            (first_reach == 0).unsqueeze(1),
            hidden,
            previous_hidden,
        )
        final_candidates = cube_mask[batch_ids, first_reach] & ~previous_hidden
        needed = (target_hidden - previous_hidden.sum(dim=1)).clamp(min=0)
        final_additions = _random_mask_subset(
            final_candidates,
            needed,
            generator=generator,
        )
        reached_hidden = previous_hidden | final_additions

        hidden = torch.where(
            reaches_in_chunk.unsqueeze(1),
            reached_hidden,
            union_states[:, -1],
        )
        complete |= reaches_in_chunk
        if complete.all():
            break

    remaining = (target_hidden - hidden.sum(dim=1)).clamp(min=0)
    if remaining.any():
        fallback_candidates = valid_patch_mask & ~hidden
        hidden |= _random_mask_subset(fallback_candidates, remaining, generator=generator)

    return valid_patch_mask & ~hidden
