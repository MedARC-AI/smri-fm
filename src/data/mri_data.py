import json
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def augment_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Apply train-time MRI augmentation to a normalized sample.

    Augmentation expects dense image masks and intentionally leaves metadata and
    cached raw samples untouched.
    """
    out = dict(sample)
    image_tensor = torch.from_numpy(sample["image"])
    image = image_tensor.float()
    mask = torch.from_numpy(sample["img_mask"]).float()
    if image.shape != mask.shape:
        raise ValueError(
            f"MRI augmentation expects image and img_mask with same shape, got "
            f"{tuple(image.shape)} and {tuple(mask.shape)}"
        )

    image, mask = _spatial_jitter(image, mask, config.get("pad_range", (4, 8)))
    mask = (mask > 0.5).float()

    blur_cfg = config.get("blur") or {}
    if torch.rand(()) < float(blur_cfg.get("p", 0.0)):
        sigma = _sample_range(blur_cfg.get("sigma", (0.3, 0.6)))
        image = _mask_normalized_gaussian_blur(image, mask, sigma=sigma)

    scale = _sample_range(config.get("scale", (1.0, 1.0)))
    shift = _sample_range(config.get("shift", (0.0, 0.0)))
    noise_std = _sample_range(config.get("noise_std", (0.0, 0.0)))
    image = image * scale + shift
    if noise_std > 0:
        image = image + torch.randn_like(image) * noise_std
    image = image * mask

    out["image"] = image.to(image_tensor.dtype).numpy()
    out["img_mask"] = mask.bool().numpy()
    return out


def prepare_model_masks(
    batch: dict[str, torch.Tensor],
    mask_fn,
):
    mask_ratio = float(mask_fn.mask_ratio)
    patch_mask = mask_fn.patch_mask_from_img_mask(batch["img_mask"])
    valid_tokens = patch_mask.sum(dim=1)
    num_visible = valid_tokens.min().clamp(
        max=int((1 - mask_ratio) * patch_mask.shape[1])
    )
    visible_mask = mask_fn(batch["img_mask"]).unsqueeze(1)
    return (
        visible_mask,
        mask_fn.patch_mask_to_volume(patch_mask).unsqueeze(1),
        {
            "mask_ratio": mask_ratio,
            "visible_tokens_mean": float(num_visible.item()),
            "valid_patch_tokens_mean": float(valid_tokens.float().mean().item()),
            "valid_patch_tokens_min": float(valid_tokens.min().item()),
        },
    )


def _sample_range(value: float | Sequence[float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != 2:
        raise ValueError(f"expected scalar or two-value range, got {value}")
    low, high = float(value[0]), float(value[1])
    if low == high:
        return low
    return float(torch.empty(()).uniform_(low, high).item())


def _spatial_jitter(
    image: torch.Tensor,
    mask: torch.Tensor,
    pad_range: int | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_min, pad_max = _to_range(pad_range)
    if pad_max <= 0:
        return image, mask
    pads = [int(torch.randint(pad_min, pad_max + 1, ()).item()) for _ in range(3)]
    padding = (pads[2], pads[2], pads[1], pads[1], pads[0], pads[0])
    image_pad = F.pad(image, padding)
    mask_pad = F.pad(mask, padding)

    _, d, h, w = image.shape
    starts = [int(torch.randint(0, 2 * pad + 1, ()).item()) if pad > 0 else 0 for pad in pads]
    ds, hs, ws = starts
    return (
        image_pad[:, ds : ds + d, hs : hs + h, ws : ws + w],
        mask_pad[:, ds : ds + d, hs : hs + h, ws : ws + w],
    )


def _to_range(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected scalar or two-value range, got {value}")
    low, high = int(value[0]), int(value[1])
    if low < 0 or high < low:
        raise ValueError(f"invalid range {value}")
    return low, high


def _mask_normalized_gaussian_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if sigma <= 0:
        return image
    kernel = _gaussian_kernel1d(sigma, device=image.device, dtype=image.dtype)
    image_b = image.unsqueeze(0)
    mask_b = mask.to(image.dtype).unsqueeze(0)
    blurred = _separable_conv3d(image_b * mask_b, kernel)
    norm = _separable_conv3d(mask_b, kernel).clamp_min(eps)
    return (blurred / norm).squeeze(0) * mask


def _gaussian_kernel1d(
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_conv3d(volume: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = volume.shape[1]
    pad = kernel.numel() // 2
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = kernel.numel()
        weight = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0, 0, 0]
        padding[axis] = pad
        volume = F.conv3d(volume, weight, padding=tuple(padding), groups=channels)
    return volume


def make_collatable(value: Any) -> Any:
    """Replace JSON null values in metadata dictionaries for PyTorch collation.

    Lists and tuples are stringified because variable-length sequences in
    metadata break `torch.utils.data.default_collate`.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return {key: make_collatable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value
