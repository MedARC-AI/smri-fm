import json
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def unpack_img_mask_batch(mask: torch.Tensor, image_shape: Sequence[int]) -> torch.Tensor:
    """Return a dense boolean batch mask from dense or bit-packed mask tensors."""
    image_shape = tuple(int(dim) for dim in image_shape)
    if tuple(mask.shape[1:]) == image_shape:
        return mask.to(dtype=torch.bool)

    mask_numel = math.prod(image_shape)
    packed_numel = math.ceil(mask_numel / 8)
    if mask.ndim == 2 and mask.shape[1] == packed_numel:
        if mask.dtype != torch.uint8:
            raise ValueError(f"packed img_mask must have dtype uint8, got {mask.dtype}")
        shifts = torch.arange(7, -1, -1, device=mask.device, dtype=torch.uint8)
        bits = (mask.unsqueeze(-1).bitwise_right_shift(shifts) & 1).reshape(mask.shape[0], -1)
        return bits[:, :mask_numel].reshape((mask.shape[0], *image_shape)).bool()

    raise ValueError(
        f"expected batched img_mask shape (B, {', '.join(map(str, image_shape))}) "
        f"or (B, {packed_numel}), got {tuple(mask.shape)}"
    )


def augment_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Apply train-time MRI augmentation to a normalized sample.

    Augmentation expects dense image masks and intentionally leaves metadata and
    cached raw samples untouched.
    """
    out = dict(sample)
    image_dtype = sample["image"].dtype
    image = torch.tensor(sample["image"], dtype=torch.float32)
    mask = torch.tensor(sample["img_mask"], dtype=torch.float32)
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

    out["image"] = image.numpy().astype(image_dtype, copy=False)
    out["img_mask"] = mask.bool().numpy()
    return out


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
