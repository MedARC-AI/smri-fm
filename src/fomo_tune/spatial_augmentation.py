import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SpatialTransform:
    matrix: tuple[tuple[float, float, float], ...]
    translation_mm: tuple[float, float, float]


def augmentation_seed(seed: int, subject: str, view: int) -> int:
    subject_seed = zlib.crc32(subject.encode())
    return (seed + subject_seed + 1009 * view) % (2**32)


def sample_spatial_transform(
    seed: int,
    max_rotation_deg: float,
    max_translation_mm: float,
    scale_range: tuple[float, float],
) -> SpatialTransform:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-max_rotation_deg, max_rotation_deg, size=3)
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    scale = rng.uniform(*scale_range)
    translation = rng.uniform(-max_translation_mm, max_translation_mm, size=3)
    return SpatialTransform(
        tuple(tuple(float(value * scale) for value in row) for row in rotation),
        tuple(float(value) for value in translation),
    )


def spatial_grid(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    transform: SpatialTransform,
    restore: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = [torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2 for size in shape]
    voxels = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    voxel_to_world = torch.as_tensor(affine[:3, :3], device=device, dtype=dtype)
    world_to_voxel = torch.linalg.inv(voxel_to_world)
    matrix = torch.as_tensor(transform.matrix, device=device, dtype=dtype)
    translation = torch.as_tensor(transform.translation_mm, device=device, dtype=dtype)
    world = voxels @ voxel_to_world.T
    if restore:
        sampled_world = world @ matrix.T + translation
    else:
        sampled_world = (world - translation) @ torch.linalg.inv(matrix).T
    sampled_voxels = sampled_world @ world_to_voxel.T
    sampled_voxels += torch.as_tensor(
        [(size - 1) / 2 for size in shape], device=device, dtype=dtype
    )

    sizes = torch.as_tensor([size - 1 for size in shape], device=device, dtype=dtype)
    normalized = 2 * sampled_voxels / sizes - 1
    return normalized.flip(-1)[None]


def resample_spatial(volume: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    _, nx, ny, nz = volume.shape
    assert tuple(grid.shape) == (1, nx, ny, nz, 3)
    return F.grid_sample(
        volume[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0]
