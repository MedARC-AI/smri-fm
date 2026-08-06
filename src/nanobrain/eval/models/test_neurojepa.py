"""Neuro-JEPA needs the optional `neurojepa` extra; these skip without it.

The weights are gated, so only the preprocessing is checked here -- against the upstream
pipeline it mirrors, which is the part we reimplemented and can get quietly wrong.
"""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

pytest.importorskip("monai")
pytest.importorskip("neurojepa")

CPU = torch.device("cpu")

from nanobrain.eval.models.neurojepa import (  # noqa: E402
    IMG_SIZE,
    preprocess,
    static_transform,
)


def make_image(shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.random.default_rng(0).random(shape, dtype=np.float32) * 1000
    return nib.Nifti1Image(data, affine)


def upstream_volume(path: str):
    """The fork's own static pipeline, reading from disk as its callers do."""
    from neurojepa.data.transforms import loading_transforms, vit3d_transforms

    sample = loading_transforms(roi=list(IMG_SIZE), spacing=(1.0, 1.0, 1.0), model_name="vit")(
        {"image": path}
    )
    cfg = SimpleNamespace(data=SimpleNamespace(img_size=list(IMG_SIZE)))
    return vit3d_transforms(cfg, mode="test")(sample)["image"]


@pytest.mark.parametrize(
    "shape, affine",
    [
        ((60, 70, 65), np.diag([1.0, 1.2, 1.0, 1.0])),  # off-1mm on one axis: resample, cheaply
        ((120, 140, 130), np.diag([-1.0, -1.0, 1.0, 1.0])),  # L,P,S -> forces reorientation
    ],
)
def test_preprocess_matches_upstream(tmp_path, shape, affine):
    path = str(tmp_path / "image.nii.gz")
    nib.save(make_image(shape, affine), path)

    ours, world_affine = preprocess(static_transform(), nib.load(path), CPU)
    upstream = upstream_volume(path)
    assert ours.shape == (1, 1, *IMG_SIZE)
    assert torch.allclose(ours[0], upstream.as_tensor(), atol=1e-4)
    # the same affine places the tokens in world space, so pin it against upstream's too
    assert np.allclose(world_affine, np.asarray(upstream.affine), atol=1e-4)


def test_patch_coords_land_on_the_marked_token():
    """A bright marker must change the token whose world centre covers it.

    Checked on `patch_embed` (the projection layer) rather than the backbone: self-attention is
    global, so the most-changed *output* token is not the one holding the marker, and this test
    passes trivially against the backbone while saying nothing about position.
    """
    from einops import rearrange
    from nibabel.affines import apply_affine

    from nanobrain.eval.models import create_model
    from nanobrain.eval.models.neurojepa import preprocess
    from nanobrain.eval.nifti import canonical_img

    model = create_model("neurojepa")
    affine = np.diag([1.3, 1.1, 1.7, 1.0])
    affine[:3, 3] = [-70.0, -90.0, -60.0]
    base = np.random.default_rng(0).uniform(0.2, 0.4, (120, 140, 110)).astype(np.float32)
    base[20:100, 30:120, 25:95] += 2.0  # a foreground box, so CropForeground sees the same thing

    def project(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, world = preprocess(model.transform, nib.Nifti1Image(volume, affine), CPU)
        patch = np.array(model.backbone.patch_embed.patch_size)
        grid = np.array(batch.shape[2:]) // patch
        centres = rearrange(np.indices(tuple(grid)), "c x y z -> (x y z) c") * patch
        return model.backbone.patch_embed(batch)[0].numpy(), apply_affine(
            world, centres + (patch - 1) / 2
        )

    reference, coords = project(base)
    pitch = float(np.linalg.norm(coords[0] - coords[1]))
    for voxel in [(30, 40, 35), (85, 100, 80)]:
        marked = base.copy()
        marked[tuple(slice(v - 4, v + 4) for v in voxel)] += 8.0
        tokens, _ = project(marked)
        hit = int(np.linalg.norm(tokens - reference, axis=1).argmax())
        truth = apply_affine(canonical_img(nib.Nifti1Image(base, affine)).affine, np.array(voxel))
        assert np.linalg.norm(coords[hit] - truth) < 1.5 * pitch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_preprocess_on_gpu_matches_cpu(tmp_path):
    # The resample runs on GPU via cupy rather than scipy, so pin that it agrees with the CPU
    # chain the equivalence test above anchors.
    path = str(tmp_path / "image.nii.gz")
    nib.save(make_image((150, 180, 150), np.diag([1.33, 1.0, 1.0, 1.0])), path)
    img = nib.load(path)

    transform = static_transform()
    on_cpu, affine_cpu = preprocess(transform, img, torch.device("cpu"))
    on_gpu, affine_gpu = preprocess(transform, img, torch.device("cuda"))

    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-3)
    assert np.allclose(affine_cpu, affine_gpu, atol=1e-4)
