"""Sliding-window blending and segmentation-head tests.

These stub out `forward`, so they exercise the accumulation arithmetic and the
decoder shape contract on CPU without running nested-tensor attention (which
needs CUDA and head_dim % 8 == 0).
"""

import numpy as np
import pytest
import torch

from asparagus_bridge.models_smri_mae import (
    LinearPatchDecode,
    SmriMaeSegBackbone,
    gaussian_window,
)

PATCH = (64, 64, 64)


def _stub_model(num_classes=3, blending="uniform", const=2.5, head="patch_decode"):
    """Segmentation backbone whose forward returns a constant, on CPU.

    The encoder is constructed but never called, so nested-tensor attention
    never runs.
    """
    model = SmriMaeSegBackbone(
        input_channels=1,
        output_channels=num_classes,
        img_size=PATCH,
        patch_size=16,
        depth=1,
        embed_dim=64,
        num_heads=8,
        head=head,
        window_blending=blending,
    )
    model.forward = lambda x: torch.full(
        (x.shape[0], num_classes, *x.shape[2:]), const
    )
    return model


def _coverage_counts(shape, patch_size=PATCH, overlap=0.5):
    """How many windows cover each voxel, as BaseNet enumerates them."""
    from gardening_tools.modules.networks.utils import get_steps_for_sliding_window

    steps = get_steps_for_sliding_window(shape, patch_size, overlap)
    counts = np.zeros(shape, dtype=int)
    for xs in steps[0]:
        for ys in steps[1]:
            for zs in steps[2]:
                counts[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] += 1
    return counts


def test_overlap_counts_are_non_uniform():
    # The premise of the fix: window coverage varies a lot across one volume,
    # so an unnormalized sum carries a spatially varying scale factor. The
    # ragged values come from get_steps_for_sliding_window appending a final
    # step at shape - patch regardless of stride.
    counts = _coverage_counts((251, 214, 198))

    assert counts.min() == 1
    assert counts.max() == 27


def test_uniform_blending_recovers_constant_logits():
    # A model that predicts a constant everywhere must yield that same constant
    # after blending, at every voxel, regardless of how many windows overlap.
    model = _stub_model(blending="uniform", const=2.5)
    x = torch.randn(1, 1, 200, 200, 160)

    with torch.no_grad():
        out = model._sliding_window_predict3D(x, PATCH, overlap=0.5)

    assert out.shape == (1, 3, 200, 200, 160)
    assert torch.allclose(out, torch.full_like(out, 2.5), atol=1e-5)


def test_gaussian_blending_recovers_constant_logits():
    # Gaussian weighting reweights which windows dominate, but the normalizing
    # divide must still leave a constant field untouched.
    model = _stub_model(blending="gaussian", const=-1.25)
    x = torch.randn(1, 1, 200, 200, 160)

    with torch.no_grad():
        out = model._sliding_window_predict3D(x, PATCH, overlap=0.5)

    assert torch.allclose(out, torch.full_like(out, -1.25), atol=1e-5)


def test_unnormalized_accumulation_would_scale_by_coverage():
    # Guards the regression this fix addresses: BaseNet's inherited path returns
    # the constant multiplied by the per-voxel window count, which is exactly
    # the spatially varying factor that trilinear reverse_preprocessing then
    # smears across voxel neighbourhoods.
    from gardening_tools.modules.networks.BaseNet import BaseNet

    model = _stub_model(blending="uniform", const=1.0)
    x = torch.randn(1, 1, 200, 200, 160)

    with torch.no_grad():
        inherited = BaseNet._sliding_window_predict3D(model, x, PATCH, overlap=0.5)
        fixed = model._sliding_window_predict3D(x, PATCH, overlap=0.5)

    counts = torch.as_tensor(_coverage_counts((200, 200, 160)), dtype=torch.float32)
    assert torch.allclose(inherited[0, 0], counts, atol=1e-4)
    assert inherited.max() > 17.0  # 18x inflation at the worst voxels
    assert torch.allclose(fixed, torch.ones_like(fixed), atol=1e-5)


def test_gaussian_window_peaks_at_centre_and_stays_positive():
    window = gaussian_window((8, 8, 8))

    assert window.shape == (8, 8, 8)
    assert window.min() > 0  # a zero weight would divide by zero at edge voxels
    assert window.max() == pytest.approx(1.0)
    assert window[4, 4, 4] > window[0, 0, 0]


def test_rejects_unknown_blending_mode():
    model = _stub_model(blending="bilinear")
    with pytest.raises(ValueError, match="window_blending"):
        model._window_weight(PATCH, torch.device("cpu"), torch.float32)


def test_linear_head_maps_tokens_to_their_own_patch():
    # The probe contract: a voxel's logits depend only on the token covering it.
    # Perturbing one token must change exactly one patch and leave the rest
    # untouched, which is what makes this a readout rather than a decoder.
    decoder = LinearPatchDecode(patch_size=(4, 4, 4), embed_dim=8, out_channels=2)
    tokens = torch.zeros(1, 8, 3, 3, 3)

    with torch.no_grad():
        base = decoder(tokens)
        tokens[0, :, 1, 2, 0] = 5.0
        bumped = decoder(tokens)

    assert base.shape == (1, 2, 12, 12, 12)

    changed = (bumped - base).abs().sum(dim=1)[0] > 0
    expected = torch.zeros(12, 12, 12, dtype=torch.bool)
    expected[4:8, 8:12, 0:4] = True
    assert torch.equal(changed, expected)


def test_linear_head_backbone_preserves_geometry():
    model = _stub_model(head="linear")
    # forward is stubbed above, so drive the decoder directly
    decoder = model.decoder

    assert isinstance(decoder, LinearPatchDecode)
    with torch.no_grad():
        out = decoder(torch.randn(1, 64, 4, 4, 4))
    assert out.shape == (1, 3, 64, 64, 64)


def test_rejects_unknown_head():
    with pytest.raises(ValueError, match="head must be"):
        SmriMaeSegBackbone(
            input_channels=1,
            output_channels=2,
            img_size=PATCH,
            patch_size=16,
            depth=1,
            embed_dim=64,
            num_heads=8,
            head="unet",
        )
