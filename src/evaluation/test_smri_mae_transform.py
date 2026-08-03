"""Round-trip tests for SmriMaeTransform and its inverse.

The transform reorients, resamples, transposes and pads/crops. Each of those is
easy to invert wrongly in a way that still produces a plausible-looking volume,
so these tests follow a single marked voxel through and back rather than
checking shapes alone.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

from evaluation.models import smri_mae
from evaluation.models.smri_mae import (
    SmriMaeTransform,
    pad_widths,
    reverse_smri_mae_transform,
)

# Voxel axis 0 -> +y, axis 1 -> -x, axis 2 -> +z: a permutation and a flip, so
# as_closest_canonical has real work to do and a no-op inverse cannot pass.
OBLIQUE = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
MARKER = (2, 3, 5)


def _source(shape=(10, 12, 14), affine=OBLIQUE, spacing=None):
    """A volume with one voxel much brighter than the rest."""
    rng = np.random.default_rng(0)
    data = rng.uniform(1.0, 2.0, size=shape).astype(np.float32)
    data[MARKER] = 100.0
    affine = affine.copy()
    if spacing is not None:
        affine[:3, :3] = affine[:3, :3] @ np.diag(spacing)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(spacing or (1.0, 1.0, 1.0))
    return img


def _onehot_at_brightest(image: torch.Tensor) -> torch.Tensor:
    """A label map marking whichever transformed voxel holds the marker."""
    flat = int(torch.argmax(image.reshape(-1)))
    pred = torch.zeros_like(image, dtype=torch.uint8)
    pred.reshape(-1)[flat] = 1
    return pred


def test_marker_returns_to_its_source_voxel():
    # The whole point of the inverse: a prediction made on the transformed grid
    # has to land on the anatomy it was predicted from. Padding only here, and
    # unit spacing, so the round trip is exact.
    img = _source()
    out = SmriMaeTransform(img_size=(16, 16, 16), return_properties=True)(img)

    pred = _onehot_at_brightest(out["image"][0].float())
    rev = reverse_smri_mae_transform(pred, out["properties"], mode="nearest")
    got = np.asarray(rev.dataobj)

    assert got.shape == img.shape
    assert np.allclose(rev.affine, img.affine)
    assert got.sum() == 1
    assert tuple(int(i) for i in np.argwhere(got)[0]) == MARKER


def test_saving_against_the_source_affine_without_the_inverse_is_wrong():
    # Guards the bug this exists to fix. predict_fomo26.py saved the prediction
    # straight against imgs[0].affine, which is neither the right shape nor,
    # for a non-RAS scan, the right orientation.
    img = _source()
    out = SmriMaeTransform(img_size=(16, 16, 16), return_properties=True)(img)
    pred = _onehot_at_brightest(out["image"][0].float())

    assert tuple(pred.shape) != img.shape
    rev = reverse_smri_mae_transform(pred, out["properties"], mode="nearest")
    assert np.asarray(rev.dataobj).shape == img.shape


def test_cropped_field_of_view_comes_back_as_background():
    # Cropping is not invertible. The contract is that the geometry is still
    # correct and the trimmed region reads as background, not that data returns.
    img = _source(shape=(24, 24, 24))
    out = SmriMaeTransform(img_size=(8, 8, 8), return_properties=True)(img)
    assert min(pad_widths((24, 24, 24), (8, 8, 8))) < 0  # it really did crop

    pred = torch.ones(tuple(out["image"].shape[1:]), dtype=torch.uint8)
    got = np.asarray(reverse_smri_mae_transform(pred, out["properties"], mode="nearest").dataobj)

    assert got.shape == img.shape
    assert got.max() == 1
    assert got.sum() == 8**3  # exactly the retained window, the rest zero


def test_resampled_volume_returns_to_the_source_grid():
    # Non-unit spacing means the forward pass resamples, so the inverse has to
    # resample back to the pre-resample shape rather than assume they match.
    img = _source(shape=(10, 12, 14), spacing=(2.0, 2.0, 2.0))
    out = SmriMaeTransform(img_size=(48, 48, 48), return_properties=True)(img)

    props = out["properties"]
    assert props["rescaled_shape"] != props["canonical_shape"]

    # Reverse as logits and argmax afterwards — the order the docstring calls
    # for, and the order asparagus uses. Nearest would be wrong here: at 2x it
    # reads a subset of positions, so a one-hot lands half a voxel off.
    pred = _onehot_at_brightest(out["image"][0].float()).float()
    got = np.asarray(reverse_smri_mae_transform(pred, props, mode="trilinear").dataobj)

    assert got.shape == img.shape
    assert np.unravel_index(got.argmax(), got.shape) == MARKER


def test_logits_keep_their_channel_axis_and_land_last():
    # asparagus reverses logits and argmaxes afterwards, so the inverse has to
    # accept (C, Z, Y, X). nibabel wants the channel axis last.
    img = _source()
    out = SmriMaeTransform(img_size=(16, 16, 16), return_properties=True)(img)

    logits = torch.randn(3, *out["image"].shape[1:])
    rev = reverse_smri_mae_transform(logits, out["properties"], mode="trilinear")

    assert np.asarray(rev.dataobj).shape == (*img.shape, 3)


def test_rejects_a_prediction_of_the_wrong_rank():
    img = _source()
    out = SmriMaeTransform(img_size=(16, 16, 16), return_properties=True)(img)
    with pytest.raises(ValueError, match="3 spatial dims"):
        reverse_smri_mae_transform(torch.zeros(2, 3, 16, 16, 16), out["properties"])


@pytest.mark.parametrize("axis_order", [(2, 1, 0), (0, 1, 2), (1, 2, 0)])
def test_round_trip_survives_a_change_of_axis_order(monkeypatch, axis_order):
    # The transform's transpose is contested — #33 drops it to keep canonical
    # (X, Y, Z). The inverse reads the order out of `properties` rather than
    # assuming one, so whichever way that lands, this keeps working. (1, 2, 0)
    # is in here because a self-inverse permutation would hide an argsort bug.
    monkeypatch.setattr(smri_mae, "AXIS_ORDER", axis_order)
    img = _source()
    out = SmriMaeTransform(img_size=(16, 16, 16), return_properties=True)(img)

    assert tuple(out["properties"]["axis_order"]) == axis_order

    pred = _onehot_at_brightest(out["image"][0].float())
    got = np.asarray(reverse_smri_mae_transform(pred, out["properties"], mode="nearest").dataobj)

    assert got.shape == img.shape
    assert got.sum() == 1
    assert tuple(int(i) for i in np.argwhere(got)[0]) == MARKER


def test_properties_are_opt_in():
    # The eval dataloader collates whatever the transform returns, so the extra
    # key must not appear unless it was asked for.
    img = _source()
    assert "properties" not in SmriMaeTransform(img_size=(16, 16, 16))(img)
