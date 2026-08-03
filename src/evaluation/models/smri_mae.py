from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evaluation.models.registry import register_model

import smri_mae.model_mae as models_mae


# Which source axes the transform hands to the model, applied as
# `data.permute(*AXIS_ORDER)`. `(2, 1, 0)` is the historical (X, Y, Z) -> (Z, Y, X)
# transpose; `(0, 1, 2)` keeps canonical NIfTI order. `SmriMaeTransform` records
# whichever is in force and `reverse_smri_mae_transform` inverts what was recorded,
# so changing this constant does not require touching the inverse.
AXIS_ORDER = (2, 1, 0)


class SmriMaeBackbone(nn.Module):
    def __init__(
        self,
        encoder: models_mae.MaskedEncoder,
        global_pool: Literal["cls", "reg", "patch"] = "patch",
        pad_to_multiple: int | None = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.global_pool = global_pool
        self.pad_to_multiple = pad_to_multiple

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        images = batch["image"]
        mask = batch["mask"]

        cls, reg, patch, _, _, token_mask = self.encoder(
            images,
            mask=mask,
            pad_to_multiple=self.pad_to_multiple,
        )

        if self.global_pool == "cls":
            embed = cls[:, 0, :]
        elif self.global_pool == "reg":
            embed = reg.mean(dim=1)
        elif self.global_pool == "patch":
            token_mask = token_mask.to(device=patch.device, dtype=torch.bool)
            denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=patch.dtype)
            embed = (patch * token_mask.unsqueeze(-1)).sum(dim=1) / denom
        return embed


class SmriMaeTransform:
    def __init__(
        self,
        img_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        return_properties: bool = False,
    ):
        self.img_size = img_size
        self.spacing = spacing
        # Off by default so the collated eval batches keep their existing keys.
        # Turn it on when the prediction has to travel back to source geometry,
        # and pass the result to `reverse_smri_mae_transform`.
        self.return_properties = return_properties

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        """
        TODO(mihir): check
        """
        source_affine = np.asarray(img.affine, dtype=np.float64)
        source_shape = tuple(img.shape[:3])

        # reorient to RAS
        img = nib.as_closest_canonical(img)

        # note, shape is (X, Y, Z) in contiguous F-order
        data = img.get_fdata(dtype=np.float32)
        # as_closest_canonical flips axes by slicing, which leaves negative
        # strides that torch.from_numpy refuses. Any scan not already stored
        # RAS-ish hits this, so the copy is required, not defensive.
        data = torch.from_numpy(np.ascontiguousarray(data))
        spacing = img.header.get_zooms()
        canonical_shape = tuple(data.shape)

        # resize
        if max(abs(s - s_) for s, s_ in zip(spacing, self.spacing)) > 0.05:
            data = rescale(data, spacing, target_spacing=self.spacing)
        rescaled_shape = tuple(data.shape)

        # permute the spatial axes, by default (X, Y, Z) F-order -> (Z, Y, X) C-order
        # TODO: this shape issue is a footgun. need to be consistent and obvious about
        # whether we are doing (X, Y, Z) or (Z, Y, X) for image as well as img_size,
        # spacing. AXIS_ORDER at least puts the choice in one place and records it.
        data = data.permute(*AXIS_ORDER).contiguous()
        widths = pad_widths(tuple(data.shape), self.img_size)
        data = F.pad(data, widths)

        # cheap mask
        # if we want a better mask, we have to compute it here.
        # model contract is nifti image -> embedding
        mask = data > data.mean()

        # z-score over brain-mask voxels (matches pretraining); background -> 0.
        # Raw intensities reach ~1e6, so this must happen before the fp16 cast.
        brain = data[mask]
        # population std (÷N, correction=0) to match the pretraining normalization.
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        # fp16 and add channel dim
        data = data.half().unsqueeze(0)
        mask = mask.unsqueeze(0)

        sample = {"image": data, "mask": mask}
        if self.return_properties:
            sample["properties"] = {
                "source_affine": source_affine,
                "source_shape": source_shape,
                "canonical_shape": canonical_shape,
                "rescaled_shape": rescaled_shape,
                "axis_order": AXIS_ORDER,
                "pad_widths": widths,
            }
        return sample


# can copy these utils to shared module if they prove generally useful


def rescale(
    x: torch.Tensor,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
):
    scales = tuple([current / target for current, target in zip(spacing, target_spacing)])
    x = F.interpolate(x[None, None], scale_factor=scales, mode="trilinear").squeeze(0, 1)
    return x


def pad_widths(shape: tuple[int, ...], target_shape: tuple[int, ...]) -> list[int]:
    """Centred `F.pad` widths taking `shape` to `target_shape`.

    In `F.pad` order, i.e. last axis first. Negative entries crop, which is how
    `pad_to_shape` handles a volume larger than the target.
    """
    padding = []
    for s, s_ in reversed(list(zip(shape, target_shape))):
        pad = s_ - s
        padding.extend([pad // 2, pad - pad // 2])
    return padding


def pad_to_shape(x: torch.Tensor, target_shape: tuple[int, ...]):
    # nb this also crops
    return F.pad(x, pad_widths(tuple(x.shape), target_shape))


def reverse_smri_mae_transform(
    pred: torch.Tensor,
    properties: dict,
    mode: Literal["trilinear", "nearest"] = "trilinear",
) -> nib.Nifti1Image:
    """Map a prediction on the transformed grid back onto the source image's grid.

    The inverse of `SmriMaeTransform`, which reorients to RAS, resamples to
    `spacing`, permutes the spatial axes by `AXIS_ORDER` and then pads-or-crops
    to `img_size`. Without this, a prediction saved against the source affine is
    silently misaligned: it is the wrong shape and, for any scan not already
    stored in RAS, the wrong orientation as well.

    Each of those four steps is undone from what `properties` recorded rather
    than from an assumption, so `AXIS_ORDER` can change without this function
    changing with it.

    `pred` is the model output on the transformed grid, spatial dims in
    `AXIS_ORDER`, with an optional leading channel dim for logits. `properties`
    is the dict produced by `SmriMaeTransform(..., return_properties=True)`.

    Reverse logits with `trilinear` and take the argmax afterwards, which is the
    order asparagus' `reverse_preprocessing` uses. A label map must use
    `nearest` — interpolating class indices invents classes that were never
    predicted.

    Cropping is not invertible. Anything `SmriMaeTransform` cut away comes back
    as background, so the returned volume is the source geometry with a hole
    wherever the field of view was trimmed.
    """
    if pred.ndim == 3:
        pred, squeeze = pred[None], True
    elif pred.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            "pred must be 3 spatial dims, optionally with a leading channel dim, "
            f"got shape {tuple(pred.shape)}"
        )

    # undo the pad/crop: same widths, negated
    pred = F.pad(pred, [-w for w in properties["pad_widths"]])

    # undo whichever axis order the forward pass recorded, rather than assuming
    # one. Inverting a permutation means asking, for each source axis, where it
    # ended up; the +1 skips the channel dim we prepended above.
    axis_order = tuple(properties["axis_order"])
    inverse_order = sorted(range(len(axis_order)), key=axis_order.__getitem__)
    pred = pred.permute(0, *(a + 1 for a in inverse_order))

    canonical_shape = tuple(properties["canonical_shape"])
    if tuple(pred.shape[1:]) != canonical_shape:
        # interpolate is float-only for these modes, so round-trip the dtype
        # rather than silently turning a uint8 label map into float32.
        dtype = pred.dtype
        pred = F.interpolate(pred[None].float(), size=canonical_shape, mode=mode)[0]
        pred = pred.to(dtype)

    # (C, X, Y, Z) -> (X, Y, Z, C), which is both nibabel's axis order and what
    # apply_orientation expects (it acts on the leading spatial axes).
    array = pred.permute(1, 2, 3, 0).cpu().numpy()

    # undo as_closest_canonical
    source_affine = np.asarray(properties["source_affine"], dtype=np.float64)
    back = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(("R", "A", "S")),
        nib.orientations.io_orientation(source_affine),
    )
    array = nib.orientations.apply_orientation(array, back)

    source_shape = tuple(properties["source_shape"])
    if array.shape[:3] != source_shape:
        raise ValueError(
            f"reverse produced spatial shape {array.shape[:3]}, expected {source_shape}"
        )

    if squeeze:
        array = array[..., 0]
    return nib.Nifti1Image(array, source_affine)


@register_model
def smri_mae(ckpt_path: str, global_pool: str = "patch"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]

    model_fn = models_mae.__dict__[args["model"]]
    model: models_mae.MaskedAutoencoderViT = model_fn(
        img_size=args["img_size"],
        in_chans=args.get("in_chans", 1),
        patch_size=args["patch_size"],
        **(args.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["model"])

    backbone = SmriMaeBackbone(
        model.encoder,
        global_pool=global_pool,
        pad_to_multiple=args.get("pad_to_multiple", 32),
    )
    transform = SmriMaeTransform(img_size=args["img_size"])

    return backbone, transform
