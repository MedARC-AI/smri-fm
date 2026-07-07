from collections.abc import Mapping
from io import BytesIO

import torch

from matplotlib import pyplot as plt
from PIL import Image
from timm.layers import to_3tuple
from torch import Tensor

VIEW_NAMES = {
    "sagittal": "Sagittal",
    "saggital": "Sagittal",
    "axial": "Axial",
    "coronal": "Coronal",
}

BLOCK_MASK_COLOR = (0.12, 0.23, 0.55)
RANDOM_MASK_COLOR = (0.57, 0.25, 0.05)


def fig2pil(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def raw_stats_from_batch(batch: dict) -> tuple[Tensor | None, Tensor | None]:
    metas = batch.get("meta")
    if not metas:
        return None, None

    means = []
    stds = []
    for meta in metas:
        try:
            mean = meta["raw_mean"]
            std = meta["raw_std"]
        except (KeyError, TypeError):
            return None, None
        if mean in ("", None) or std in ("", None):
            return None, None
        means.append(float(mean))
        stds.append(float(std))
    return torch.tensor(means), torch.tensor(stds)


def plot_mask_pred(
    target: Tensor,
    pred: Tensor,
    pred_mask: Tensor | None = None,
    block_mask: Tensor | None = None,
    img_mask: Tensor | None = None,
    sample_idx: int = 0,
    channel_idx: int = 0,
    slice_idx: int | Mapping[str, int] | None = None,
    patch_size: int | tuple[int, int, int] = 16,
    views: tuple[str, ...] = ("sagittal", "axial", "coronal"),
    cmap: str = "gray",
    figsize: tuple[float, float] | None = None,
    raw_mean: float | Tensor | None = None,
    raw_std: float | Tensor | None = None,
):
    """
    Plot masked input, reconstruction composite, and target slices.

    When ``block_mask`` is provided, the masked input row colors block-hidden
    pixels blue and random-hidden pixels amber.
    """
    target_vol = _select_volume(target, sample_idx=sample_idx, channel_idx=channel_idx)
    pred_vol = _select_volume(pred, sample_idx=sample_idx, channel_idx=channel_idx)
    if raw_mean is not None and raw_std is not None:
        raw_mean = _select_scalar(raw_mean, sample_idx=sample_idx)
        raw_std = _select_scalar(raw_std, sample_idx=sample_idx)
        target_vol = target_vol * raw_std + raw_mean
        pred_vol = pred_vol * raw_std + raw_mean
    pred_mask_vol = (
        torch.zeros_like(target_vol)
        if pred_mask is None
        else _select_volume(pred_mask, sample_idx=sample_idx, channel_idx=channel_idx) > 0
    )
    block_mask_vol = None
    if block_mask is not None:
        block_mask_vol = (
            _select_volume(block_mask, sample_idx=sample_idx, channel_idx=channel_idx) > 0
        )
    img_mask_vol = None
    if img_mask is not None:
        img_mask_vol = _select_volume(img_mask, sample_idx=sample_idx, channel_idx=channel_idx) > 0

    composite_vol = _prediction_composite(target_vol, pred_vol, pred_mask_vol)
    vmin, vmax = _intensity_limits(target_vol, img_mask_vol)

    patch_size = to_3tuple(patch_size)
    view_items = []
    for view in views:
        view_key = view.lower()
        if view_key not in VIEW_NAMES:
            raise ValueError(f"unknown MRI view {view!r}; expected one of {tuple(VIEW_NAMES)}")
        target_slice = _extract_view_slice(target_vol, view_key, slice_idx)
        composite_slice = _extract_view_slice(composite_vol, view_key, slice_idx)
        mask_slice = _extract_view_slice(pred_mask_vol.float(), view_key, slice_idx) > 0
        block_mask_slice = None
        if block_mask_vol is not None:
            block_mask_slice = (
                _extract_view_slice(block_mask_vol.float(), view_key, slice_idx) > 0
            )
        img_mask_slice = None
        if img_mask_vol is not None:
            img_mask_slice = _extract_view_slice(img_mask_vol.float(), view_key, slice_idx) > 0
        view_items.append(
            {
                "key": view_key,
                "title": VIEW_NAMES[view_key],
                "target": _masked_input_display(
                    target_slice,
                    mask_slice,
                    img_mask_slice,
                    vmin,
                    vmax,
                    block_mask_slice,
                ),
                "composite": _apply_display_mask(composite_slice, img_mask_slice, vmin),
                "actual": _apply_display_mask(target_slice, img_mask_slice, vmin),
                "mask": mask_slice,
                "block_mask": block_mask_slice,
                "img_mask": img_mask_slice,
                "patch_rc": _view_patch_size(view_key, patch_size),
            }
        )

    _crop_view_items(view_items)

    fig, axes, layout = _make_figure_canvas(view_items, figsize=figsize)
    for item, x in zip(view_items, layout["col_centers"]):
        fig.text(
            x,
            layout["title_y"],
            item["title"],
            ha="center",
            va="center",
            color="#f8fafc",
            fontsize=7,
        )
    for label, y in zip(("Masked", "Pred", "Actual"), layout["row_centers"]):
        fig.text(
            layout["label_x"],
            y,
            label,
            ha="right",
            va="center",
            color="#cbd5e1",
            fontsize=6,
        )

    for item, top_ax, middle_ax, bottom_ax in zip(view_items, axes[0], axes[1], axes[2]):
        _imshow(top_ax, item["target"], cmap=cmap, vmin=vmin, vmax=vmax)
        _style_axis(top_ax)

        _imshow(middle_ax, item["composite"], cmap=cmap, vmin=vmin, vmax=vmax)
        _style_axis(middle_ax)

        _imshow(bottom_ax, item["actual"], cmap=cmap, vmin=vmin, vmax=vmax)
        _style_axis(bottom_ax)
    return fig


def _select_volume(
    x: Tensor,
    sample_idx: int = 0,
    channel_idx: int = 0,
) -> Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 5:
        return x[sample_idx, channel_idx]
    if x.ndim == 4:
        return x[sample_idx]
    if x.ndim == 3:
        return x
    raise ValueError(f"expected a 3D volume tensor, got shape {tuple(x.shape)}")


def _select_scalar(value: float | Tensor, sample_idx: int = 0) -> float:
    if isinstance(value, Tensor):
        value = value.detach().float().cpu()
        if value.ndim > 0:
            value = value.reshape(-1)[sample_idx]
        return float(value)
    return float(value)


def _prediction_composite(target: Tensor, pred: Tensor, pred_mask: Tensor) -> Tensor:
    pred_mask = pred_mask.to(dtype=target.dtype)
    return target * (1 - pred_mask) + pred * pred_mask


def _extract_view_slice(
    volume: Tensor,
    view: str,
    slice_idx: int | Mapping[str, int] | None = None,
) -> Tensor:
    if isinstance(slice_idx, Mapping):
        slice_idx = slice_idx.get(view)

    if view in ("sagittal", "saggital"):
        idx = _resolve_slice_idx(volume.shape[0], slice_idx)
        return volume[idx, :, :].transpose(0, 1).flip(0)
    if view == "axial":
        idx = _resolve_slice_idx(volume.shape[2], slice_idx)
        return volume[:, :, idx].transpose(0, 1).flip(0)
    if view == "coronal":
        idx = _resolve_slice_idx(volume.shape[1], slice_idx)
        return volume[:, idx, :].transpose(0, 1).flip(0)
    raise ValueError(f"unknown MRI view {view!r}")


def _resolve_slice_idx(size: int, slice_idx: int | None) -> int:
    idx = size // 2 if slice_idx is None else int(slice_idx)
    if idx < 0:
        idx += size
    if idx < 0 or idx >= size:
        raise IndexError(f"slice index {idx} is out of bounds for axis with size {size}")
    return idx


def _intensity_limits(volume: Tensor, mask: Tensor | None = None) -> tuple[float, float]:
    values = volume[mask] if mask is not None and mask.any() else volume.flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0, 1.0
    if values.numel() < 32:
        vmin = values.min()
        vmax = values.max()
    else:
        vmin, vmax = torch.quantile(values, torch.tensor([0.005, 0.995]))
    if torch.isclose(vmin, vmax):
        delta = max(abs(float(vmin)) * 0.05, 1.0)
        return float(vmin) - delta, float(vmax) + delta
    return float(vmin), float(vmax)


def _apply_display_mask(image: Tensor, mask: Tensor | None, fill_value: float) -> Tensor:
    if mask is None:
        return image
    return torch.where(mask, image, torch.full_like(image, fill_value))


def _masked_input_display(
    image: Tensor,
    pred_mask: Tensor,
    img_mask: Tensor | None,
    fill_value: float,
    max_value: float,
    block_mask: Tensor | None = None,
) -> Tensor:
    if block_mask is not None:
        return _masked_input_rgb_display(
            image,
            pred_mask,
            img_mask,
            min_value=fill_value,
            max_value=max_value,
            block_mask=block_mask,
        )
    display = torch.where(pred_mask, torch.full_like(image, fill_value), image)
    return _apply_display_mask(display, img_mask, fill_value)


def _masked_input_rgb_display(
    image: Tensor,
    pred_mask: Tensor,
    img_mask: Tensor | None,
    min_value: float,
    max_value: float,
    block_mask: Tensor,
) -> Tensor:
    scale = max(max_value - min_value, 1e-6)
    gray = ((image - min_value) / scale).clamp(0.0, 1.0)
    if img_mask is not None:
        gray = torch.where(img_mask, gray, torch.zeros_like(gray))
        pred_mask = pred_mask & img_mask
        block_mask = block_mask & img_mask

    rgb = gray.unsqueeze(-1).repeat(1, 1, 3)
    block_pixels = pred_mask & block_mask
    random_pixels = pred_mask & ~block_mask
    rgb[block_pixels] = image.new_tensor(BLOCK_MASK_COLOR)
    rgb[random_pixels] = image.new_tensor(RANDOM_MASK_COLOR)
    return rgb


def _crop_view_items(view_items: list[dict]) -> None:
    for item in view_items:
        mask = item["img_mask"]
        if mask is None:
            mask = item["actual"] != item["actual"].min()
        row_slice, col_slice = _content_crop(mask, item["patch_rc"])
        for key in ("target", "composite", "actual", "mask"):
            item[key] = item[key][row_slice, col_slice]
        if item["img_mask"] is not None:
            item["img_mask"] = item["img_mask"][row_slice, col_slice]


def _content_crop(mask: Tensor, patch_size: tuple[int, int]) -> tuple[slice, slice]:
    mask = mask.detach().cpu() > 0
    if not mask.any():
        return slice(None), slice(None)

    rows, cols = mask.nonzero(as_tuple=True)
    patch_h, patch_w = patch_size
    height, width = mask.shape
    row0 = max((int(rows.min()) // patch_h - 1) * patch_h, 0)
    row1 = min((int(rows.max()) // patch_h + 2) * patch_h, height)
    col0 = max((int(cols.min()) // patch_w - 1) * patch_w, 0)
    col1 = min((int(cols.max()) // patch_w + 2) * patch_w, width)
    return slice(row0, row1), slice(col0, col1)


def _view_patch_size(view: str, patch_size: tuple[int, int, int]) -> tuple[int, int]:
    p_x, p_y, p_z = patch_size
    if view in ("sagittal", "saggital"):
        return p_z, p_y
    if view == "axial":
        return p_y, p_x
    if view == "coronal":
        return p_z, p_x
    raise ValueError(f"unknown MRI view {view!r}")


def _make_figure_canvas(
    view_items: list[dict[str, Tensor | str | tuple[int, int]]],
    figsize: tuple[float, float] | None = None,
):
    dpi = 160
    left = 58
    right = 6
    top = 18
    bottom = 8
    row_gap = 14
    col_gap = 8
    widths = [int(item["target"].shape[1]) for item in view_items]
    heights = [int(item["target"].shape[0]) for item in view_items]
    row_h = max(heights)
    num_rows = 3
    fig_w = left + right + sum(widths) + col_gap * (len(widths) - 1)
    fig_h = top + bottom + row_h * num_rows + row_gap * (num_rows - 1)

    scale = 1.35
    if figsize is not None:
        requested_w = figsize[0] * dpi
        requested_h = figsize[1] * dpi
        scale = max(requested_w / fig_w, requested_h / fig_h)
    figsize = (fig_w * scale / dpi, fig_h * scale / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#0b0f14")

    axes = [[] for _ in range(num_rows)]
    col_centers = []
    x = left
    for width, height in zip(widths, heights):
        col_centers.append((x + width / 2) / fig_w)
        ys = [
            bottom + (num_rows - row - 1) * (row_h + row_gap) + (row_h - height) / 2
            for row in range(num_rows)
        ]
        for row, y in enumerate(ys):
            axes[row].append(
                fig.add_axes(
                    [
                        x / fig_w,
                        y / fig_h,
                        width / fig_w,
                        height / fig_h,
                    ],
                    facecolor="black",
                )
            )
        x += width + col_gap
    row_centers = [
        (bottom + (num_rows - row - 1) * (row_h + row_gap) + row_h / 2) / fig_h
        for row in range(num_rows)
    ]
    layout = {
        "col_centers": col_centers,
        "row_centers": row_centers,
        "label_x": (left - 8) / fig_w,
        "title_y": (fig_h - top / 2) / fig_h,
    }

    return fig, axes, layout


def _imshow(ax, image: Tensor, cmap: str, vmin: float, vmax: float) -> None:
    kwargs = {
        "interpolation": "nearest",
        "origin": "upper",
    }
    if image.ndim == 2:
        kwargs.update({"cmap": cmap, "vmin": vmin, "vmax": vmax})
    ax.imshow(image, **kwargs)


def _style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
