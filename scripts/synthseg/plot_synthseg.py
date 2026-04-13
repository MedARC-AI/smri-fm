"""Plot a 2x8 grid of original images (top) and SynthSeg segmentations (bottom)."""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# FreeSurfer color LUT for SynthSeg labels (subcortical + cortical parcellation)
FREESURFER_LUT = {
    0: (0, 0, 0),
    2: (245, 245, 245),
    3: (205, 62, 78),
    4: (120, 18, 134),
    5: (196, 58, 250),
    7: (220, 248, 164),
    8: (230, 148, 34),
    10: (0, 118, 14),
    11: (122, 186, 220),
    12: (236, 13, 176),
    13: (12, 48, 255),
    14: (204, 182, 142),
    15: (42, 204, 164),
    16: (119, 159, 176),
    17: (220, 216, 20),
    18: (103, 255, 255),
    24: (60, 60, 60),
    26: (255, 165, 0),
    28: (165, 42, 42),
    41: (0, 225, 0),
    42: (205, 62, 78),
    43: (120, 18, 134),
    44: (196, 58, 250),
    46: (220, 248, 164),
    47: (230, 148, 34),
    49: (0, 118, 14),
    50: (122, 186, 220),
    51: (236, 13, 176),
    52: (12, 48, 255),
    53: (220, 216, 20),
    54: (103, 255, 255),
    58: (255, 165, 0),
    60: (165, 42, 42),
    # Cortical parcellation (Desikan-Killiany) - left hemisphere
    1001: (25, 100, 40),
    1002: (125, 100, 160),
    1003: (100, 25, 0),
    1005: (220, 20, 100),
    1006: (220, 20, 10),
    1007: (180, 220, 140),
    1008: (220, 60, 220),
    1009: (180, 40, 120),
    1010: (140, 20, 140),
    1011: (20, 30, 140),
    1012: (35, 75, 50),
    1013: (225, 140, 140),
    1014: (200, 35, 75),
    1015: (160, 100, 50),
    1016: (20, 220, 60),
    1017: (60, 220, 60),
    1018: (220, 180, 140),
    1019: (20, 100, 50),
    1020: (220, 60, 20),
    1021: (120, 100, 60),
    1022: (220, 20, 20),
    1023: (220, 180, 220),
    1024: (60, 20, 220),
    1025: (160, 140, 180),
    1026: (80, 20, 140),
    1027: (75, 50, 125),
    1028: (20, 220, 160),
    1029: (20, 180, 140),
    1030: (140, 220, 220),
    1031: (80, 160, 20),
    1032: (100, 0, 100),
    1033: (70, 70, 70),
    1034: (150, 150, 200),
    1035: (255, 192, 32),
    # Right hemisphere cortical parcellation
    2001: (25, 100, 40),
    2002: (125, 100, 160),
    2003: (100, 25, 0),
    2005: (220, 20, 100),
    2006: (220, 20, 10),
    2007: (180, 220, 140),
    2008: (220, 60, 220),
    2009: (180, 40, 120),
    2010: (140, 20, 140),
    2011: (20, 30, 140),
    2012: (35, 75, 50),
    2013: (225, 140, 140),
    2014: (200, 35, 75),
    2015: (160, 100, 50),
    2016: (20, 220, 60),
    2017: (60, 220, 60),
    2018: (220, 180, 140),
    2019: (20, 100, 50),
    2020: (220, 60, 20),
    2021: (120, 100, 60),
    2022: (220, 20, 20),
    2023: (220, 180, 220),
    2024: (60, 20, 220),
    2025: (160, 140, 180),
    2026: (80, 20, 140),
    2027: (75, 50, 125),
    2028: (20, 220, 160),
    2029: (20, 180, 140),
    2030: (140, 220, 220),
    2031: (80, 160, 20),
    2032: (100, 0, 100),
    2033: (70, 70, 70),
    2034: (150, 150, 200),
    2035: (255, 192, 32),
}


def labels_to_rgb(seg_slice):
    """Convert a 2D label array to an RGB image using the FreeSurfer LUT."""
    h, w = seg_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in FREESURFER_LUT.items():
        mask = seg_slice == label_id
        if mask.any():
            rgb[mask] = color
    # Any unknown labels get a random but consistent color
    unique = np.unique(seg_slice)
    for label_id in unique:
        if label_id not in FREESURFER_LUT and label_id != 0:
            np.random.seed(int(label_id))
            rgb[seg_slice == label_id] = np.random.randint(50, 255, 3)
    return rgb


def load_reoriented(nii_path):
    """Load a NIfTI and reorient to RAS for consistent display."""
    img = nib.load(str(nii_path))
    img_ras = nib.as_closest_canonical(img)
    return np.asarray(img_ras.dataobj)


def get_mid_axial_slice(nii_path):
    """Load a NIfTI and return the middle axial (z-axis in RAS) slice."""
    data = load_reoriented(nii_path)
    mid = data.shape[2] // 2
    return np.rot90(data[:, :, mid])


def main():
    parser = argparse.ArgumentParser(description="Plot SynthSeg results grid")
    parser.add_argument("--raw-dir", required=True, help="Directory with raw NIfTI scans")
    parser.add_argument("--seg-dir", required=True, help="Directory with SynthSeg outputs")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="SynthSeg Segmentation Results", help="Plot title")
    parser.add_argument("--ncols", type=int, default=8, help="Number of columns")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    seg_dir = Path(args.seg_dir)

    # Find matching pairs
    seg_files = sorted(seg_dir.glob("*_synthseg.nii.gz"))
    pairs = []
    for seg_path in seg_files:
        raw_name = seg_path.name.replace("_synthseg.nii.gz", ".nii.gz")
        raw_path = raw_dir / raw_name
        if raw_path.exists():
            pairs.append((raw_path, seg_path))

    if not pairs:
        print("No matching raw/segmentation pairs found!")
        return

    ncols = args.ncols
    nrows_per_pair = 2  # original + segmentation
    n_subject_rows = (len(pairs) + ncols - 1) // ncols  # ceil division
    total_rows = n_subject_rows * nrows_per_pair

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(ncols * 2.2, total_rows * 2.2),
    )
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    # Turn off all axes first
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for idx, (raw_path, seg_path) in enumerate(pairs):
        col = idx % ncols
        row_group = idx // ncols
        raw_row = row_group * nrows_per_pair
        seg_row = raw_row + 1

        # Raw image
        raw_slice = get_mid_axial_slice(raw_path)
        axes[raw_row, col].imshow(raw_slice, cmap="gray")
        sub_id = raw_path.name.split("_")[0]
        axes[raw_row, col].set_title(sub_id, fontsize=7)

        # Segmentation
        seg_slice = get_mid_axial_slice(seg_path)
        seg_rgb = labels_to_rgb(seg_slice.astype(int))
        axes[seg_row, col].imshow(seg_rgb)

    # Row labels
    for row_group in range(n_subject_rows):
        raw_row = row_group * nrows_per_pair
        axes[raw_row, 0].set_ylabel("Original", fontsize=9)
        axes[raw_row + 1, 0].set_ylabel("SynthSeg", fontsize=9)

    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output} ({len(pairs)} subjects)")


if __name__ == "__main__":
    main()
