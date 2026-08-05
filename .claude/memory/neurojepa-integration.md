---
name: neurojepa-integration
description: Neuro-JEPA fork, gated weights, deferred segmentation and MNI fidelity gap — integration facts for the eval wrapper.
metadata:
  type: reference
---

NYUMedML, arXiv 2606.14957. Fork `clane9/Neuro-JEPA` @ `e9dff69` (relaxed pins, quiet import-time
logging). Weights are **gated** — needs `HF_TOKEN` with granted access.

- **Segmentation deliberately deferred:** patches span ~21mm and 768-d per voxel is ~24GB on the
  FOMO task-4 grid.
- **Fidelity gap on raw-space tasks:** pretrained on 1mm scans affine-registered to MNI152, and
  their `CropForeground(x > 0.0)` is a no-op after percentile scaling.

At integration: `adni_age` MAE 4.339 / r 0.616, vs 5.155 / 0.414 for `random_unet`. Full ABIDE run
4m22s end to end.

See [[neurojepa-fixed-input-shape]], [[neurojepa-monai-spacing-gpu]].
