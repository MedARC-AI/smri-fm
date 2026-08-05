---
name: add-eval-model
description: Add a pretrained backbone to the nanobrain frozen-feature eval suite. Use when wiring up a new model for evaluation — forking upstream, writing the wrapper and preprocessing, and verifying it against the reference pipeline.
scope: project
---

# Adding an eval backbone

Settled over Neuro-JEPA, NeuroVFM, SynthSeg and sMRI MAE. The model-side counterpart to
`add-eval-dataset`. Per-backbone findings are in `.claude/HISTORY.md` under `[models]`.

## 1. Assess before writing anything

Clone upstream to `third_party/<name>` and read the inference path. Report back and get agreement
before writing code:

- where the weights live and whether they are gated (Neuro-JEPA needs `HF_TOKEN`)
- whether the pins can resolve against nanobrain
- what the inference entry point is
- whether `dense_embed` is feasible

Deferring segmentation with a stated reason is fine — most backbones have.

## 2. Fork with the minimum diff, pin the SHA

**Do the minimum diff necessary on the fork.** Relax `==` pins to `>=`, loosen `requires-python`,
and otherwise change only what blocks import or install. Don't split modules out or restyle their
code. The same rule applies to a starter wrapper Connor hands over: keep his structure, comments
and helpers, and change only what is broken or missing.

Push to a branch on his fork, pin the commit in `[tool.uv.sources]`, add an optional extra in
`[project.optional-dependencies]`. `uv sync --extra <name>` is the acceptance test for this half.

## 3. Expect one dependency that will not install cleanly

Every backbone has had one. Seen so far: a CUDA extension the prebuilt wheels don't ship
(flash-attn's `fused_dense_lib`); an sdist-only package needing a compile against torch
(`torch-scatter`); a package whose API moved and needs a hard pin (`outlines`).

Fixes in order of preference: guard the import with `try`/`except ImportError` in the style the
file already uses; move the stack to an optional extra; pin. Check what an eager `__init__.py`
chain drags in — NeuroVFM pulled pytorch-lightning, outlines, peft and openai for a frozen encoder.

## 4. The preprocessing equivalence test is the highest-value thing you write

Their loaders take paths or directories, so reimplement the chain for the in-memory nifti the
harness hands over, then test it against *their* pipeline reading the same volume from `tmp_path`
— voxel for voxel or token for token. Parametrize over an off-native spacing and an L,P,S affine
that forces reorientation. This caught the bug that mattered: without a MONAI `MetaTensor` carrying
the affine, `Spacing` silently no-ops. Gate the module with `pytest.importorskip`.

Where there is no upstream inference path to test against (sMRI MAE), reconstruct the transform
from the pretraining pipeline and write down which parts are stand-ins.

## 5. Profile the preprocessing, not the model

It dominates, and it is where the surprises are. Time each transform separately, run the chain on
the GPU, and **benchmark warm** — cupy's first call is ~5s of kernel JIT and made a 37x win look
like 1.4x. Check the cohort's actual voxel spacings before concluding anything: ADNI is 1mm and
hides all resampling cost, while ABIDE and ADHD-200 are mostly not.

## 6. Verify substitutions in fp32

When swapping an implementation (SDPA for flash kernels, pure torch for fused ops), diff against
the real thing before deleting it. bf16 rounding is ~7e-3 and will mask a real algorithmic
difference; fp32 agreement should be ~1e-6. Watch for module-global lookups — monkeypatching
`layer_norm_fn` changed *both* sides of my first comparison and read a meaningless 0.0.

Assert the state dict loads with 0 missing and 0 unexpected. Upstream loaders often use
`strict=False` and hide mismatches.

## 7. Reach for the builtin before reimplementing

Before transcribing a reference's interpolation or filtering loop, do the algebra to see which
builtin it already is — SynthSeg's resample grid turns out to be literally
`F.interpolate(align_corners=False)`. Check what actually exists first (torch has no 3D gaussian
blur; MONAI's is not scipy-compatible). Reserve hand-rolled code for when nothing fits, and keep
the deviation to one clearly-stated invariant.

## 8. Smoke test through the harness

`dlbs_sex` is the anchor — sex, 464 subjects, and every backbone lands >= 0.96 AUROC, so anything
much below means the wiring is wrong rather than the model being bad. Avoid
`abide_autism_control`, which is at chance for everything.

Use `gpu-session` for anything needing a device.

## Interface reminders

The model owns device placement and canonicalizes internally. Use `nifti.canonical_img` — HF's
`Nifti1ImageWrapper` reorients incorrectly. Register the builder with `@register_model`.
