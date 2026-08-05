# History

Append-only record of rationale, measurements, rejected alternatives and subtle bugs. Dated,
never pruned. Search by tag; don't bulk read.

Tags in use: `[eval]` `[probe]` `[datasets]` `[models]` `[cost]` `[harness]` and one per
backbone/dataset (`[neurojepa]`, `[synthseg]`, `[adni]`, ...).

---

## 2026-07-27 — eval model interface reworked to nifti-in `[eval] [probe]`

Moved `src/nanobrain/eval` from a `(model, transform)` contract to a uniform per-nifti interface:
`global_embed(nifti) -> (D,)` and `dense_embed(nifti) -> (X, Y, Z, D)` on the RAS-canonical grid.
Models canonicalize, normalize and own device placement internally. Shared `canonical()` in
`nifti.py` is the grid contract the seg probe aligns labels to. Commit `50cb2a8`.

Motivation: make segmentation scoring independent of any model's patch grid, so backbones are
comparable. **Cost of the trade: batched inference is gone** — one volume at a time.

`probe_seg.py` was rewritten in the same pass: subject-level repeated CV, generic over K foreground
classes (labels 1..K, 0 = background), scored by per-subject Dice (argmax, implicit 0.5) and
voxel-AP. Two embed passes — subsample for training, full-brain for scoring — to bound peak memory
near one volume's embedding. Subjects with empty ground truth score specificity in Dice and NaN in
AP.

Use `nifti.canonical_img`, not HF's `Nifti1ImageWrapper`, which reorients incorrectly.

## 2026-07-27 — `/polish` skill tuning `[harness] [polish]`

Built and tuned by dry-running against commit `50cb2a8`. Three blind spots found and patched; do
not re-derive them:

1. **Locality.** The first pass only found defects visible inside a single function. It missed a
   probe signature asymmetry that was only visible in `main.py`, where all three probes meet, and
   fixed one `try`/`except` test antipattern without grepping for the identical one in a sibling
   file. Fixed with a peer-symmetry check, a rule to read wiring/entry-point files in full
   regardless of diff size, and a sweep-for-siblings rule after each fix.
2. **Re-litigating settled decisions.** It reported deliberate deferrals as fresh findings. Fixed
   with a triage step that checks TODOs, commit bodies and notes before treating an observation as
   a finding.
3. **Proportionality.** The justification rule was binary (reason / no reason) and approved a
   46-line pure-motion hunk to fix a stale section banner. Connor reverted it. Fixed by weighing
   fix size against problem size, and never mixing pure motion with substantive edits.

Note the pattern: gap 1 was partly *introduced* by the fix for an earlier gap — a read-budget
heuristic that buried the small-but-load-bearing file. Tuning edits need the same scrutiny as code.

## 2026-07-28 — HF `from_generator` caches ignore code changes `[datasets] [cache]`

`Dataset.from_generator` (datasets 5.0.0) keys its cache dir on `gen_kwargs` only. Module-level
globals the generator reads are **not** hashed — dill pickles importable functions by reference, so
changing `fomo.BASE_URL` leaves `Hasher.hash(fomo._generate_task1)` identical, and the
`FOMO_EVAL_BASE_URL` / `FOMO_EVAL_TASK5_URL` overrides silently reuse whichever source was cached
first.

Confirmed the hard way 2026-07-27: after fixing seg/image grid alignment, both
`fomo_task1_infarct_seg` and `fomo_task2_meningioma` reloaded pre-fix data and Task 2 failed again
with the same assertion. Only `rm -rf $HF_HOME/datasets/generator/default-<hash>` forced a rebuild.

One exception, verified 2026-07-28 while adding `cnp.py`: the `Features` schema *is* part of the
fingerprint, so adding a `scanner` column triggered a full rebuild with no manual purge. Schema
changes are safe; changes to generator bodies and preprocessing helpers are not.

A durable fix would thread a `version` value through `gen_kwargs` and bump it when preprocessing
changes.

## 2026-07-28 — confound-first dataset policy `[eval] [datasets]`

Established over CNP, ABIDE and ADHD-200. A probe on frozen features will happily read scanner,
site or age instead of the biology, and an AUROC that is really site is worse than no benchmark.
"Model AUROC minus confound AUROC" was considered and rejected as a fix — AUROCs do not subtract.

Connor consistently spends subjects to kill a shortcut, accepting 265 -> 82 on one CNP task rather
than documenting a confound and living with it. He does not want IQ matched.

Traps found: acquisition-QC flags can be confounded themselves, so check before filtering on them;
and phenotype tables can carry duplicate subject rows — ADHD-200 lists 101 subjects twice.

## 2026-07-28 — probe cost scales hard with embedding width `[probe] [cost]`

`adni_age` RidgeCV over 1000 subjects: **31s at 1024-d** (`random_features`) vs **605s at 3840-d**
(`random_unet`). Feature width is a real wall-clock knob for the sklearn heads, not just a memory
one — worth weighing when picking a backbone's output dim.

## 2026-07-29 — Neuro-JEPA `[models] [neurojepa]`

NYUMedML, arXiv 2606.14957. Fork `clane9/Neuro-JEPA` @ `e9dff69` (relaxed pins, quiet import-time
logging). Gated weights: needs `HF_TOKEN` with granted access.

- **Do not change the input shape from 96x108x96.** `RoPEAttention.separate_positions` decomposes
  the token index assuming `tokens_per_frame = H_p*W_p`, but `PatchEmbed3D.flatten(2)` gives the
  first axis stride `W_p*D_p`. They agree only when `D_p == W_p` (here 8 vs 9), so the y/z position
  axes are entangled. Consistent between their pretraining and inference at the trained shape, but
  any other shape changes the positional semantics. Empirically 2x input gives pooled-feature
  cosine 0.926 vs native, where a 6-voxel shift control gives 0.998.
- **The MONAI chain must run on the GPU.** `Spacing(pixdim=1mm, mode=5)` is 96% of preprocessing:
  **7.03s CPU vs 0.19s GPU (37x)** for an off-1mm volume. MONAI's bspline path needs `cupy`, hence
  `cupy-cuda12x` in the extra; without it MONAI raises `OptionalImportError`.
- **Reordering the transform does not help.** I assumed the README's "resample first" variant would
  dodge the cost and measured it at **0.8x — slower**. The packaged order pads or crops to
  180x216x180 *before* the resample, which usually shrinks the array first.
- **Cohorts are mostly off-1mm**, which is what made this bite: ABIDE 157/250 volumes, ADHD-200
  196/250. ADNI is 1mm, which is why the cliff went unnoticed until the first sweep.
- **Segmentation deliberately deferred**: patches span ~21mm and 768-d per voxel is ~24GB on the
  task-4 grid.
- **Fidelity gap on raw-space tasks**: pretrained on 1mm scans affine-registered to MNI152, and
  their `CropForeground(x > 0.0)` is a no-op after percentile scaling.

Results at integration: `adni_age` MAE 4.339 / r 0.616 vs 5.155 / 0.414 for `random_unet`. Full
ABIDE run 4m22s end to end.

## 2026-07-29 — NeuroVFM, and why flash-attn was dropped `[models] [neurovfm] [flash-attn]`

MLNeurosurg, Nat Med 2026. Fork `clane9/neurovfm`. Weights `mlinslab/neurovfm-encoder` are public,
ViT-B, patch 4x16x16 at 1x1x4mm, varlen tokens + coords, background tokens dropped, ~85.8M params.

**flash-attn evaluated and rejected.** Prebuilt wheels exist on GitHub releases for exactly
cp311/torch2.8/cu12/cxx11abiTRUE and install in 21s (PyPI has only an sdist;
`torch._C._GLIBCXX_USE_CXX11_ABI` picks the variant). But `FusedDense`/`FusedMLP` live in the
`fused_dense_lib` CUDA extension, which **is not in that wheel** — it builds from
`csrc/fused_dense_lib` with `--no-build-isolation` in ~4 min, which means `uv sync` cannot do it in
one pass and nvcc is needed on whatever machine syncs. That fragility, not the wheel, is the
reason. (`from flash_attn.modules.mlp import FusedMLP` *is* a valid import path — it re-exports
from `ops.fused_dense` — it just resolves to `None` without the extension.)

**The pure-torch fallback is not an approximation.** With flash-attn temporarily installed on an
H100, diffed end to end through the encoder: **fp32 rel err 1.2e-6**, bf16 6.6e-3 (bf16 eps is
7.8e-3). `FusedDense` vs `F.linear` and `FusedMLP` vs `fc2(gelu(fc1, approximate="tanh"))` are
bit-exact. `layer_norm_fn(prenorm, residual_in_fp32)` accumulates the residual in fp32 and
normalizes *that*, so a naive `x + residual` in the input dtype would drift;
`neurovfm/models/torch_fallback.py` reproduces it.

**Benchmarking gotcha:** `layer_norm_fn` is looked up as a module global at call time, so
monkeypatching it changes *both* models under comparison. My first diff read a meaningless 0.0.

**SDPA:** swapped the `use_flash_attn=False` path off its materialized `(B,H,N,N)` score matrix
(cosine 0.999995). Honest result: at real token counts (~2000) it is **not a speedup** — 1.6x at
N=500, 0.9x at N=2000 — because the qkv/proj linears dominate. Kept for memory and scaling.

**Fork dependency traps** (a lazy-`__init__` refactor was rejected as out of scope, so the eager
import chain must be satisfied): `neurovfm/__init__.py` pulls the training and VLM stacks, so
pytorch-lightning, outlines, peft, transformers, torchmetrics and openai are import-time
requirements for a frozen encoder. `outlines` must be pinned to **1.1.1** (1.3.x moved
`outlines.processors.structured`). `torch_scatter` has no PyPI wheel but data.pyg.org hosts
`torch_scatter-2.1.2+pt28cu128-cp311`. Installing this set upgrades huggingface-hub 0.36 -> 1.25.

`preprocess_image()` takes a SimpleITK image, so the wrapper converts the nifti in memory
(RAS affine -> LPS origin/direction) rather than writing a temp file. `SelfAttention` hardcodes
`dtype=torch.bfloat16` on its linears, so a pure-fp32 forward needs `model.float()` first.

## 2026-07-29 — SynthSeg `[models] [synthseg]`

Billot, Med Image Anal 2023. Fork `MedARC-AI/SynthSeg` @ `pytorch-port`. The port was written
separately beforehand and did most of the usual work.

- **numpy 2 broke it in three places**, all the same shape: a shape-(1,) `np.where` result assigned
  into a scalar slot. One in `align_volume_to_ref`, firing only for volumes whose axis *order* is
  not x/y/z (LIA, ASL — most raw T1w); three in `get_flip_indices`, firing **unconditionally**, so
  the port could not construct a predictor at all. The `numpy<2` pin was load-bearing; it belongs
  to TensorFlow (<2.16) and moved to the `tensorflow` extra. Running the fork's own suite found the
  second one; the first was found by reading.
- **Their resample IS `F.interpolate`.** The grid `start=-(f-1)/(2f), step=1/f` simplifies to
  `(i+0.5)/f - 0.5`, exactly `align_corners=False`. Verified at 1e-12 vs scipy. Do not hand-roll
  it. One gap: their length is `ceil(n*f)` where torch gives `floor`, and their extra sample is
  edge-clamped. Fix by replicate-padding the **source** by `ceil(1/f)` and trimming to `ceil(n*f)`
  — padding the *output* is wrong, because the appended value is the source edge voxel, not the
  last output slice. Agreement then 6.7e-6 of range.
- **The uncropped padding cliff makes the length matter.** One voxel of FOV difference flips the
  pad target by 32 (193 -> 224 vs 192 -> 192), which moved `global_embed` cosine to 0.998. After
  the source-pad fix, 1.00000000.
- **Cost:** `RegularGridInterpolator` was the whole story — 3.2-3.7s/volume off-1mm, 0.5s with
  torch (7-9x). The gaussian pre-blur is a no-op for coarser-than-1mm inputs (sigmas are zeroed
  when upsampling). `rescale_volume`'s `np.percentile` (~0.24s) is now the largest remaining
  preprocessing cost. End to end ~0.74s/volume on an H100.
- **No 3D gaussian blur builtin exists anywhere**: torch has none, torchvision's is 2D, MONAI's
  disagrees with scipy by 17% of range. A separable `conv3d` matches scipy's interior to float32
  noise; only the boundary convention differs, within `radius` voxels of the edge, and radius is 1
  at realistic sigmas.
- **TF32 gives this U-Net nothing**: 2.7e-4 relative cost on `global_embed` for a 0.99x speedup on
  an H100. Left at the default (a model should not mutate a global backend flag), pinned off in the
  GPU test.
- **Their default inference path does not crop.** `--crop` has no argparse default, so it is `None`
  and `min_pad=128`; the help string's "Default is 192" is stale and wires nothing. A brain-centred
  192 crop was built and then removed to match their default — Connor's call, and worth knowing it
  changed features a lot (cosine 0.83-0.91 vs uncropped).
- **Pooling masks the padding.** The scan occupies 24-99% of the padded volume depending on FOV, so
  an unmasked mean mixes brain and padding in a subject-dependent ratio. `_bottleneck_box` keeps
  cells strictly inside the scan (**ceil** for the start, floor for the stop — flooring both
  included ~14% pure padding). Still approximate: a cell's receptive field reaches past the STRIDE
  voxels it covers.
- `SynthSegPredictor` is not an `nn.Module` and freezes `self.device` at construction, so the
  wrapper owns placement and never calls its `forward_embedding`.

Result: `dlbs_sex` AUROC 0.9863 (bal acc 0.947), 464 subjects in 5m44s.

## 2026-07-29 — sMRI MAE `[models] [smri_mae]`

Mihir's in-house 3D ViT MAE over the vendored `src/smri_mae/`. Not a fork — the training code is in
the repo, so there is no upstream pipeline to equivalence-test the transform against.

**Test checkpoint:** `/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth`
— epoch 99, `mae_vit_large`, patch 8, `img_size` 208x240x208, 26x30x26 patch grid, 1024-d. 3.9 GB
because the optimizer state rides along, so load with `mmap=True`. Its `args.model_kwargs` carries
`decoding: attn`, which today's `MaskedAutoencoderViT` no longer takes — filter through
`smri_mae.utils.filter_kwargs` or construction raises TypeError. Weights load strictly (0 missing,
0 unexpected).

**Pretraining preprocessing** (`src/preprocessing/pipeline.py` in `/data/connor/smri-fm`, plus
`datasets/FOMO300/metadata.json`): ANTs *rigid* registration to TemplateFlow `MNI152NLin2009cAsym`
at 1mm (193x229x193), SynthSeg brain mask, per-sample z-score over masked brain voxels *after* the
shape fit to 208x240x208, stored as fp16 brain-only voxels plus a bit-packed mask. The eval
transform's mean-intensity threshold is a stand-in for SynthSeg and keeps skull and neck, which
pretraining never saw. That is the main known fidelity gap.

**The forward pass will not run on the login node.** The blocks use nested-tensor SDPA, whose
backend selection calls `torch._C._can_use_cudnn_attention`, which raises `AcceleratorError` on a
CUDA-built torch with no visible device. Tests therefore build a **depth-0** encoder, which keeps
patchify, masking, pos embed and pooling while skipping every attention block. Real forward passes:
0.13 s/volume warm, 2.1 GB peak at 208x240x208.

To re-check the axis order (open thread 4 in NOTES), read a shard directly out of
`/data/connor/smri-fm/data/FOMO300/train/*.tar` — keys `image_values.npy`, `img_mask.npy`,
`meta.json`.

## 2026-07-29 — `medarc/adni-mini` re-uploaded in place `[datasets] [adni]`

Replaced on 2026-07-29 (sha `629f32de`) by a local derivative of `medarc/adni-mini-v1-3`. Three
things changed at once: split renamed `test` -> `eval` and 1000 -> 1200 scans; images brain-masked
(`image[synthseg_dseg == 0] = 0.0`), so voxels outside the SynthSeg brain are exactly zero — a real
change to model input, not just packaging; and the README's `dataset_info` was broken, which is
what actually broke loading.

Task sizes after: age/sex 1200, ad_cn 750, amyloid_centiloid 997, tau_suvr 555. Any ADNI number
computed before this date came from unmasked images and a different cohort size.

Fixed upstream in `3fdea48` (2026-07-29), which touched only the README — shards byte-identical to
`629f32d` — so `tasks/adni.py` is back on a plain `load_dataset(revision=...)`.

## 2026-07-30 — HF README `dataset_info` overrides the parquet schema `[datasets] [hf]`

`load_dataset("<repo>")` builds its target schema from the README YAML `dataset_info.features`, not
from the metadata embedded in the parquet shards. If the README under-declares columns, every shard
fails to cast with `CastError: ... because column names don't match`, naming only the declared
subset as the target. Bypass by loading the shards as a plain parquet dataset, which reads the
embedded schema and recovers `ClassLabel` and `Nifti` types:

```python
load_dataset("parquet", data_files={"eval": "hf://datasets/<repo>/data/eval-*.parquet"}, split="eval")
```

**Pinning a revision on that path needs the `@rev` URL form.** `load_dataset(revision=...)` only
applies when loading by repo id; the packaged `arrow`/`parquet` builders take it and ignore it
silently — verified, a garbage revision still loaded all 1,000 rows of `medarc/ppmi-mini`. Put it
in the path, where a bad revision fails loudly. On the plain repo-id path the kwarg *is* honored.

## 2026-07-30 — PPMI-mini task selection `[datasets] [ppmi]`

`medarc/ppmi-mini` v0.1 has no site or scanner column, so scan date is the only acquisition handle,
and the cohorts were enrolled in waves. Unmatched confound AUROC floors: PD vs CN — scan year 0.61;
Prodromal vs CN — scan year 0.84, header geometry (shape + zooms) 0.79; PD vs Prodromal — scan year
0.78, geometry 0.69; SWEDD vs CN — geometry 0.66.

Shipped `ppmi_age` (999), `ppmi_pd_cn` (426), `ppmi_pd_prodromal` (324), matched on (scan-year band,
age band, sex). **Rejected:** Prodromal vs CN, whose matched pool is only 146 and still leaves
geometry at 0.57 (CN is 2010-2013 with a small 2021-2024 tail, prodromal almost all 2018+, so they
barely overlap in time); and SWEDD vs CN, where SWEDD is 61 subjects all scanned 2011-2013,
matching still leaves scan year at 0.73, and the group is biologically mixed anyway. No `ppmi_sex`
— the suite already has three.

**Header geometry is an independent scanner proxy** on this dataset, worth checking on any release
that ships raw multi-site scans without a scanner column. Matching on year bands (2013/2017/2021)
is enough; exact-year or geometry-class matching costs 130-140 more subjects and buys nothing once
the floors are at chance. One non-3D volume excluded: `sub-3200_ses-20101202_T1w`, (512, 512, 78, 2).

## 2026-07-29..30 — working-style corrections `[harness] [feedback]`

Four corrections in one week, all the same defect on my side: I over-produce the artifact.

- **Comments and docstrings** trimmed twice in one session, then deleted outright in the fork
  (`2690d9a`) — including the one-line invariant I had already trimmed them to, annotating a change
  from `np.where(...)` to `np.where(...)[0][0]`. Connor's framing: *"make sure the code is
  obviously correct and not bother explaining the history that got us here."*
- **Commit bodies** then grew to absorb what I'd cut from comments — the PPMI body ran five
  paragraphs. Same defect relocated.
- **Plots**: I built the eval radar to a full design spec (validated palette, CVD secondary
  encoding, dataset wedges, per-spoke chance arcs). *"the data viz skill is a bit intense for my
  liking… ideally the first version should be short and easily readable code."*
- **Rewriting his drafts** into my own house style when handed a starter wrapper, instead of the
  minimum diff.

Also: *"pls dont submit for gpu node btw"* — the cluster is shared and the queue is his.

These are now encoded in `.claude/CLAUDE.md` (minimum artifact, comments, HISTORY as the sink) and
`.claude/skills/gpu-session`, so the per-instance memories that recorded them are deleted.

## 2026-08-01 — harness migration `[harness]`

Replaced `AGENTS.md` (75 lines, 10 numbered sections) with `.claude/CLAUDE.md` (78 lines:
principles, process stages, where-things-live, rules, conventions). Cut from the old file: wandb /
grad-norm logging specifics, multiple-seeds-per-experiment, the debug/smoke config bullet, the
experiment-folder bullet (moved to the layout table), the standalone docstring rule, and most
worked examples. Added: entropy as the frame, minimum artifact, the stage vocabulary, the
where-things-live table, and a 100-line budget on the file itself.

Deleted the 18-file assistant memory directory at
`~/.claude/projects/-data-connor-nanobrain-1/memory/`, routing its contents here, to
`.claude/NOTES.md`, and to two skills. Rationale: memory outside the repo is state Connor cannot
read, review or delete, and it had grown 45KB in ten days with no pruning mechanism.

Rejected during the design: a `.claude/rules/` directory (no content yet, and its boundary against
CLAUDE.md's principles was undefined — a speculative seam); a `.claude/scratch/` directory
(duplicated the existing `.scratch/`); and a "every addition names what it replaces" rule,
superseded by the line budget, which `wc -l` can check.
