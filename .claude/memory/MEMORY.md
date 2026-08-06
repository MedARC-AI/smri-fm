# Memory

One fact per file. Read the ones whose hook matches what you're doing; don't bulk read.

Record observations and arguments, not descriptions of what another file currently contains — that
is the failure mode that killed the previous `HISTORY.md`. See
[memory-lives-in-the-repo](memory-lives-in-the-repo.md).

## Working style

- [over-producing-the-artifact](over-producing-the-artifact.md) — the most repeated correction; cutting it in one place relocates it to another.
- [no-explanatory-comments](no-explanatory-comments.md) — code obviously correct, not annotated with the history that got us here.
- [plain-matplotlib-for-internal-plots](plain-matplotlib-for-internal-plots.md) — default colors, short code; full treatment only for published figures.
- [dont-submit-slurm-jobs](dont-submit-slurm-jobs.md) — the GPU queue is Connor's.
- [code-review-agent-blind-spots](code-review-agent-blind-spots.md) — locality, re-litigating deferrals, disproportionate fixes. Don't re-derive.

## Eval harness

- [eval-interface-is-nifti-in](eval-interface-is-nifti-in.md) — why the per-nifti contract exists; the batching it cost, and why that price is no longer owed.
- [seg-probe-design](seg-probe-design.md) — patch point cloud in world mm, nearest-patch assignment, one forward pass.
- [seg-probe-world-coord-guards](seg-probe-world-coord-guards.md) — measured: the coverage assert catches units and origin, never an axis swap.
- [verifying-patch-world-coords](verifying-patch-world-coords.md) — the marker test must probe a pre-attention layer; which statistic per architecture.
- [smri-mae-pad-to-multiple-is-inert](smri-mae-pad-to-multiple-is-inert.md) — padded token slots are dropped before the blocks, so it cannot move a number.
- [no-memory-references-in-code](no-memory-references-in-code.md) — state the fact in the source; keep the rationale in memory, uncited.
- [dice-ceiling-diagnostic-deferred](dice-ceiling-diagnostic-deferred.md) — separating "too coarse" from "uninformative"; deferred, with the exact formula and the ratio trap.
- [hf-nifti-wrapper-reorients-wrong](hf-nifti-wrapper-reorients-wrong.md) — use `nifti.canonical_img`, never HF's wrapper.
- [probe-cost-scales-with-embed-width](probe-cost-scales-with-embed-width.md) — 31s at 1024-d vs 605s at 3840-d.

## Datasets and HF

- [hf-from-generator-cache-ignores-code](hf-from-generator-cache-ignores-code.md) — editing a generator silently reuses stale cached data.
- [hf-readme-overrides-parquet-schema](hf-readme-overrides-parquet-schema.md) — README YAML beats the shards; the error blames the wrong file.
- [hf-revision-ignored-by-packaged-builders](hf-revision-ignored-by-packaged-builders.md) — `revision=` is silently dropped; use the `@rev` URL form.
- [adni-mini-reuploaded-in-place](adni-mini-reuploaded-in-place.md) — pre-2026-07-29 ADNI numbers are not comparable.
- [ppmi-mini-confound-floors](ppmi-mini-confound-floors.md) — measured floors, what shipped, and the two splits rejected as unfixable.

## Backbones

- [neurojepa-fixed-input-shape](neurojepa-fixed-input-shape.md) — 96x108x96 is mandatory; other shapes entangle the position axes.
- [neurojepa-monai-spacing-gpu](neurojepa-monai-spacing-gpu.md) — 96% of preprocessing, 37x on GPU, reordering is slower.
- [neurojepa-integration](neurojepa-integration.md) — fork, gated weights, deferred segmentation, MNI fidelity gap.
- [flash-attn-rejected](flash-attn-rejected.md) — the wheel installs; the `fused_dense_lib` extension doesn't ship with it.
- [neurovfm-torch-fallback-verified](neurovfm-torch-fallback-verified.md) — exact to 1.2e-6 in fp32; the fp32 residual accumulation is the subtle part.
- [sdpa-not-faster-at-real-token-counts](sdpa-not-faster-at-real-token-counts.md) — 1.6x at N=500, 0.9x at the real N=2000.
- [neurovfm-dependency-traps](neurovfm-dependency-traps.md) — eager `__init__`, `outlines==1.1.1`, torch_scatter off data.pyg.org.
- [neurovfm-integration-notes](neurovfm-integration-notes.md) — fork, arch, SimpleITK conversion, hardcoded bfloat16.
- [synthseg-numpy2-bugs](synthseg-numpy2-bugs.md) — shape-(1,) `np.where` into scalar slots, one of them silent.
- [synthseg-resample-length-fix](synthseg-resample-length-fix.md) — it is `F.interpolate`, but pad the source, not the output.
- [synthseg-no-crop-by-default](synthseg-no-crop-by-default.md) — deliberate match to upstream; the stale help string says otherwise.
- [synthseg-pooling-masks-padding](synthseg-pooling-masks-padding.md) — scan occupies 24-99% of the padded volume; mask before pooling.
- [synthseg-integration](synthseg-integration.md) — fork, cost profile, TF32 buys nothing, not an `nn.Module`.
- [smri-mae-axis-order](smri-mae-axis-order.md) — native RAS is right, the transpose was measured wrong, flag deleted.
- [smri-mae-checkpoint](smri-mae-checkpoint.md) — path, `mmap=True`, the `decoding` kwarg that must be filtered.
- [smri-mae-preprocessing-gap](smri-mae-preprocessing-gap.md) — the eval transform's brain mask is a stand-in; skull and neck survive.
- [nested-tensor-sdpa-needs-a-device](nested-tensor-sdpa-needs-a-device.md) — no CPU forward at all; test with a depth-0 encoder.

## Harness

- [memory-lives-in-the-repo](memory-lives-in-the-repo.md) — why memory is project-local, and what the trade-off is.
