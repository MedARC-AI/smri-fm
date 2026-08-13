# explore_fomo_task1

Task 1 scores AUROC 0.990 out of fold. What is that made of? This pairs each subject's
probability from `experiments/fomo_tune_baseline/output/task1/log.txt` with what is in its
images — the seg the 13 positives carry, and the acquisition geometry.

```bash
uv run python explore.py     # -> explore.tsv, figures/{scores,lesions,subjects}.png
```

## 1. The score is one discordant pair, and it is cross-site

104 pairs, AUROC 0.990 = 103/104. The single inversion is **sub-06** (positive, p=0.493) under
**sub-21** (negative, p=0.498). Both are on the borderline: the whole cohort sits in p ∈ [0.474,
0.536], the head is heavily regularized and nothing is confidently classified.

The 4th modality is `swi` on 16 subjects and `t2s` on sub-03..07, which is a site or protocol
split, and it lines up with the scores. **Within either site the ranking is perfect** — the one
error is a t2s positive losing to an swi negative:

| group | n | pos | AUROC | mean p, label 0 | mean p, label 1 |
|---|---|---|---|---|---|
| swi | 16 | 10 | 1.000 | 0.4848 | 0.5203 |
| t2s | 5 | 3 | 1.000 | 0.4875 | 0.4997 |

The offset is **one-sided**: the negatives are indistinguishable across sites (+0.003) and the
whole shift is in the positives (−0.021). But it cannot be attributed to site, because the three
t2s positives also have the larger brains (1439 mL against 1338 mL for the swi positives), and
`p` tracks brain volume — see §3. At n=3 the two explanations are collinear and this cohort
cannot separate them. What would: a t2s positive with a small brain, of which there is none.

## 2. The score does not track the infarct

This is the uncomfortable part. Across the positives:

- **Size**: spearman(p, lesion volume) = +0.35 (p=0.24) — nothing detectable at n=13, which is
  not the same as a demonstrated absence. The counterexample is sharper than the coefficient:
  sub-02 has a **98 mL** MCA territory infarct and scores 0.514, sub-14 has **0.8 mL** and scores
  0.536, the highest in the cohort. Three orders of magnitude, and p moves 0.02 the wrong way.
- **Conspicuity**: spearman(p, lesion mean DWI z) = +0.10. sub-06, the error, has the *second
  brightest* lesion (z=2.5) and the lowest positive p.
- **Location**: no visible pattern in centroid x/y/z (figure, bottom right). Both hemispheres,
  deep and cortical, all score alike.
- **Preprocessing is not the problem**: `retained = 1.000` for all 13 and `in_mask` ≥ 0.969, so
  no lesion is cropped off the 208x240x208 box or zeroed by the mean-intensity mask.

What p *does* track is **brain-mask volume**, spearman −0.589.

## 3. Non-backbone features get most of the way there

Leave-one-out logistic heads on scalars that need no model at all:

| features | AUROC |
|---|---|
| acquisition geometry (in-plane mm, slice mm, FOV x/z) | 0.490 |
| brain-mask volume, 1 scalar | 0.817 |
| DWI histogram (p99, p99.9, frac z>3, skew) | 0.827 |
| all of the above | **0.942** |
| the frozen MAE + logistic head | 0.990 |

Single scalars, with a label-permutation p:

| scalar | AUROC | sign | perm p |
|---|---|---|---|
| brain-mask volume | 0.827 | smaller ⇒ positive | 0.006 |
| DWI 99th pct | 0.769 | brighter ⇒ positive | 0.021 |
| DWI skew | 0.769 | +tail ⇒ positive | 0.022 |
| slice thickness | 0.712 | thinner ⇒ positive | 0.050 |

Two different things are in there and they should not be conflated:

**The DWI tail is real signal.** p99 and skew are elevated exactly in the subjects with large
bright infarcts (sub-13, sub-17), which is what a b1000 lesion does to a histogram. A dumb
detector finding the big lesions is a fair reading of the task.

**Brain volume is not.** Positives average 1361 mL of mask against 1505 mL for negatives, and
a single threshold on it reaches 0.827 with permutation p=0.006. Nothing about an acute infarct
should shrink the brain by 10% within hours; the plausible reading is age or atrophy, which is a
stroke risk factor and is therefore a genuine cohort confound rather than a pipeline bug. Also
note *pure* geometry scores 0.490, so this is not scanner metadata leaking — it is head size.

Caveat on the number: "brain-mask volume" is `(data > data.mean()).sum()` on the 1mm grid, the
transform's own mask. The task-1 images look skull-stripped so it is close to a brain volume, but
it is a proxy and inherits whatever the threshold does.

**But the model is not only brain volume.** Rank-residualize `p` against brain volume and the
remainder still scores **0.865**, so a large component of the ranking is something else that
these scalars do not capture. The correlation is also asymmetric — within positives rho = −0.51,
within negatives **+0.25** — so whatever it is does not act uniformly on the two classes.

And 0.942 against 0.990 is 6 discordant pairs against 1, at n=21. That difference is not
resolvable here. The defensible claim is that the dumb baselines are indistinguishable from the
model, which also forbids claiming the backbone adds nothing.

## 4. Reading the error against all of this

sub-06 has a 0.5 mL deep left corona radiata infarct, clearly visible on both DWI and ADC
(`figures/lesions.png`), fully retained by the transform. Nothing is wrong with the image. It
loses because it is (a) in the t2s block, and (b) has the second largest brain in the positive
class at 1527 mL, up among the negatives. Both of the things the model actually keys on are
pointed the wrong way for it, and its infarct — which is the thing the label is about — does not
outvote them.

sub-04, the next weakest positive (p=0.499), is the same story: t2s block, 1437 mL, and a 0.3 mL
lesion too small to move the histogram.

## Figures

- `scores.png` — p by label, and p against lesion volume, lesion DWI z, brain volume, and
  lesion centroid location. Colour is the label, marker is the 4th modality (square swi, circle
  t2s), on every panel but the location one.
- `lesions.png` — the 13 positives, DWI and ADC, cropped 100mm around the lesion, one acquired
  slice apart.
- `subjects.png` — all 21 subjects, DWI and ADC, 10 slices spanning the brain, seg in red and
  the transform's mask in cyan. This is what the backbone sees.

## What this suggests next

Task 1 is not leaking — the protocol is clean, and no lesion is lost in preprocessing. But 0.942
of the 0.990 is reachable without the backbone, most of the cohort-level separation is available
from head size, and the ranking is uncorrelated with lesion size, location and conspicuity. That
is much weaker evidence of infarct detection than 0.990 suggests.

Cheap next checks, in order:

1. **Regress brain volume out** — fit the head on features residualized against brain-mask
   volume, or report AUROC stratified by it. If the score collapses, the number is mostly a
   confound.
2. **PCA the 21 embeddings** and correlate the leading PCs with brain volume and with the
   swi/t2s split. Mean-pooled embeddings from this backbone are known to sit at cosine ~0.985
   even across different subjects and modalities, so the usable variance is all in the residual
   and it is worth knowing what the leading directions encode. Needs a GPU.
3. **Lesion-masked ablation** — zero the seg region in the DWI of positives and re-extract. If p
   does not drop, the head is provably not reading the infarct.
