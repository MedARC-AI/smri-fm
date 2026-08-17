"""Token poolings: the one thing tasks 6 and 7 actually let us tune.

The challenge withholds the labels for tasks 6 and 7 and fits its own probes, so
the whole submitted surface is `patch_embeds -> (D,)`. Everything here is a pure
function of the encoder's token output and its mask, which is what makes the
bench cheap: run the backbone once per subject, cache the tokens, then every
variant below is a numpy op over that cache.

Each pooling takes `(N, D)` tokens plus an `(N,)` boolean mask and returns a
1-D float32 embedding. Dimensionality is allowed to differ between poolings
(`mean_std` returns 2D) but must be fixed across subjects for a given pooling,
which `bench.py` asserts.

Why anything other than the mean is worth trying: task 1's signal is a small
infarct and task 2's is a single meningioma, so the informative tokens are a
handful out of thousands and the mean divides them by N. The order-statistic
poolings keep the peak; `gem` and `logsumexp` interpolate between mean and max
with one knob.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _masked(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The (M, D) real tokens. Raises rather than returning a silent zero vector,
    because an all-masked subject is a data bug and averaging it into a bench
    table would look like a pooling that merely scores badly."""
    tokens = np.asarray(tokens, dtype=np.float64)
    keep = np.asarray(mask).astype(bool).reshape(-1)
    if keep.size != tokens.shape[0]:
        raise ValueError(f"mask has {keep.size} entries for {tokens.shape[0]} tokens")
    if not keep.any():
        raise ValueError("every token is masked out; nothing to pool")
    return tokens[keep]


def mean(tokens, mask):
    """The current submission. Baseline for everything below."""
    return _masked(tokens, mask).mean(axis=0)


def max_(tokens, mask):
    """Peak response per channel. Keeps a focal signal the mean dilutes, at the
    cost of being the noisiest possible order statistic."""
    return _masked(tokens, mask).max(axis=0)


def topk_mean(tokens, mask, frac=0.05):
    """Mean of the strongest `frac` tokens per channel: `max_` with the variance
    averaged down. `frac=1` is `mean`, `frac->0` is `max_`."""
    t = _masked(tokens, mask)
    k = max(1, int(round(frac * t.shape[0])))
    if k >= t.shape[0]:
        return t.mean(axis=0)
    # Partition is O(N) per channel and enough, the top block needs no ordering.
    return np.partition(t, -k, axis=0)[-k:].mean(axis=0)


def gem(tokens, mask, p=3.0):
    """Generalised mean, (1/M sum x^p)^(1/p). p=1 is the mean, p->inf is the max.

    Defined on non-negative values, so it is applied to the positive part and the
    negative part separately and recombined. Clipping to the positive part alone
    would throw away half the token signal for a GELU-activated encoder.
    """
    t = _masked(tokens, mask)
    pos = np.power(np.clip(t, 0, None) + EPS, p).mean(axis=0) ** (1.0 / p)
    neg = np.power(np.clip(-t, 0, None) + EPS, p).mean(axis=0) ** (1.0 / p)
    return pos - neg


def logsumexp(tokens, mask, tau=1.0):
    """tau * log mean exp(x/tau). A smooth max: tau->0 is the max, large tau is
    the mean. Shifted by the per-channel max before exponentiating, so it does
    not overflow on a confident channel."""
    t = _masked(tokens, mask) / tau
    m = t.max(axis=0, keepdims=True)
    return (tau * (m + np.log(np.exp(t - m).mean(axis=0)))).reshape(-1)


def mean_std(tokens, mask):
    """Mean concatenated with per-channel spread, so a probe can use *how much*
    a channel varies across the volume and not only its average. Returns 2D."""
    t = _masked(tokens, mask)
    return np.concatenate([t.mean(axis=0), t.std(axis=0)])


def mean_topk(tokens, mask, frac=0.05):
    """Mean concatenated with the top-`frac` mean: keeps the global summary the
    baseline already scores well on and adds the focal signal beside it, rather
    than trading one for the other. Returns 2D."""
    return np.concatenate([mean(tokens, mask), topk_mean(tokens, mask, frac)])


#: The bench grid. Keep it small and interpretable: one baseline, the two
#: knobbed families at a low and a high setting, and the two concatenations.
#: A pooling only earns a place here if it is a plausible submission.
VARIANTS: dict = {
    "mean": (mean, {}),
    "max": (max_, {}),
    "topk_mean_p01": (topk_mean, {"frac": 0.01}),
    "topk_mean_p05": (topk_mean, {"frac": 0.05}),
    "topk_mean_p25": (topk_mean, {"frac": 0.25}),
    "gem_p3": (gem, {"p": 3.0}),
    "gem_p6": (gem, {"p": 6.0}),
    "logsumexp_t1": (logsumexp, {"tau": 1.0}),
    "logsumexp_t01": (logsumexp, {"tau": 0.1}),
    "mean_std": (mean_std, {}),
    "mean_topk_p05": (mean_topk, {"frac": 0.05}),
}


def apply(name: str, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fn, kwargs = VARIANTS[name]
    return np.asarray(fn(tokens, mask, **kwargs), dtype=np.float32)
