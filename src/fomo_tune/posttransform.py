"""Fixed transforms applied to the pooled embedding before it is shipped.

The second half of the tasks 6/7 surface, and the half nobody has tried. The
challenge contract is `nifti -> (D,) float32`; nothing requires D to be the
encoder width or the vector to be a raw pooling. Everything here is fitted on
UNLABELED images and applied as one matrix multiply at inference, so it costs
nothing against the 2-minute budget and needs no data beyond what the Methods
track already allows.

The motivation is the anisotropy literature from NLP -- Mu & Viswanath's
"all-but-the-top" (ICLR 2018) and the BERT-whitening line -- where frozen
transformer embeddings are dominated by a few high-variance directions that
encode nuisance rather than task signal, and where removing or rescaling them
reliably improves linear probes. The brain-MRI analogue of "word frequency" is
intensity scaling, head size and site, which is consistent with Connor's read
that tasks 1 and 5 sit at ceiling on "brain mask volume + brightness as proxy".

WHAT IS AND IS NOT ESTABLISHED. In simulation, against a faithful copy of the
challenge's own probe (multi-head SGD, lr sweep, val selection), reducing
1024 -> 128 dims moved paired disparity -0.023 [-0.039, -0.006] and probe macro
F1 +0.034 [+0.025, +0.042] over 60 seeds, both CIs excluding zero; 1024 -> 256
did nothing measurable. That says the mechanism can work, and that it improves
tasks 6 and 7 together rather than trading them off. It is NOT evidence about
the walnut embeddings: the simulation assumes the dominant variance directions
carry no class signal, and if the real top components carry signal, dropping
them will hurt. `bench.py` is what settles it on real features.

Fit on the pooled training embeddings only, then reuse -- `fit` returns a
callable that `bench.py` applies to train and test alike.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _svd_basis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, S, Vt


def identity(X: np.ndarray):
    """The current submission: ship the pooled vector as-is."""
    f = lambda Z: np.asarray(Z, dtype=np.float32)   # noqa: E731
    f.state = {"kind": "identity"}
    return f


def center(X: np.ndarray):
    """Subtract the training mean. Nothing else.

    Worth its own entry because the challenge's probe does not center: fomo-lp
    feeds the raw `.npy` straight to a linear head (`embedding_dataset.py`:
    "No image transforms are applied", `identity_model.py` is a pass-through),
    so whatever mean offset the embedding carries goes into the optimisation.
    Gradient descent on off-center features is badly conditioned, and the head
    has 20 epochs to recover, so an offset costs accuracy that never comes back.

    This is invisible to any bench whose own head standardises first, which is
    why it was missed on the first pass.
    """
    mu = np.asarray(X, dtype=np.float64).mean(axis=0)

    def f(Z):
        return (np.asarray(Z, dtype=np.float64) - mu).astype(np.float32)
    f.state = {"kind": "pca", "mu": mu, "V": np.eye(len(mu)), "scale": np.array([])}
    return f


def l2(X: np.ndarray):
    """Unit-norm each embedding. Removes the global scale factor that overall
    image brightness writes into every channel at once."""
    def f(Z):
        Z = np.asarray(Z, dtype=np.float64)
        return (Z / np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), EPS)).astype(np.float32)
    f.state = {"kind": "l2"}
    return f


def pca(X: np.ndarray, dim: int = 128, drop_top: int = 0, whiten: bool = False):
    """Project onto components `drop_top : drop_top + dim` of the training
    embeddings.

    `drop_top` is all-but-the-top: discard the leading directions outright.
    `whiten` rescales each kept component to unit variance, which is the
    stronger form -- it equalises the axes a linear probe sees, so no single
    high-variance nuisance direction dominates the gradient.
    """
    X = np.asarray(X, dtype=np.float64)
    mu, S, Vt = _svd_basis(X)

    # Keep only components the fit actually resolves. A fold's training set has
    # rank <= n_samples-1, so asking for more than that appends near-null
    # directions whose singular values are numerical dust; projecting onto them
    # and handing the result to a ridge head produced MAE ~1e15 in the bench.
    keep = int((S > S[0] * 1e-8).sum()) if S.size and S[0] > 0 else 0
    if keep == 0:
        raise ValueError("training embeddings have no resolvable variance")
    hi = min(drop_top + dim, keep) if dim else keep
    V = Vt[drop_top:hi]
    if V.shape[0] == 0:
        raise ValueError(
            f"drop_top={drop_top} dim={dim} leaves no components "
            f"({keep} resolvable in a {X.shape[0]}x{X.shape[1]} fit)")
    scale = None
    if whiten:
        n = max(X.shape[0] - 1, 1)
        sv = S[drop_top:hi] / np.sqrt(n)
        # Guard the divide: a component with ~zero variance would otherwise be
        # amplified to dominate every downstream distance.
        scale = np.maximum(sv, max(sv.max() * 1e-6, EPS))

    def f(Z):
        out = (np.asarray(Z, dtype=np.float64) - mu) @ V.T
        if scale is not None:
            out = out / scale
        return out.astype(np.float32)
    f.state = {"kind": "pca", "mu": mu, "V": V,
               "scale": scale if scale is not None else np.array([])}
    return f


#: The bench grid. `identity` is the incumbent; the rest are one axis each so a
#: win is attributable. Dims bracket the simulated effect (which appeared at
#: 128, not 256) and go below it, since the real optimum is unknown.
VARIANTS: dict = {
    "identity": (identity, {}),
    "center": (center, {}),
    "l2": (l2, {}),
    "pca_256": (pca, {"dim": 256}),
    "pca_128": (pca, {"dim": 128}),
    "pca_64": (pca, {"dim": 64}),
    "pca_128_whiten": (pca, {"dim": 128, "whiten": True}),
    "pca_128_drop2": (pca, {"dim": 128, "drop_top": 2}),
    "pca_64_drop5": (pca, {"dim": 64, "drop_top": 5}),
}


def fit(name: str, X_train: np.ndarray):
    """Fit on training embeddings, return the transform to apply to any split."""
    fn, kwargs = VARIANTS[name]
    return fn(X_train, **kwargs)


# --- persistence -------------------------------------------------------------
# A fitted transform has to survive into the container: the challenge hands
# predict.py one image at a time, so nothing can be estimated at inference.
# The closures above carry their parameters on `.state` for exactly this.


def save_state(f, path) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in f.state.items()})


def load_state(path):
    """Rebuild a fitted transform from `save_state`."""
    blob = np.load(path, allow_pickle=False)
    kind = str(blob["kind"])
    if kind == "identity":
        return identity(np.zeros((1, 1)))
    if kind == "l2":
        return l2(np.zeros((1, 1)))
    if kind != "pca":
        raise ValueError(f"unknown transform kind {kind!r}")

    mu, V = blob["mu"], blob["V"]
    scale = blob["scale"]
    scale = None if scale.size == 0 else scale

    def f(Z):
        out = (np.asarray(Z, dtype=np.float64) - mu) @ V.T
        if scale is not None:
            out = out / scale
        return out.astype(np.float32)
    f.state = {"kind": "pca", "mu": mu, "V": V,
               "scale": scale if scale is not None else np.array([])}
    return f
