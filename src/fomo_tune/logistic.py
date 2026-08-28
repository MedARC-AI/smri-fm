"""Torch multi-output logistic regression at a fixed penalty.

Every output is an independent binary problem against a soft target, but they share the design
matrix, so one L-BFGS fit solves all of them at the cost of one.
"""

import torch
import torch.nn.functional as F


def fit_logistic(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    max_iter: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coefficients and intercept minimizing soft-target BCE with an L2 penalty.

    The penalty follows `ridge.solve`: summed loss plus `alpha` times the squared coefficients,
    intercept unpenalized, here divided through by the number of terms.
    """
    n, d = features.shape
    assert targets.ndim == 2 and len(targets) == n, (
        f"targets {tuple(targets.shape)} do not match {n} samples of {d} features"
    )
    n_outputs = targets.shape[1]

    coef = torch.zeros(d, n_outputs, device=features.device, dtype=features.dtype)
    intercept = torch.logit(targets.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    coef.requires_grad_(True)
    intercept.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [coef, intercept], max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = features @ coef + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + alpha * coef.square().sum() / (n * n_outputs)
        loss.backward()
        return loss

    optimizer.step(closure)

    # l-bfgs stops on its own gradient and step tolerances; exhausting the budget means neither met
    n_iter = optimizer.state[coef]["n_iter"]
    assert n_iter < max_iter, f"l-bfgs used all {max_iter} iterations without converging"
    return coef.detach(), intercept.detach()


class Logistic:
    """Torch multi-output logistic regression at a fixed penalty."""

    def __init__(self, alpha: float = 1e4, standardize: bool = True):
        self.alpha = alpha
        self.standardize = standardize

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> "Logistic":
        if self.standardize:
            self.mean_ = features.mean(dim=0)
            self.scale_ = features.std(dim=0, correction=0).clamp(min=1e-12)
        else:
            self.mean_ = features.new_zeros(features.shape[1])
            self.scale_ = features.new_ones(features.shape[1])

        scaled = (features - self.mean_) / self.scale_ if self.standardize else features
        coef, intercept = fit_logistic(scaled, targets, self.alpha)
        self.alpha_ = self.alpha
        self.coef_ = coef.T
        self.intercept_ = intercept
        return self

    def to(self, device: str | torch.device) -> "Logistic":
        self.coef_ = self.coef_.to(device)
        self.intercept_ = self.intercept_.to(device)
        self.mean_ = self.mean_.to(device)
        self.scale_ = self.scale_.to(device)
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        scaled = (features - self.mean_) / self.scale_ if self.standardize else features
        return torch.sigmoid(scaled @ self.coef_.T + self.intercept_)
