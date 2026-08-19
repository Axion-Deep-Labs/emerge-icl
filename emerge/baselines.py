"""Closed-form Bayes-optimal baselines for in-context linear regression.

Both baselines predict y at position k from the k-1 preceding (x, y) pairs.
Position 0 has an empty context; the optimal prediction there is 0 under
either prior.

ridge_predictions: Bayes predictor under the continuous prior w ~ N(0, I).
dmmse_predictions: Bayes predictor under the discrete uniform prior over a
finite task pool (posterior-weighted mixture with Gaussian likelihood).
A trained model that memorized its pool tracks dMMSE on unseen tasks; a
model that learned the general algorithm tracks ridge.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ridge_predictions(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """x (B, K, d), y (B, K) -> ridge predictions (B, K).

    Posterior mean under w ~ N(0, I), noise N(0, sigma^2):
    w_hat = (X^T X + sigma^2 I)^{-1} X^T Y over the context prefix.
    """
    b, k, d = x.shape
    preds = torch.zeros(b, k, dtype=x.dtype, device=x.device)
    eye = torch.eye(d, dtype=x.dtype, device=x.device)
    for pos in range(1, k):
        xc, yc = x[:, :pos], y[:, :pos]
        gram = xc.transpose(1, 2) @ xc + (sigma**2) * eye
        rhs = torch.einsum("bkd,bk->bd", xc, yc)
        w_hat = torch.linalg.solve(gram, rhs)
        preds[:, pos] = (w_hat * x[:, pos]).sum(-1)
    return preds


@torch.no_grad()
def dmmse_predictions(
    x: torch.Tensor,
    y: torch.Tensor,
    pool: torch.Tensor,
    sigma: float,
    chunk: int = 64,
) -> torch.Tensor:
    """x (B, K, d), y (B, K), pool (M, d) -> dMMSE predictions (B, K).

    Posterior over pool tasks after k-1 pairs:
    log p_m proportional to -sum_{i<k} (y_i - w_m.x_i)^2 / (2 sigma^2).
    Prediction at x_k is the posterior-weighted mean of w_m.x_k.
    Batched in chunks because the (B, K, M) error tensor can be large.
    """
    b, k, d = x.shape
    preds = torch.zeros(b, k, dtype=x.dtype, device=x.device)
    pool = pool.to(x.device, x.dtype)
    for start in range(0, b, chunk):
        xs, ys = x[start : start + chunk], y[start : start + chunk]
        f = torch.einsum("bkd,md->bkm", xs, pool)
        sq = (ys.unsqueeze(-1) - f) ** 2
        cum = sq.cumsum(dim=1)
        # log-likelihood of the context strictly before each position
        loglik = torch.zeros_like(cum)
        loglik[:, 1:] = -cum[:, :-1] / (2 * sigma**2)
        post = torch.softmax(loglik, dim=-1)
        preds[start : start + chunk] = (post * f).sum(-1)
    return preds


@torch.no_grad()
def mse_per_position(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(B, K) predictions and targets -> per-position MSE (K,)."""
    return ((preds - targets) ** 2).mean(dim=0)
